from typing import List, Optional

import numpy as np
import torch

from ttnte.cad._patch import plot as _patch_plot
from ttnte.parallel import Communicator, DataType, MPITag


class _StubLabel:
    """Duck-typed stand-in for a ttnte Label/MaterialLabel object, backed by values
    received from another rank -- see GatheredPatch."""

    def __init__(self, int_value: int, str_value: str):
        self._int = int_value
        self._str = str_value

    def to_int(self) -> int:
        return self._int

    def to_string(self) -> str:
        return self._str


class GatheredPatch:
    """Duck-typed stand-in for a ttnte.cad.Patch, backed by data gathered from the rank
    that actually owns it. Carries just enough (control points, a pre-evaluated plot-
    resolution grid, bounding box, fill/label) for `patch_slice()`/`Patch.plot()` to
    render it exactly like a live Patch, without this rank ever holding the owning
    rank's full geometry/DOF state -- everything here is bounded by plot resolution or
    by that patch's own (small) control-point count, never by problem size.

    `get_ctrlpts_size()` always reports 1 regardless of the real
    control-point grid shape: the only caller, `patch_slice()`'s field
    handling, uses it solely to reshape a `field` argument before calling
    `evaluate_field()` -- and `evaluate_field()` here ignores that argument
    entirely (the field was already evaluated on the owning rank), so any
    reshape target works.
    """

    def __init__(
        self,
        ndim: int,
        fill: _StubLabel,
        label: _StubLabel,
        bbox: torch.Tensor,
        ctrlpts: torch.Tensor,
        points: torch.Tensor,
        field: Optional[torch.Tensor],
    ):
        self.ndim = ndim
        self.fill = fill
        self.label = label
        self.device = torch.device("cpu")
        self.dtype = torch.float64
        self.ctrlpts = ctrlpts
        self.has_field = field is not None
        self.field_placeholder = torch.zeros(1, dtype=torch.float64)
        self.field_values = field
        self._bbox = bbox
        self._points = points

    def get_bbox(self, epsilon: float = 0.0) -> torch.Tensor:
        return self._bbox

    def get_ctrlpts_size(self, dim: int) -> int:
        return 1

    def __call__(self, tensor_product_pts) -> torch.Tensor:
        # Ignores its argument -- already evaluated at exactly this
        # resolution on the owning rank.
        return self._points

    def evaluate_field(self, field: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        # Ignores both arguments for the same reason.
        return self.field_values.reshape(-1, 1)


# Reuse Patch.plot()'s exact rendering logic (it's written generically
# against a duck-typed `self`, never `isinstance`-checked) rather than
# special-casing GatheredPatch at every call site.
GatheredPatch.plot = _patch_plot


def _send_patch(
    comm: Communicator,
    patch,
    resolution: int,
    field: Optional[torch.Tensor],
    target_rank: int,
) -> None:
    """Evaluate `patch`'s own geometry (and `field`, if given) at `resolution`, and send
    the result to `target_rank`.

    Local, cheap
    (bounded by `resolution`/this patch's own control-point count): no raw
    DOF state or full-resolution geometry ever crosses ranks.
    """
    ndim = patch.ndim
    ctrlpts = patch.ctrlpts.to(torch.float64).cpu().contiguous()
    phys_dim = ctrlpts.shape[-1]
    fill_str = patch.fill.to_string()
    label_str = patch.label.to_string()

    tensor_product_pts = ndim * [
        torch.linspace(0, 1, resolution, device=patch.device, dtype=patch.dtype)
    ]
    points = patch(tensor_product_pts).to(torch.float64).cpu().contiguous()

    field_values = None
    if field is not None:
        ctrlpts_shape = [patch.get_ctrlpts_size(d) for d in range(ndim)]
        reshaped = field.reshape(*ctrlpts_shape, -1)
        mesh_coords = torch.meshgrid(tensor_product_pts, indexing="ij")
        flat_points = torch.stack([c.flatten() for c in mesh_coords], dim=-1)
        field_values = (
            patch.evaluate_field(reshaped, flat_points)[..., 0]
            .reshape(ndim * [resolution])
            .to(torch.float64)
            .cpu()
            .contiguous()
        )

    bbox = patch.get_bbox().to(torch.float64).cpu().contiguous()

    ctrlpts_shape_padded = list(ctrlpts.shape[:-1]) + [0] * (3 - ndim)
    # fill.to_int()/label.to_int() are 64-bit hashes that can exceed
    # signed int64's range -- build the header as uint64 (which holds any
    # such value) and bit-reinterpret as int64 for the wire (Communicator
    # only supports signed integer DataTypes); the receiver reverses this,
    # recovering the exact original value.
    header = (
        torch.tensor(
            [
                ndim,
                phys_dim,
                1 if field_values is not None else 0,
                len(fill_str),
                len(label_str),
                patch.fill.to_int(),
                patch.label.to_int(),
                *ctrlpts_shape_padded,
            ],
            dtype=torch.uint64,
        )
        .view(torch.int64)
        .contiguous()
    )
    comm.send(
        header.data_ptr(),
        header.numel(),
        target_rank,
        MPITag.SOLUTION_GATHER_SIZE,
        DataType.INT64,
    )

    payload_parts = [
        torch.tensor([float(ord(c)) for c in fill_str], dtype=torch.float64),
        torch.tensor([float(ord(c)) for c in label_str], dtype=torch.float64),
        bbox.flatten(),
        ctrlpts.flatten(),
        points.flatten(),
    ]
    if field_values is not None:
        payload_parts.append(field_values.flatten())
    payload = torch.cat(payload_parts).contiguous()
    comm.send(
        payload.data_ptr(),
        payload.numel(),
        target_rank,
        MPITag.SOLUTION_GATHER_DATA,
        DataType.DOUBLE,
    )


def _recv_patch(comm: Communicator, source_rank: int, resolution: int) -> GatheredPatch:
    """Receive one patch's gathered data from `source_rank` -- the receive-side
    counterpart of `_send_patch()`."""
    header = torch.zeros(10, dtype=torch.int64)
    comm.recv(
        header.data_ptr(),
        header.numel(),
        source_rank,
        MPITag.SOLUTION_GATHER_SIZE,
        DataType.INT64,
    )
    # Reverse the uint64->int64 bit-reinterpretation _send_patch() applied,
    # recovering fill_int/label_int's exact original (possibly-huge) value.
    header = header.view(torch.uint64)
    (
        ndim,
        phys_dim,
        has_field,
        fill_len,
        label_len,
        fill_int,
        label_int,
        s0,
        s1,
        s2,
    ) = header.tolist()

    ctrlpts_shape = [s0, s1, s2][:ndim]
    ctrlpts_numel = int(np.prod(ctrlpts_shape)) * phys_dim
    points_numel = (resolution**ndim) * phys_dim
    field_numel = resolution**ndim if has_field else 0
    total = (
        fill_len + label_len + 2 * phys_dim + ctrlpts_numel + points_numel + field_numel
    )

    payload = torch.zeros(total, dtype=torch.float64)
    comm.recv(
        payload.data_ptr(),
        payload.numel(),
        source_rank,
        MPITag.SOLUTION_GATHER_DATA,
        DataType.DOUBLE,
    )

    idx = 0
    fill_str = "".join(
        chr(int(round(x))) for x in payload[idx : idx + fill_len].tolist()
    )
    idx += fill_len
    label_str = "".join(
        chr(int(round(x))) for x in payload[idx : idx + label_len].tolist()
    )
    idx += label_len
    bbox = payload[idx : idx + 2 * phys_dim].reshape(2, phys_dim).clone()
    idx += 2 * phys_dim
    ctrlpts = (
        payload[idx : idx + ctrlpts_numel].reshape(*ctrlpts_shape, phys_dim).clone()
    )
    idx += ctrlpts_numel
    points = (
        payload[idx : idx + points_numel]
        .reshape(*(ndim * [resolution]), phys_dim)
        .clone()
    )
    idx += points_numel
    field = None
    if has_field:
        field = payload[idx : idx + field_numel].reshape(ndim * [resolution]).clone()

    return GatheredPatch(
        ndim=ndim,
        fill=_StubLabel(fill_int, fill_str),
        label=_StubLabel(label_int, label_str),
        bbox=bbox,
        ctrlpts=ctrlpts,
        points=points,
        field=field,
    )


def gather_mesh_patches(
    mesh,
    resolution: int,
    solution=None,
    root: int = 0,
) -> List:
    """Gather every patch across all MPI ranks into a single list of (live or
    `GatheredPatch`) objects, for whole-mesh compositing on `root`. Always collective --
    every rank must call this. Non-root ranks return an empty list.

    No raw geometry or DOF state ever crosses ranks -- each patch
    contributes only its own already-evaluated plot-resolution grid
    (bounded by `resolution`, not by DOF count) plus its own (small)
    control-point array.

    Parameters
    ----------
    mesh: ttnte.mesh.IGAMesh
        This rank's own (already-culled, local-only) mesh. `mesh.gid2rank`
        (GID -> owning rank, for every GID in the whole mesh) is read
        directly from it -- always populated, trivially if the mesh was
        never distributed across ranks.
    resolution: int
        The number of parametric points to evaluate along a parametric
        dimension -- must match what the caller will pass to `plot()`.
    solution: ttnte.driver.TransportSolution, optional
        If given, each patch's field (via `solution.get_local_field(gid)`)
        is gathered alongside its geometry. Must be spatial-only (the
        result of `compute_scalar_flux()`).
    root: int, default=0
        The rank that receives every patch and gets the combined list.

    Returns
    -------
    patches: list of ttnte.cad.Patch or GatheredPatch
        On `root`, every patch in the mesh, in ascending GID order. On any
        other rank, an empty list.
    """
    comm = Communicator.world()
    rank = comm.rank()
    gid2rank = mesh.gid2rank

    local_patches = {patch.gid: patch for patch in mesh.blocks}

    if rank != root:
        for gid in sorted(local_patches):
            patch = local_patches[gid]
            field = (
                solution.get_local_field(gid).to_dense()
                if solution is not None
                else None
            )
            _send_patch(comm, patch, resolution, field, root)
        return []

    patches = []
    for gid in sorted(gid2rank):
        owner = gid2rank[gid]
        if owner == root:
            patches.append(local_patches[gid])
        else:
            patches.append(_recv_patch(comm, owner, resolution))
    return patches
