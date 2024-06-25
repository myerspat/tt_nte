import numpy as np


class Geometry:
    def __init__(self, regions, left_bc, right_bc):
        self._regions = np.array(regions)
        self._left_bc = left_bc
        self._right_bc = right_bc

        self._unique_regions = set(self._regions)

        assert self._left_bc == "vacuum" or self._left_bc == "reflective"
        assert self._right_bc == "vacuum" or self._right_bc == "reflective"

    def region_mask(self, r):
        mask = np.zeros((self.num_nodes, 1), dtype=float)

        idx = 0
        for region in self._regions:
            if r == region:
                mask[idx : idx + region.num_nodes, 0] = 1

            idx += region.num_nodes

        return mask

    # ===================================================
    # Getters

    @property
    def regions(self):
        return self._regions

    @property
    def unique_regions(self):
        return self._unique_regions

    @property
    def left_bc(self):
        return self._left_bc

    @property
    def right_bc(self):
        return self._right_bc

    @property
    def num_nodes(self):
        return sum([region.num_nodes for region in self._regions])

    @property
    def dx(self):
        dx = np.zeros((self.num_nodes, 1), dtype=float)

        idx = 0
        for region in self._regions:
            dx[idx : idx + region.num_nodes, 0] = region.dx

            idx += region.num_nodes

        return dx
