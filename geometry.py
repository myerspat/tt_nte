import numpy as np

class Geometry:
    def __init__(self, regions, left_bc, right_bc, 
                 # top_bc, bottom_bc, 
                 # into_bc, outof_bc
                     ):
        self._regions = np.array(regions)
        self._left_bc = left_bc
        self._right_bc = right_bc
        # self._bottom_bc = bottom_bc
        # self._top_bc = top_bc
        # self._into_bc = into_bc
        # self._outof_bc = outof_bc

        self._unique_regions = set(self._regions)

        assert self._left_bc == "vacuum" or self._left_bc == "reflective"
        assert self._right_bc == "vacuum" or self._right_bc == "reflective"
        # assert self._top_bc == "vacuum" or self._top_bc == "reflective"
        # assert self._bottom_bc == "vacuum" or self._bottom_bc == "reflective"
        # assert self._into_bc == "vacuum" or self._into_bc == "reflective"
        # assert self._outof_bc == "vacuum" or self._outof_bc == "reflective"

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
    def bottom_bc(self):
        return self._bottom_bc

    @property
    def top_bc(self):
        return self._top_bc
    
    @property
    def into_bc(self):
        return self._into_bc

    @property
    def outof_bc(self):
        return self._outof_bc

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
    
    @property
    def dy(self):
        dy = np.zeros((self.num_nodes, 1), dtype=float)

        idy = 0
        for region in self._regions:
            dy[idy : idy + region.num_nodes, 0] = region.dy

            idy += region.num_nodes
        
        return dy
    
    @property
    def dz(self):
        dz = np.zeros((self.num_nodes, 1), dtype=float)

        idz = 0
        for region in self._regions:
            dz[idz : idz + region.num_nodes, 0] = region.dz

            idz += region.num_nodes
        
        return dz
        
