class Region:
    def __init__(self, material, thickness, num_nodes):
        self._material = material
        self._thickness = thickness
        self._num_nodes = num_nodes

    # ===================================================
    # Getters

    @property
    def material(self):
        return self._material

    @property
    def thickness(self):
        return self._thickness

    @property
    def num_nodes(self):
        return self._num_nodes

    @property
    def dx(self):
        return self._thickness / self._num_nodes

    # ===================================================
    # Setters

    @num_nodes.setter
    def num_nodes(self, num_nodes):
        self._num_nodes = num_nodes
