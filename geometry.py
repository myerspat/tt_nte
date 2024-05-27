class Geometry:
    def __init__(self, materials):
        self._materials = materials

    def mat_thickness(self, mat):
        return self._materials[mat]["thickness"]

    def mat_num_nodes(self, mat):
        return self._materials[mat]["num_nodes"]

    def mat_dx(self, mat):
        return self.mat_thickness(mat) / self.mat_num_nodes(mat)

    @property
    def materials(self):
        return list(self._materials.keys())
