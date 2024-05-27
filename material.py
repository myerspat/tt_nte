class Material:
    def __init__(self, nuclides):
        self._nuclides = nuclides

    def atom_frac(self, nuc):
        return self._nuclides[nuc]

    @property
    def nuclides(self):
        return list(self._nuclides.keys())
