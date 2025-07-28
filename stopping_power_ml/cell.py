#!/usr/bin/env python
# coding=utf-8
import numpy as np
from pymatgen.core.structure import Structure
from stopping_power_ml.rc import *
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from scipy import linalg

class Cell:
    """
    extract the crystal information that will be useful for later simulation
    """
    def __init__(self, atoms):
        """Create the class
        :param atoms: ase.Atoms, crystal structure being studied (including particle as the last atom)
        """
        # Store the main details
        self.atoms = atoms

        # Compute the primitive unit cell vectors
        self.simulation_cell = AseAtomsAdaptor.get_structure(atoms)
        spg = SpacegroupAnalyzer(self.simulation_cell)
        self.prim_strc = spg.find_primitive()
        self.conv_strc = spg.get_conventional_standard_structure()

        self.simulation_to_conv = np.round(linalg.solve(self.conv_strc.lattice.matrix, self.simulation_cell.lattice.matrix))

        # Compute the matrix that we will use to map lattice vectors in the conventional cell to ones in the primitive
        self.conv_to_prim = np.round(linalg.solve(self.prim_strc.lattice.matrix, self.conv_strc.lattice.matrix))

        logging.info(f"Simulation cell (bohr)\n {self.simulation_cell}")
        logging.info(f"Simulation cell lattice vectors\n {self.simulation_cell.lattice.matrix}")
        logging.info(f"Conventional unit cell (bohr)\n {self.conv_strc}")
        logging.info(f"Conventional cell lattice vectors\n {self.conv_strc.lattice.matrix}")
        logging.info(f"Primitive cell (bohr)\n {self.prim_strc}")
        logging.info(f"Primitive cell lattice vectors\n {self.prim_strc.lattice.matrix}")
        logging.info(f"Simulation cell to conventional unit cell conversion\n {self.simulation_to_conv}")
        logging.info(f"Conventional unit cell to primitive cell conversion\n {self.conv_to_prim}")

    def simulation_cell_lattice(self):
        return self.simulation_cell.lattice.matrix
    
    def conventional_cell_lattice(self):
        return self.conv_strc.lattice.matrix

    def primitive_cell_lattice(self):
        return self.prim_strc.lattice.matrix

    def cartesian_to_conventional(self, pos):
        return self.conv_strc.lattice.get_fractional_coords(pos)

    def conventional_to_cartesian(self, pos):
        return self.conv_strc.lattice.get_cartesian_coords(pos)

    def simulation_to_cartesian(self, pos):
        return self.simulation_cell.lattice.get_cartesian_coords(pos)

    def cartesian_to_simulation(self, pos):
        return self.simulation_cell.lattice.get_fractional_coords(pos)

    def primitive_to_cartesian(self, pos):
        return self.prim_strc.lattice.get_cartesian_coords(pos)

    def cartesian_to_primitive(self, pos):
        return self.prim_strc.lattice.get_fractional_coords(pos)

