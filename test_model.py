import unittest
import numpy as np
import matplotlib.pyplot as plt

from main_model import *

test_water = np.asarray(
    [
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1],
    ], dtype=np.bool)

test_height = np.asarray(
    [
        [0, 0, 0, 0],
        [0, 50, 50, 0],
        [0, 500, 1000, 0],
        [0, 0, 0, 0],
    ], dtype=np.int16)

def test_cells():
    return land_coords(test_water)

def test_polities():
    cells = test_cells()
    return new_polities(cells, 0)


class TestModel(unittest.TestCase):
    def test_invade_0_ultra(self):
        used = set()
        a_cell = (1, 1)
        d_cell = (1, 2)
        ultra = np.zeros((4, 4))
        mil = np.zeros((4, 4))
        polities = test_polities()
        polity_map = update_polity_values(polities, ultra, mil)
        elevation = test_height
        roll = .1

        invasion = invade_cells(used, a_cell, d_cell, polity_map, polities, elevation, roll)
        self.assertIsNone(invasion)

    def test_invade_1_ultra(self):
        used = set()
        a_cell = (1, 1)
        d_cell = (1, 2)
        ultra = np.zeros((4, 4))
        ultra[a_cell] = 1
        mil = np.zeros((4, 4))
        polities = test_polities()
        polity_map = update_polity_values(polities, ultra, mil)
        elevation = test_height
        roll = .1

        invasion = invade_cells(used, a_cell, d_cell, polity_map, polities, elevation, roll)
        self.assertIsNotNone(invasion)


if __name__ == '__main__':
    unittest.main()
