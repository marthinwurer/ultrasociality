import random

import numpy as np
import matplotlib.pyplot as plt

# constants
from tqdm import tqdm

n_ultra = 10  # number of ultrasocial traits
n_mil = 10  # number of military tech traits
power_coefficient = 1  # translates ultrasociality traits into the polity’s power
miltech_diffusion = 0.25  # probability of MilTech diffusing to a nearby cell
e_min = 1/20  # baseline probability of ethnocide
e_max = 1  # maximum probability of ethnocide
disintegration_base = 0.05  # disintegration probability
disintegration_size = 0.05  # disintegration size coefficient
disintegration_ultra = 2  # ultrasocial trait stability bonus
mutation_gain = 0.0001  # probability that a cell will gain an ultrasocial trait
mutation_loss = 0.002  # probability that a cell will lose an ultrasocial trait
distance_sea = 1  # base distance of sea-based attack
d_sea_gain = 0.25  # increment of sea distance each time step
dt = 2 # number of years per timestep
t_start = -1500  # Start year of the simulation
t_end = 1500 # End year of the simulation
# t_end = -900 # End year of the simulation
attack_prob = 0.5  # probablilty that an attack will occur
height_coefficient = 4  # coefficient translating elevation into defensive power

x_dim = 360
y_dim = 180
world_shape = (y_dim, x_dim)

neighbors = np.zeros((y_dim, x_dim, 4, 2), dtype=np.int)

def unpack_bits(a):
    np.unpackbits(a.view(np.uint8))


def shuffled(l):
    out = l.copy()
    random.shuffle(out)
    return out


class Polity(object):
    def __init__(self, name, cells):
        self.name = name
        self.cells = cells
        self.average_u = 0
        self.average_m = 0


def new_polities(cells, year):
    polities = [Polity("%s %s %s" % (y, x, year), [(y, x)]) for y, x in cells]
    return polities



def land_coords(water):
    for y in range(water.shape[0]):
        for x in range(water.shape[1]):
            if not water[y, x]:
                yield y, x

def wrap_coord(coord, max_x, max_y):
    y, x = coord
    return


def gen_neighbors():
    for y in range(neighbors.shape[0]):
        for x in range(neighbors.shape[1]):
            north = (y - 1, x)
            south = (y + 1, x)
            east = (y, x + 1)
            west = (y, x - 1)
            coords = [north, south, east, west]
            for i, (c_y, c_x) in enumerate(coords):
                neighbors[y, x, i] = [c_y % y_dim, c_x % x_dim]

def random_neighbor(cell):
    direction = random.randrange(4)
    other_cell = tuple(neighbors[cell[0], cell[1], direction])
    return other_cell


def main():
    gen_neighbors()
    ultrasocial = np.zeros((y_dim, x_dim, n_ultra), dtype=np.bool)
    military = np.zeros((y_dim, x_dim, n_mil), dtype=np.bool)
    year = t_start

    # load datasets
    water = np.load("./data/underwater_mask.npy")
    agriculture = np.load("./data/agriculture.npy")
    steppes = np.load("./data/steppes.npy")
    height = np.load("./data/height.npy")
    stddev = np.load("./data/stddev.npy")

    # generate polities
    cells = list(land_coords(water))
    polities = new_polities(cells, t_start)

    mil_seed_mask = steppes & (agriculture == False)
    military[mil_seed_mask] = True

    # turchin assumes that only agricultural tiles do operations.
    # I will instead assume that all tiles do operations, but that only agricultural tiles have ultrasocial traits.
    for year in tqdm(range(t_start, t_end + 1, dt)):
        # calculate community values
        nu = np.sum(ultrasocial, axis=-1)
        nm = np.sum(military, axis=-1)

        polity_map = update_polity_values(polities, nu, nm)

        np.save("./outputs/polity_map_%s.npy" % (year,), polity_map)
        np.save("./outputs/ultra_%s.npy" % (year,), nu)
        np.save("./outputs/military_%s.npy" % (year,), nm)

        # warfare
        invasions = warfare(polities, polity_map, nu, nm, cells, height)

        # diffusion
        diffusion(military, cells)
        # ethnocide
        ethnocide(invasions, ultrasocial, nm, height)
        # evolution
        evolution(ultrasocial)
        ultrasocial[agriculture == False] = False  # reset ultrasocial traits in non-agricultural locations
        # polity disintegration
        next_polities = disintegration(polities, invasions, year)

        polities = next_polities


def update_polity_values(polities, ultra, mil):
    # build polity identity map
    polity_map = np.full(world_shape, fill_value=-1, dtype=np.int16)
    for identity, polity in enumerate(polities):
        total_u = 0
        total_m = 0
        for cell in polity.cells:
            polity_map[cell] = identity
            total_u += ultra[cell]
            total_m += mil[cell]

        polity.average_u = total_u / len(polity.cells)
        polity.average_m = total_m / len(polity.cells)

    return polity_map



def warfare(polities, polity_map, ultra, mil, cells, elevation):
    used = set()
    invasions = []
    for cell in shuffled(cells):
        if cell in used:
            continue  # each cell can invade or be invaded only once
        used.add(cell)
        own_polity = polity_map[cell]
        other_cell = random_neighbor(cell)
        other_polity = polity_map[other_cell]
        # check if invasion happens
        if (own_polity == other_polity or  # if they're the same polity, no attack is made
                other_polity == -1 or  # -1 is an invalid polity
                other_cell in used  # can't attack a cell that's been already used
                # or random.random() > attack_prob
                ):  # attack chance
            continue

        a_polity = polities[own_polity]
        d_polity = polities[other_polity]

        p_att = 1 + power_coefficient * a_polity.average_u * len(a_polity.cells)
        p_def = 1 + power_coefficient * d_polity.average_u * len(d_polity.cells) +\
                    height_coefficient * (elevation[other_cell] / 1000)  # divide by 1k because it's supposed to be in Km

        p_success = (p_att - p_def) / (p_att + p_def)

        # check if invasion succeeds
        if random.random() > p_success:
            continue  # Attack was not successful

        used.add(other_cell)
        invasions.append((cell, other_cell, own_polity, other_polity))



    return invasions


def diffusion(miltech, cells):
    # At each time step, the model samples all agricultural cells and randomly selects a particular locus
    for cell in shuffled(cells):
        tech = miltech[cell]
        locus = random.randrange(n_mil)
        trait = tech[locus]
        # If the value of the trait at this locus is 0, nothing happens.
        if trait == 0:
            continue
        else:
            # the simulation randomly chooses
            # one of four directions and checks whether the neighbor cell in this direction
            # has the particular technology trait
            other_cell = random_neighbor(cell)
            if random.random() < miltech_diffusion:
                miltech[other_cell][locus] = 1


def ethnocide(invasions, ultrasocial, mil, elevation):
    for a_cell, d_cell, _, _ in invasions:
        p_ethnocide = e_min + (e_max - e_min) * (mil[a_cell] / n_mil) -\
                      height_coefficient * (elevation[d_cell] / 1000)

        if p_ethnocide > 0 and random.random() < p_ethnocide:
            ultrasocial[d_cell] = ultrasocial[a_cell]


def evolution(ultra):
    keep = np.random.random(ultra.shape) > mutation_loss
    gain = np.random.random(ultra.shape) < mutation_gain

    ultra &= keep
    ultra |= gain


def disintegration(polities, invasions, year):
    next_polities = []

    # do the invasions in place.
    # Edge cases:
    #   one tile gets invaded multiple times. Fix this with a used set: Done
    #   invading polity disintegrates: compute after invasions
    #   invasion pushes polity over the edge of disintegration: I think missing this is acceptable
    for a_cell, d_cell, a_polity, d_polity in invasions:
        a_polity = polities[a_polity]
        d_polity = polities[d_polity]
        a_polity.cells.append(d_cell)
        d_polity.cells.remove(d_cell)

    # do disintegraton calcs
    for polity in polities:
        size = len(polity.cells)
        if size == 0:
            continue
        elif size == 1:
            # if there's only one community, it will not fall apart
            next_polities.append(polity)
        else:
            prob = disintegration_base + disintegration_size * size +\
                        disintegration_ultra * polity.average_u
            if prob < random.random():
                next_polities.append(polity)
            else:
                new = new_polities(polity.cells, year)
                next_polities.extend(new)

    return next_polities













if __name__ == '__main__':
    main()