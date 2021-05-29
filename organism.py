import pygame
import numpy as np
from math import pi, sqrt
import time

from util import cart2pol, pol2cart, sum_polars, sigmoid

GLOBAL_SPEED_FACTOR = .05
GLOBAL_SPEED_LIMIT = 40

# DIVISION_RATE = 2
TARGET_FORCE_FACTOR = 1/100
WALL_FORCE_FACTOR = 1/50
HERD_FORCE_FACTOR = 1/20
SCREEN_DIMS = (512, 512)
WALL_FORCE_DECAY = 1.
TARGET_FORCE_DECAY = 1.
HERD_FORCE_DECAY = 1.
STEP_SIZE = .05
DEATH_RATE = .0006
PREDATOR_SPEED_ADVANTAGE = 5

TRACK_LIMIT = 1000
KILL_RANGE = 5
N_DIVISIONS = 2

GENOME = {
    "TARGET_FORCE_FACTOR": TARGET_FORCE_FACTOR,
    "WALL_FORCE_FACTOR": WALL_FORCE_FACTOR,
    "TARGET_FORCE_DECAY": TARGET_FORCE_DECAY,
    "WALL_FORCE_DECAY": WALL_FORCE_DECAY,
    "HERD_FORCE_FACTOR":  HERD_FORCE_FACTOR,
    "HERD_FORCE_DECAY": HERD_FORCE_DECAY,
    # "COLOR": np.random.randint(0,256,3)
}

"""
OTHER PROPERTIES TO MAKE EVOLUTIONARY 
- AVOID OTHER PREDATORS
- AVOID OTHER PREY
- N TARGETS TRACKED (TOP N)

"""


class Organism:

    def __init__(self, init_pos, init_vector, color, init_gene=GENOME):

        self.alive = True
        self.dob = time.time()

        self.position = np.array(init_pos)
        self.vector = np.array(init_vector)  # reverse vector to make tail

        self.gene = init_gene

        self.speed_factor = GLOBAL_SPEED_FACTOR
        self.speed_limit = GLOBAL_SPEED_LIMIT
        self.death_rate = .0

        self.color = color
        self.tail_color = [(v - 122.5) * .8 + 122.5 for v in self.color]

    def move(self):

        translation = self.vector
        translation[0] = np.clip(translation[0], 0., self.speed_limit)

        self.position = self.position + self.speed_factor * pol2cart(*translation)
        # self.position = np.array([np.clip(self.position[0], 1., SCREEN_DIMS[0]-2),
        #                           np.clip(self.position[1], 1., SCREEN_DIMS[1]-2)])

        if np.random.random() < self.death_rate or \
                not 1 < self.position[0] < SCREEN_DIMS[0] - 1 or \
                not 1 < self.position[1] < SCREEN_DIMS[1] - 1:
            self.kill()

    def mutate(self):

        if not self.type == "predator":
            self.color = sigmoid((self.color - 127.5) / 255. + .05 * np.random.random() - .5) * 255.
        
        for name, allele in self.gene.items():
            self.gene[name] = (allele + STEP_SIZE * np.random.normal() * self.gene[name])

    def avoid_walls(self):
        # l r b t

        wall_points = [np.array(c) for c in [[0., self.position[1]], [SCREEN_DIMS[0], self.position[1]],
                                             [self.position[0], 0.], [self.position[0], SCREEN_DIMS[1]]]]

        self._track_targets(wall_points,
                            force_factor=self.gene["WALL_FORCE_FACTOR"],
                            force_decay=self.gene["WALL_FORCE_DECAY"],
                            mode="avoid")

    def get_tail(self, scale=.5):

        tail_coordinates = self.position + -pol2cart(scale * self.vector[0], self.vector[1])

        return self.position, tail_coordinates

    def _track_targets(self, targets, force_factor, force_decay, n=TRACK_LIMIT, mode="approach"):


        vectors = np.array([cart2pol(*self._get_tracking_vector(target)) for target in targets])

        if len(vectors) > 0:
            vectors = vectors[np.argsort(vectors[:, 0])]
        else:
            return

        sign = 1 if mode == "approach" else -1

        for i, vector in enumerate(vectors):

            vector[0] = (sign * (1/force_factor) /
                         vector[0] ** force_decay)

            self._apply_force(vector)

            if i >= n:
                break

    def _apply_force(self, vector):
        self.vector = sum_polars(self.vector, vector)

        return self.vector

    def _get_tracking_vector(self, coordinates):
        return coordinates - self.position

    def herd_behaviour(self, targets):
        "Maintain a fixed distance from herd members. (optimize distance to neighbours)"

        force_transformation = lambda x: - self.gene["HERD_FORCE_FACTOR"] / x ** self.gene["HERD_FORCE_DECAY"] + 1

        vectors = [force_transformation(self._get_tracking_vector(target)) for target in targets]

        for vector in vectors:
            self._apply_force(vector)




    def replicate(self, n_divisions, mut_rate=.01):

        children = {self.__class__(init_pos=self.position + np.random.random(2) * 10,
                                   init_vector=(self.vector[0], a), color=self.color)
                    for a in np.linspace(0, 2 * np.pi, n_divisions)}

        for child in children:
            if np.random.random() < mut_rate:
                child.mutate()

        self.kill()

        return children

    def kill(self):
        self.alive = False


class Predator(Organism):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.speed_limit += 5
        self.type = "predator"
        self.death_rate = DEATH_RATE

        if "color" not in kwargs:
            self.color = (255., 0., 0.)

    def track_prey(self, prey):

        self._track_targets([p.position for p in prey],
                            force_factor=self.gene["TARGET_FORCE_FACTOR"],
                            force_decay=self.gene["TARGET_FORCE_DECAY"],
                            n=1,
                            mode="approach")
        for p in prey:
            if cart2pol(*self._get_tracking_vector(p.position))[0] < KILL_RANGE:

                offspring = self.replicate(n_divisions=2)

                p.kill()

                return offspring

        return set()


class Prey(Organism):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.type = "prey"

    def avoid_predators(self, predators):
        self._track_targets([p.position for p in predators],
                            force_factor=self.gene["TARGET_FORCE_FACTOR"],
                            force_decay=self.gene["TARGET_FORCE_DECAY"],
                            mode="avoid")







