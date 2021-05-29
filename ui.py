import pygame
import numpy as np
import time
import sys
import os
import pygame_gui

from organism import Predator, Prey
from util import cart2pol, pol2cart

SCREEN_DIMS = (512,512)
ORGANISM_RADIUS = 3
GAME_FPS = 1/50
MUT_RATE = .9

REPLICATION_RATE = .0008
SEED = 2345

N_STARTING_PREY = 10
N_STARTING_PRED = 3

def main():

    np.random.seed(SEED)

    pygame.display.init()

    screen = pygame.display.set_mode(SCREEN_DIMS)

    manager = pygame_gui.UIManager((800, 600))

    prey = {
        Prey(init_pos=np.random.randint(0, 512, 2),
             init_vector=(0, 2 * np.random.random() * np.pi),
             color=np.random.randint(0, 255, 3)) for _ in range(N_STARTING_PREY)
    }

    predators = {
        Predator(init_pos=(np.random.randint(0, 512, 2)),
                 init_vector=(0, 2 * np.random.random() * np.pi),
                 color=np.array([255,0,0]))
        for _ in range(N_STARTING_PRED)
    }

    population = {*prey, *predators}
    for organism in population:
        organism.mutate()

    clock = pygame.time.Clock()
    running = True

    while running:

        time_delta = clock.tick(60) / 1000.0

        pygame.display.update()

        events = pygame.event.get()

        pygame.draw.rect(screen, [255,255,255], (0, 0, *SCREEN_DIMS))

        for organism in population:

            if not organism.alive:
                population = population.difference({organism})

                if organism.type == 'prey':
                    prey = prey.difference({organism})
                elif organism.type == 'predator':
                    predators = predators.difference({organism})

        for organism in population:
            pygame.draw.circle(screen, organism.color, organism.position, radius=ORGANISM_RADIUS)
            pygame.draw.line(screen, organism.tail_color, *organism.get_tail(), width=2)

        prey_offspring = set()
        for organism in prey:
            if np.random.random() < REPLICATION_RATE:
                prey_offspring = prey_offspring.union(organism.replicate(np.random.randint(1,4)))

        pred_offspring = set()

        for organism in population:
            if organism.type == "predator":
                pred_offspring = pred_offspring.union(organism.track_prey(prey))

            elif organism.type == "prey":
                organism.avoid_predators(predators)

        predators = predators.union(pred_offspring)
        prey = prey.union(prey_offspring)

        population = population.union(prey).union(predators)

        for organism in population:
            organism.avoid_walls()
            organism.move()


        time.sleep(GAME_FPS)

        for event in events:

            if event.type == pygame.QUIT:
                pygame.quit()

            manager.process_events(event)

        manager.draw_ui(screen)

if __name__ == "__main__":
    main()

