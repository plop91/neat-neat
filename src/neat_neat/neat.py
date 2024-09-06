from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

import numpy as np

from genome import *
from utils import *

import os
import shutil
import time
import multiprocessing as mp

import warnings
warnings.filterwarnings("ignore")


def process_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (32, 32))
    img = img.flatten()
    c_floats = (ctypes.c_float * len(img))(*img)
    return c_floats


class Neat:

    def __init__(self, max_population=100, gui=False):
        self.max_population = max_population  # Number of genomes in the population
        self.population = []  # List of genomes
        self.gui = gui

    def create_population(self):
        for i in range(self.max_population):
            g = Genome(f"initial genome: {i}")
            g.new_genome(1024, len(COMPLEX_MOVEMENT))
            self.population.append(g)
        print(f'Created population of {self.max_population} genomes')

    def evolve(self, generations=100):
        print('Evolving population...')
        generation = 0
        while generation < generations:
            self.evaluate(generation)
            # Sort population by fitness
            self.population.sort(key=lambda x: x.fitness, reverse=True)

            if self.max_population >= 5:
                # print top 5 genomes
                print('Top 5 genomes:')
                for i in range(5):
                    print(
                        f'Genome {i} fitness: {self.population[i].get_fitness()}')

            parent = self.population.pop(0)
            self.population = []
            self.population.append(parent)
            for i in range(self.max_population-1):
                child = parent.copy()
                child.mutate()
                self.population.append(child)
            generation += 1

    def evaluate(self, generation):
        print(f'Evaluation of generation {generation} started')
        use_mp = True
        if use_mp:
            print('Using multiprocessing')
            with mp.Pool(4) as pool:
            # print(f'Using {mp.cpu_count()} cores')
            # with mp.Pool(mp.cpu_count()) as pool:
                pool.starmap(self.evaluate_genome, [(generation, i)
                                                    for i in range(self.max_population)])
        else:
            for i in range(self.max_population):
                self.evaluate_genome(generation, i)
        print(f'Evaluation of generation {generation} completed')

    def evaluate_genome(self, generation, genome):
        print(f'Generation {generation} Genome {genome} Evaluating')

        env = gym_super_mario_bros.make(
            'SuperMarioBros-1-1-v0', render_mode='rgb_array', apply_api_compatibility=True)
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
        env = SkipFrame(env, skip=4)

        last_state = np.zeros((32, 32, 3), dtype=np.uint8)
        env.reset()
        score = 0
        while True:
            c_floats = process_image(last_state)
            self.population[genome].feed_forward(c_floats)
            next_state, reward, done, trunc, info = env.step(
                env.action_space.sample())
            if self.gui:
                frame = cv2.cvtColor(next_state, cv2.COLOR_RGB2BGR)
                cv2.imshow(f'SuperMarioBros-1-1=G{generation}-{genome}', frame)
                cv2.waitKey(1)
            score += reward
            if done or trunc:
                break
            last_state = next_state
        self.population[genome].set_fitness(score)
        print(
            f'Generation {generation} Genome {genome} fitness: {self.population[genome].get_fitness()}')
        if self.gui:
            cv2.destroyWindow(f'SuperMarioBros-1-1=G{generation}-{genome}')
        return score

    def save(self, path: str):
        # first create a directory to save the population
        if not os.path.exists(path):
            os.makedirs(path)

        # save Neat and population information to a file
        with open(os.path.join(path, 'neat.info'), 'w+') as f:
            # TODO: Save additional information about the population
            f.write(f'{self.max_population}\n')

        # save each genome in the population to a file
        os.mkdir(os.path.join(path, 'genomes'))
        for i in range(len(self.population)):
            self.population[i].save(os.path.join(
                path, 'genomes', f'genome_{i}.txt'))

        # zip the files and save them to a file
        shutil.make_archive(path, 'zip', path)

        # remove the directory
        shutil.rmtree(path)

    def load(self, filename):
        # unzip the file and extract the files to a directory
        shutil.unpack_archive(filename + ".zip", filename)

        # load Neat and population information from the file
        with open(os.path.join(filename, 'neat.info'), 'r') as f:
            self.max_population = int(f.readline())

        print(f'Info file read, {self.max_population} genomes')

        # load each genome in the population from a file
        self.population = []
        for i in range(self.max_population):
            g = Genome("test string")
            g.load(os.path.join(filename, 'genomes', f'genome_{i}.txt'))
            self.population.append(g)

        print(f'Genomes loaded, {len(self.population)} genomes')

        # remove the directory
        shutil.rmtree(filename)


def main(args):
    """
    Debugging function
    """

    if os.path.exists('./Makefile'):
        subprocess.run(['make', 'clean'])
        subprocess.run(['make'])
    else:
        raise Exception('Makefile not found')

    try:
        neat = Neat(max_population=args.max_population, gui=args.gui)
        if args.load:
            neat.load('./p1')
        else:
            neat.create_population()
        epocs = 0
        max_epocs = args.epocs
        while epocs < max_epocs:
            neat.evolve(args.generations)
            if args.save:
                neat.save('./p1_'+str(epocs))
            epocs += 1
    except KeyboardInterrupt:
        pass
    finally:
        if os.path.exists('./libgenome.so'):
            os.remove('./libgenome.so')
        if os.path.exists('./genome.o'):
            os.remove('./genome.o')


if __name__ == '__main__':
    """
    Debugging code
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true', help='Enable GUI')
    parser.add_argument('--load', action='store_true', help='Load population')
    parser.add_argument('--save', action='store_true', help='Save population')
    parser.add_argument('--epocs', type=int, default=1,
                        help='Number of epocs')
    parser.add_argument('--max_population', type=int, default=40,
                        help='Number of genomes in the population')
    parser.add_argument('--generations', type=int,
                        default=10, help='Number of generations')
    args = parser.parse_args()

    main(args)
