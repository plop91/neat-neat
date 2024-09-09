from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

import numpy as np

from genome import *
from utils import *

from tqdm import tqdm
import os
import shutil
import time
import multiprocessing as mp
import ctypes

import warnings
warnings.filterwarnings("ignore")


def process_image(img: np.ndarray):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (32, 32))
    img = img.flatten()
    c_floats = (ctypes.c_float * len(img))(*img)
    return c_floats


class Neat:

    def __init__(self, name: str, max_population: int = 100, gui: bool = False, use_mp: bool = False):
        self.name = name
        self.max_population = max_population  # Number of genomes in the population
        # self.population = []  # List of genomes
        self.gui = gui
        self.use_mp = use_mp

    def create_population(self, input_size: int, output_size: int) -> None:
        for i in range(self.max_population):
            genome = Genome.init_genome(f"initial genome: {i}")
            Genome.create_genome(genome, input_size, output_size)
            Genome.save(
                genome, f'./genomes/{self.name}/Checkpoint0/genome_{i}.txt')
        # TODO: Save information about the population

    def evolve(self, generations: int = 100) -> None:
        print('Evolving population...')

        generation = 0  # Current generation

        # While the number of generations is less than the maximum number of generations
        while generation < generations:

            # Evaluate the population
            results = self.evaluate(generation)

            # Sort the results by fitness
            results.sort(key=lambda x: x[1], reverse=True)

            # Get the fittest genome
            fittest_result = results[0]

            print(
                f"Generation {generation} Fittest genome {fittest_result[0]} scored: {fittest_result[1]}")

            # Load the fittest genome
            fittest_genome = Genome.init_genome(f"Fittest Genome {generation}")
            Genome.load(
                fittest_genome, f'./genomes/{self.name}/Checkpoint{generation}/genome_{fittest_result[0]}.txt')

            # Check if the fittest genome has connections, if not its a problem
            if not Genome.does_genome_have_connections(fittest_genome):
                raise Exception(
                    'Fittest genome has no connections!! Aborting...')

            # Create a directory for the next generation
            os.makedirs(
                f'./genomes/{self.name}/Checkpoint{generation+1}', exist_ok=True)

            # Save the fittest genome as the first genome of the next generation
            Genome.save(fittest_genome,
                        f'./genomes/{self.name}/Checkpoint{generation+1}/genome_0.txt')

            # Create the next generation
            for i in range(1, self.max_population):
                # Copy the fittest genome
                new_genome = Genome.copy(
                    fittest_genome, f"Mutant of Fittest Genome {generation}")

                # TODO: implement crossover
                # new_genome = Genome.crossover(fittest_genome, second_fittest_genome)
                Genome.mutate(new_genome)

                # Save the new genome
                Genome.save(
                    new_genome, f'./genomes/{self.name}/Checkpoint{generation+1}/genome_{i}.txt')

            generation += 1

    def evaluate(self, generation):
        print(f'Evaluation of generation {generation} started')
        results = []
        if self.use_mp:
            print('Using multiprocessing')
            with mp.Pool(int(mp.cpu_count()/2)) as pool:
                genome_combos = [(generation, i) for i in range(self.max_population)]
                results = pool.starmap(self.evaluate_genome, genome_combos)
        else:
            for i in range(self.max_population):
                results.append(self.evaluate_genome(generation, i))
        print(f'Evaluation of generation {generation} completed')
        print(f"Results: {results}")
        return results

    def evaluate_genome(self, generation_index: int, genome_index: int):
        # print(f'Generation {generation_index} Genome {genome_index} Evaluating')

        window_name = f'SuperMarioBros AI:{self.name} Generation:{generation_index} Genome:{genome_index}'
        genome_name = f'Generation {generation_index} Genome {genome_index}'
        genome_path = f'./genomes/{self.name}/Checkpoint{generation_index}/genome_{genome_index}.txt'

        env = gym_super_mario_bros.make(
            'SuperMarioBros-1-1-v0', render_mode='rgb_array', apply_api_compatibility=True)
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
        env = SkipFrame(env, skip=4)

        genome = Genome.init_genome(genome_name)
        Genome.load(genome, genome_path)

        # print(f'Generation {generation_index} Genome {genome_index} loaded')

        last_state = np.zeros((32, 32, 3), dtype=np.uint8)
        env.reset()
        score = 0
        no_move_count = 0
        last_x_pos = 0
        last_y_pos = 0
        i = 0
        while True:
            c_floats = process_image(last_state)
            action = Genome.feed_forward(genome, c_floats)
            next_state, reward, done, trunc, info = env.step(action)

            if info['x_pos'] == last_x_pos and info['y_pos'] == last_y_pos:
                no_move_count += 1
            else:
                no_move_count = 0
                last_x_pos = info['x_pos']
                last_y_pos = info['y_pos']

            if no_move_count > 200:
                score -= 1000
                trunc = True
                print('No movement detected')

            if self.gui:
                frame = cv2.cvtColor(next_state, cv2.COLOR_RGB2BGR)
                cv2.imshow(window_name, frame)
                cv2.waitKey(1)
            score += reward
            if done or trunc:
                if done:
                    score += 1000
                break
            last_state = next_state

        Genome.set_fitness(genome, score)
        print(
            f'Generation {generation_index} Genome {genome_index} fitness: {Genome.get_fitness(genome)}')

        # shut down the environment

        if self.gui:
            cv2.destroyWindow(window_name)

        env.close()

        # save and delete the genome
        Genome.save(genome, genome_path)
        Genome.delete_genome(genome)

        return (genome_index, score)

    def save(self, path: str) -> None:
        # first create a directory to save the population
        if not os.path.exists(path):
            os.makedirs(path)

        # save Neat and population information to a file
        with open(os.path.join(path, 'neat.info'), 'w+') as f:
            # TODO: Save additional information about the population
            f.write(f'{self.max_population}\n')

        # # save each genome in the population to a file
        # os.mkdir(os.path.join(path, 'genomes'))
        # for i in range(len(self.population)):
        #     self.population[i].save(os.path.join(
        #         path, 'genomes', f'genome_{i}.txt'))

        # zip the files and save them to a file
        # shutil.make_archive(path, 'zip', path)

        # remove the directory
        # shutil.rmtree(path)

    def load(self, filename: str) -> None:
        # unzip the file and extract the files to a directory
        # shutil.unpack_archive(filename + ".zip", filename)

        # load Neat and population information from the file
        with open(os.path.join(filename, 'neat.info'), 'r') as f:
            self.max_population = int(f.readline())

        print(f'Info file read, {self.max_population} genomes')

        # # load each genome in the population from a file
        # self.population = []
        # for i in range(self.max_population):
        #     g = Genome("test string")
        #     g.load(os.path.join(filename, 'genomes', f'genome_{i}.txt'))
        #     self.population.append(g)

        print(f'Genomes loaded, {len(self.population)} genomes')

        # remove the directory
        # shutil.rmtree(filename)


def main(args):
    """
    Debugging function
    """
    if os.path.exists('./Makefile'):
        subprocess.run(['make', 'clean'])
        subprocess.run(['make'])
    else:
        raise Exception('Makefile not found')

    if not os.path.exists('./genomes'):
        os.mkdir('./genomes')

    try:
        neat = Neat(args.name, max_population=args.max_population,
                    gui=args.gui, use_mp=args.use_mp)
        if args.load:
            # TODO: get latest checkpoint
            neat.load(f'./genomes/{args.name}/checkpoint0')
        else:
            neat.create_population(1024, len(COMPLEX_MOVEMENT))
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
    parser.add_argument('name', type=str, help='Name of the network')
    parser.add_argument('--gui', action='store_true', help='Enable GUI')
    parser.add_argument('--use_mp', action='store_true',
                        help='Enable multiprocessing')
    parser.add_argument('--debug', action='store_true',
                        help='Enable Debugging')
    parser.add_argument('--load', action='store_true', help='Load population')
    parser.add_argument('--save', action='store_true', help='Save population')
    parser.add_argument('--epocs', type=int, default=1,
                        help='Number of epocs')
    parser.add_argument('--max_population', type=int, default=4,
                        help='Number of genomes in the population')
    parser.add_argument('--generations', type=int,
                        default=4, help='Number of generations')

    args = parser.parse_args()

    main(args)
