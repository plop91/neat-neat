from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

import numpy as np

from genome import *
from utils import *

import os
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


regimes = [['1-1'], ['1-1', '1-2', '1-3', '1-4'], ['1-1', '1-2', '1-3', '1-4', '2-1', '2-3', '2-4'],
           ['1-1', '1-2', '1-3', '1-4', '2-1', '2-3', '2-4', '3-1', '3-2', '3-3', '3-4', '4-1', '4-2', '4-3',
            '4-4', '5-1', '5-2', '5-3', '5-4', '6-1', '6-2', '6-3', '6-4', '7-1', '7-3', '7-4', '8-1', '8-2', '8-3', '8-4']]  # all stages minus the water levels


class Neat:

    def __init__(self, name: str, max_population: int = 100, gui: bool = False, use_mp: bool = False):
        self.name = name                      # Name of the Model
        self.max_population = max_population  # Number of genomes in the population
        self.gui = gui                        # Enable GUI
        self.use_mp = use_mp                  # Enable multiprocessing
        self.current_generation = 0           # Current generation

    def create_population(self, input_size: int, output_size: int) -> None:
        # Create a directory to save the population
        os.makedirs(f'./genomes/{self.name}/Checkpoint0', exist_ok=True)

        for i in range(self.max_population):
            genome = Genome.init_genome(f"initial genome: {i}")
            Genome.create_genome(genome, input_size, output_size)
            Genome.save(
                genome, f'./genomes/{self.name}/Checkpoint0/genome_{i}.txt')
        # TODO: Save information about the population

    def evolve(self, generations: int = 100) -> None:
        print('Evolving population...')

        generation = self.current_generation  # Current generation
        total_generations = generation + generations  # Total number of generations

        # While the number of generations is less than the maximum number of generations
        while generation < total_generations:

            # Evaluate the population
            results = self.evaluate(generation)

            # This is a better way to get the fittest genome than sorting the results list because 
            highest_score = 0
            highest_score_index = 0

            second_highest_score = 0
            second_highest_score_index = 0

            for i in range(len(results)):
                if results[i][1] >= highest_score:
                    highest_score = results[i][1]
                    highest_score_index = i

                if results[i][1] >= second_highest_score and i != highest_score_index:
                    second_highest_score = results[i][1]
                    second_highest_score_index = i

            fittest_result = results[highest_score_index]

            print(
                f"Generation {generation} Fittest genome {fittest_result[0]} scored: {fittest_result[1]}")

            # Load the fittest genome
            fittest_genome = Genome.init_genome(f"Fittest Genome {generation}")
            Genome.load(
                fittest_genome, f'./genomes/{self.name}/Checkpoint{generation}/genome_{fittest_result[0]}.txt')
            
            second_fittest_genome = Genome.init_genome(f"Second Fittest Genome {generation}")
            Genome.load(
                second_fittest_genome, f'./genomes/{self.name}/Checkpoint{generation}/genome_{second_highest_score_index}.txt')

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
            
            # Save the second fittest genome as the second genome of the next generation
            Genome.save(second_fittest_genome,
                        f'./genomes/{self.name}/Checkpoint{generation+1}/genome_1.txt')

            # Create the next generation
            for i in range(2, self.max_population):
                # Copy the fittest genome
                new_genome = Genome.copy(
                    fittest_genome, f"Mutant of Fittest Genome {generation}")
                
                # Crossover the fittest genome with the second fittest genome
                # new_genome = Genome.crossover(fittest_genome, second_fittest_genome)

                # Mutate the new genome
                Genome.mutate(new_genome)

                # Save the new genome
                Genome.save(
                    new_genome, f'./genomes/{self.name}/Checkpoint{generation+1}/genome_{i}.txt')

            generation += 1
            self.current_generation = generation
            # TODO: implement a check to see if the population has changed in the last 20 generations
            # check if the new genome0 matches the old genome0 if it does then the population has converged or is stuck

    def evaluate(self, generation):
        print(f'Evaluation of generation {generation} started')
        results = []
        if self.use_mp:
            with mp.Pool(int(mp.cpu_count()/2)) as pool:
                genome_combos = [(generation, i)
                                 for i in range(self.max_population)]
                results = pool.starmap(self.evaluate_genome, genome_combos)
                
        else:
            for i in range(self.max_population):
                results.append(self.evaluate_genome(generation, i))
        print(f'Evaluation of generation {generation} completed')
        print(f"Results: {results}")
        return results

    def evaluate_genome(self, generation_index: int, genome_index: int, regime=0):
        # print(f'Generation {generation_index} Genome {genome_index} Evaluating')

        window_name = f'SuperMarioBros AI:{self.name} Generation:{generation_index} Genome:{genome_index}'
        genome_name = f'Generation {generation_index} Genome {genome_index}'
        genome_path = f'./genomes/{self.name}/Checkpoint{generation_index}/genome_{genome_index}.txt'

        if regime == 0:
            env = gym_super_mario_bros.make(
                'SuperMarioBros-1-1-v0', render_mode='rgb_array', apply_api_compatibility=True)
        else:
            env = gym_super_mario_bros.make(f"SuperMarioBrosRandomStages-v0",
                                            stages=regimes[regime], render_mode='rgb_array', apply_api_compatibility=True)

        env = JoypadSpace(env, COMPLEX_MOVEMENT)
        env = SkipFrame(env, skip=4)

        genome = Genome.init_genome(genome_name)
        Genome.load(genome, genome_path)

        # print(f'Generation {generation_index} Genome {genome_index} loaded')

        last_state = np.zeros((32, 32, 3), dtype=np.uint8)
        env.reset()
        score = 0
        no_move_count = 0
        hold_jump_count = 0
        last_x_pos = 0
        last_y_pos = 0
        while True:
            c_floats = process_image(last_state)
            action = Genome.feed_forward(genome, c_floats)
            next_state, reward, done, trunc, info = env.step(action)

            # print(info)

            score += reward

            # penalize the genome if it doesn't move
            if info['x_pos'] == last_x_pos and info['y_pos'] == last_y_pos:
                no_move_count += 1
            else:
                no_move_count = 0
                last_x_pos = info['x_pos']
                last_y_pos = info['y_pos']

            if no_move_count > 200:
                score -= 10
                trunc = True
                print('No movement detected')

            # track how long the jump button is held
            # actions that jump are 2, 4, 5, 7, 9
            if action in [2, 4, 5, 7, 9]:
                hold_jump_count += 1
            else:
                hold_jump_count = 0

            #
            if hold_jump_count > 200:
                print('Jump held for too long')
                score -= 100
                hold_jump_count = 0

            if self.gui:
                frame = cv2.cvtColor(next_state, cv2.COLOR_RGB2BGR)
                cv2.imshow(window_name, frame)
                cv2.waitKey(1)

            if done or trunc:
                break
            last_state = next_state

        # reward the genome for its score, number of coins, lives, status(big/small), time
        score += info['score']
        score += info['coins'] * 10
        # score += last_info['live'] * 100 # disable live reward for now because im not sure it will help anything
        if info['status'] != 'small':
            score += 100
        # score += last_info['time'] * 10 # disable time reward for now because it might incentivize the genome end the game early to get the best score
        if info['flag_get']:
            score += 2000

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

    def load(self):
        # check if the genomes and model directories exists
        if not os.path.exists(f'./genomes/{self.name}'):
            raise Exception('Genomes directory not found')
        # check if there are any checkpoints
        latest_checkpoint = 0
        for f in os.listdir(f'./genomes/{self.name}'):
            if os.path.isdir(f'./genomes/{self.name}/{f}'):
                if f.startswith('Checkpoint'):
                    i = int(f.replace('Checkpoint', ""))
                    if i > latest_checkpoint:
                        latest_checkpoint = i

        self.current_generation = latest_checkpoint


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
            neat = Neat(args.name, max_population=args.max_population,
                        gui=args.gui, use_mp=args.use_mp)
            neat.load()
        else:
            neat = Neat(args.name, max_population=args.max_population,
                        gui=args.gui, use_mp=args.use_mp)
            neat.create_population(1024, len(COMPLEX_MOVEMENT))
        neat.evolve(args.generations)

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
    parser.add_argument('--max_population', type=int, default=4,
                        help='Number of genomes in the population')
    parser.add_argument('--generations', type=int,
                        default=4, help='Number of generations')

    args = parser.parse_args()

    main(args)
