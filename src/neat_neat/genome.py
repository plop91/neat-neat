import ctypes
import subprocess
import os

# TODO: replace this with a proper build system
if os.path.exists('./Makefile'):
    subprocess.run(['make'])
else:
    raise Exception('Makefile not found')


class Genome:
    lib = ctypes.cdll.LoadLibrary('./libgenome.so')

    # Init Genome
    lib.InitGenome.argtypes = [ctypes.c_char_p]
    lib.InitGenome.restype = ctypes.c_void_p

    # Delete Genome
    lib.DeleteGenome.argtypes = [ctypes.c_void_p]
    lib.DeleteGenome.restype = None

    # Copy Genome
    lib.CopyGenome.argtypes = [ctypes.c_void_p]
    lib.CopyGenome.restype = ctypes.c_void_p

    # Create genome
    lib.CreateGenome.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.CreateGenome.restype = None

    # Load Genome
    lib.LoadGenome.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.LoadGenome.restype = None

    # Save Genome
    lib.SaveGenome.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.SaveGenome.restype = None

    # Mutate Genome
    lib.MutateGenome.argtypes = [ctypes.c_void_p]
    lib.MutateGenome.restype = None

    # Crossover Genome
    lib.CrossoverGenome.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.CrossoverGenome.restype = None

    # Feed Forward Genome
    lib.FeedForwardGenome.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.FeedForwardGenome.restype = ctypes.c_int

    # Set Name
    lib.SetName.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.SetName.restype = None

    # Get Name
    lib.GetName.argtypes = [ctypes.c_void_p]
    lib.GetName.restype = ctypes.c_char_p

    # Set Fitness
    lib.SetFitness.argtypes = [ctypes.c_void_p, ctypes.c_float]
    lib.SetFitness.restype = None

    # Get Fitness
    lib.GetFitness.argtypes = [ctypes.c_void_p]
    lib.GetFitness.restype = ctypes.c_float

    # Print Genome
    lib.PrintGenomeInfo.argtypes = [ctypes.c_void_p]
    lib.PrintGenomeInfo.restype = None

    @staticmethod
    def init_genome(name):
        return Genome.lib.InitGenome(name.encode('utf-8'))

    @staticmethod
    def delete_genome(genome):
        Genome.lib.DeleteGenome(genome)

    @staticmethod
    def copy(genome, name):
        genome = Genome.lib.CopyGenome(genome)
        # Genome.lib.SetName(genome, name.encode('utf-8'))
        return genome

    @staticmethod
    def create_genome(genome, num_inputs, num_outputs):
        Genome.lib.CreateGenome(genome, num_inputs, num_outputs)

    @staticmethod
    def load(genome, filename):
        Genome.lib.LoadGenome(genome, filename.encode('utf-8'))

    @staticmethod
    def save(genome, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        Genome.lib.SaveGenome(genome, filename.encode('utf-8'))

    @staticmethod
    def mutate(genome):
        Genome.lib.MutateGenome(genome)

    @staticmethod
    def crossover(genome, other):
        Genome.lib.CrossoverGenome(genome, other)

    @staticmethod
    def feed_forward(genome, inputs):
        return Genome.lib.FeedForwardGenome(genome, inputs)

    @staticmethod
    def set_name(genome, name):
        Genome.lib.SetName(genome, name.encode('utf-8'))

    @staticmethod
    def get_name(genome):
        return Genome.lib.GetName(genome).decode('utf-8')

    @staticmethod
    def set_fitness(genome, fitness):
        Genome.lib.SetFitness(genome, fitness)

    @staticmethod
    def get_fitness(genome):
        return Genome.lib.GetFitness(genome)

    @staticmethod
    def print_genome_info(genome):
        Genome.lib.PrintGenomeInfo(genome)
