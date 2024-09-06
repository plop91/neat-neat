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

    # New Genome
    lib.NewGenome.argtypes = [ctypes.c_char_p]
    lib.NewGenome.restype = ctypes.c_void_p

    # Delete Genome
    lib.DeleteGenome.argtypes = [ctypes.c_void_p]
    lib.DeleteGenome.restype = None

    # Copy Genome
    lib.CopyGenome.argtypes = [ctypes.c_void_p]
    lib.CopyGenome.restype = ctypes.c_void_p

    # Init new genome
    lib.InitGenome.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.InitGenome.restype = None

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

    @property
    def name(self):
        return self.get_name()

    @property
    def fitness(self):
        return self.get_fitness()

    def __init__(self, name):
        self.genome = self.lib.NewGenome(name.encode('utf-8'))
        self.genome_initialized = False

    def __del__(self):
        self.lib.DeleteGenome(self.genome)

    def copy(self):
        if not self.genome_initialized:
            raise Exception('Genome not initialized')
        g = Genome("test string")
        g.genome = self.lib.CopyGenome(self.genome)
        g.genome_initialized = True
        return g

    def new_genome(self, num_inputs, num_outputs):
        self.lib.InitGenome(self.genome, num_inputs, num_outputs)
        self.genome_initialized = True

    def load(self, filename):
        self.lib.LoadGenome(self.genome, filename.encode('utf-8'))
        self.genome_initialized = True

    def save(self, filename):
        if not self.genome_initialized:
            raise Exception('Genome not initialized')
        self.lib.SaveGenome(self.genome, filename.encode('utf-8'))

    def mutate(self):
        if not self.genome_initialized:
            raise Exception('Genome not initialized')
        self.lib.MutateGenome(self.genome)

    def crossover(self, other):
        if not self.genome_initialized or not other.genome_initialized:
            raise Exception('Genome not initialized')
        self.lib.CrossoverGenome(self.genome, other.genome)

    def feed_forward(self, inputs):
        if not self.genome_initialized:
            raise Exception('Genome not initialized')
        return self.lib.FeedForwardGenome(self.genome, inputs)

    def set_name(self, name):
        self.lib.SetName(self.genome, name.encode('utf-8'))

    def get_name(self):
        return self.lib.GetName(self.genome).decode('utf-8')

    def set_fitness(self, fitness):
        self.lib.SetFitness(self.genome, fitness)

    def get_fitness(self):
        return self.lib.GetFitness(self.genome)

    def print_genome_info(self):
        self.lib.PrintGenomeInfo(self.genome)
