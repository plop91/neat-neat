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

    # DEBUG FUNCTIONS
    lib.DoesGenomeHaveConnections.argtypes = [ctypes.c_void_p]
    lib.DoesGenomeHaveConnections.restype = ctypes.c_bool

    lib.PrintOutputLayerValues.argtypes = [ctypes.c_void_p]
    lib.PrintOutputLayerValues.restype = None

    @staticmethod
    def init_genome(name: str) -> ctypes.c_void_p:
        return Genome.lib.InitGenome(name.encode('utf-8'))

    @staticmethod
    def delete_genome(genome: ctypes.c_void_p):
        Genome.lib.DeleteGenome(genome)

    @staticmethod
    def copy(genome: ctypes.c_void_p, name: str) -> ctypes.c_void_p:
        genome = Genome.lib.CopyGenome(genome)
        Genome.lib.SetName(genome, name.encode('utf-8'))
        return genome

    @staticmethod
    def create_genome(genome: ctypes.c_void_p, num_inputs: int, num_outputs: int):
        Genome.lib.CreateGenome(genome, num_inputs, num_outputs)

    @staticmethod
    def load(genome: ctypes.c_void_p, filename: str):
        Genome.lib.LoadGenome(genome, filename.encode('utf-8'))

    @staticmethod
    def save(genome: ctypes.c_void_p, filename: str):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        Genome.lib.SaveGenome(genome, filename.encode('utf-8'))

    @staticmethod
    def mutate(genome: ctypes.c_void_p):
        Genome.lib.MutateGenome(genome)

    @staticmethod
    def crossover(genome: ctypes.c_void_p, other: ctypes.c_void_p):
        Genome.lib.CrossoverGenome(genome, other)

    @staticmethod
    def feed_forward(genome: ctypes.c_void_p, inputs):
        return Genome.lib.FeedForwardGenome(genome, inputs)

    @staticmethod
    def set_name(genome: ctypes.c_void_p, name: str):
        Genome.lib.SetName(genome, name.encode('utf-8'))

    @staticmethod
    def get_name(genome: ctypes.c_void_p) -> str:
        return Genome.lib.GetName(genome).decode('utf-8')

    @staticmethod
    def set_fitness(genome: ctypes.c_void_p, fitness: float):
        Genome.lib.SetFitness(genome, fitness)

    @staticmethod
    def get_fitness(genome: ctypes.c_void_p) -> float:
        return Genome.lib.GetFitness(genome)

    @staticmethod
    def print_genome_info(genome: ctypes.c_void_p) -> None:
        Genome.lib.PrintGenomeInfo(genome)

    @staticmethod
    def does_genome_have_connections(genome: ctypes.c_void_p) -> bool:
        return Genome.lib.DoesGenomeHaveConnections(genome)
    
    @staticmethod
    def print_output_layer_values(genome: ctypes.c_void_p):
        Genome.lib.PrintOutputLayerValues(genome)
