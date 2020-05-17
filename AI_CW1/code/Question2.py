from amnesiac import blurry_memory
import random
import string


class Chromosome:
    """
    CLass to define password.
    Sequence refers to the password string.
    Fitness refers to the fitness value received for the password from blurry_memory method
    """
    def __init__(self, sequence, fitness=None):
        self.sequence = sequence
        self.fitness = fitness

    def __repr__(self):
        return repr((self.sequence, self.fitness))


# list of hyper-parameters are declared below
POPULATION_SIZE = 100  # population size of each generation
MUTATION_PROB = 0.05  # probability of mutation in a generation
SEQUENCE_LEN = 10  # the length of the password
TOURNAMENT_SIZE = 3  # number of participation for tournament selection
CROSSOVER_POINTS = 2  # N for n-point crossover
PASSWORD_CHAR = string.ascii_uppercase + string.digits + '_'  # allowed characters in the password

PASSWORD_TYPE = 0  # the password type (0 or 1)
STUDENT_NO = 190385633  # student number

# fixed seed for exact reproduction of the results as shown in the report.
# random.seed(5)

def compute_fitness(pop, password_type):
    """
    this method computes the fitness for the population
    :param pop: population - a list of Chromosomes
    :param password_type: (0 or 1) - the password for which to compute the fitness of the population for.
    :return: poplutaion - a list of Chromosomes with their evaluated fitness value
    """
    population_list = [individual.sequence for individual in pop]
    evaluated_pop = blurry_memory(population_list, STUDENT_NO, password_type)
    pop = [Chromosome(key, value) for key, value in evaluated_pop.items()]
    return pop


def sequence_generation():
    """
    generates random string made from the allowed password characters and of the defined sequence length
    :return: string - random candidate password
    """
    return ''.join(random.choice(PASSWORD_CHAR) for i in range(SEQUENCE_LEN))


def initialise_population():
    """
    initialises the population
    :return: a list of Chromosomes with fitness = None
    """
    pop = []
    for i in range(POPULATION_SIZE):
        pop.append(Chromosome(sequence_generation()))
    return pop


def selection(pop):
    """
    Performs tournament selection. Selects 2 fittest individuals from a random pool of "TOURNAMENT_SIZE"
    :param pop: population - a list of Chromosomes
    :return: two fittest individual as the winner of the tournament
    """
    tournament_list = []
    for i in range(TOURNAMENT_SIZE):
        tournament_list.append(random.choice(pop))
    tournament_list = sorted(tournament_list, key=lambda chromosome: chromosome.fitness, reverse=True)
    parent1 = tournament_list[0]
    parent2 = tournament_list[1]
    return parent1, parent2


def crossover(p1, p2):
    """
    Performs crossover between two parents to produce two children.
    Crossover is performed at n-random crossover points.
    N is defined by "CROSSOVER_POINTs"
    :param p1: parent1 of type Chromosome
    :param p2: parent2 of type Chromosome
    :return: two children with fitness=None
    """
    sequence1 = p1.sequence
    sequence2 = p2.sequence
    for i in range(CROSSOVER_POINTS):
        point = random.randint(0, SEQUENCE_LEN - 1)
        new_sequence1 = sequence1[:point] + sequence2[point:]
        new_sequence2 = sequence2[:point] + sequence1[point:]
        sequence1 = new_sequence1
        sequence2 = new_sequence2
    return Chromosome(sequence1), Chromosome(sequence2)


def mutation(pop):
    """
    Performs mutation on the population.
    Mutates a random gene (char in the string) of a chromosome with a probability of "MUTATION_PROB".
    The char at the position is replaced by a random char generated from the allowed characters
    :param pop: list of Chromosome
    :return: mutated population - list of chromosome
    """
    for i in range(POPULATION_SIZE):
        individual = pop[i]
        if random.uniform(0, 1) < MUTATION_PROB:
            position = random.randint(0, SEQUENCE_LEN - 1)
            mutated_gene = random.choice(PASSWORD_CHAR)
            str = list(individual.sequence)
            str[position] = mutated_gene
            individual.sequence = "".join(str)
    return pop


# Number of generation
generation_count = 1
# initialise the first generation
population = initialise_population()
# compute the fitness of the first generation
population = compute_fitness(population, PASSWORD_TYPE)
# sort the generation wrt fitness value
sorted_population = sorted(population, key=lambda chromosome: chromosome.fitness, reverse=True)

# create generations until the highest fitness value of the generation is 1
while sorted_population[0].fitness != 1.0:
    next_generation = list()
    for i in range(int(POPULATION_SIZE / 2)):
        # selection
        parent1, parent2 = selection(population)
        # crossover
        child1, child2 = crossover(parent1, parent2)
        # add the children to the next generation
        next_generation.append(child1)
        next_generation.append(child2)
    # mutation
    population = mutation(next_generation)
    # compute the fitness of the new generation
    population = compute_fitness(population, PASSWORD_TYPE)
    # sort the individuals in the population wrt fitness in decreasing order
    sorted_population = sorted(population, key=lambda chromosome: chromosome.fitness, reverse=True)
    generation_count += 1

print("PASSWORD TYPE: ", PASSWORD_TYPE)
print("No. of Generations: ", generation_count)
print("Password: ", sorted_population[0].sequence)
