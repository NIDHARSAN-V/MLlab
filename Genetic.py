import random

def generate_population(pop_size, gene_length):
    return [[random.randint(0, 1) for _ in range(gene_length)] for _ in range(pop_size)]

def fitness(individual):
    return sum(individual)

def tournament_selection(population, k=3):
    selected = random.sample(population, k)
    selected.sort(key=fitness, reverse=True)
    return selected[0]

if __name__ == "__main__":
    pop_size = 5
    gene_length = 6

    population = generate_population(pop_size, gene_length)
    print("Generated Population with Fitness:")
    for i, individual in enumerate(population, 1):
        print(f"Individual {i}: {individual} -> Fitness: {fitness(individual)}")

    selected = tournament_selection(population, k=3)
    print("\nSelected Individual (Tournament):", selected)
    print("Selected Fitness:", fitness(selected))
    
    
    
    
    
    
    
    
    
    
import random

# Single-point crossover
def single_point_crossover(p1, p2):
    point = random.randint(1, len(p1) - 1)
    return p1[:point] + p2[point:], p2[:point] + p1[point:]

# Multi-point crossover
def multi_point_crossover(p1, p2, points=2):
    if points >= len(p1):
        raise ValueError("Too many crossover points")
    crossover_points = sorted(random.sample(range(1, len(p1)), points))
    child1, child2 = [], []
    toggle = False
    prev = 0
    for point in crossover_points + [len(p1)]:
        if toggle:
            child1 += p2[prev:point]
            child2 += p1[prev:point]
        else:
            child1 += p1[prev:point]
            child2 += p2[prev:point]
        toggle = not toggle
        prev = point
    return child1, child2

# Uniform crossover
def uniform_crossover(p1, p2, swap_prob=0.5):
    child1, child2 = [], []
    for gene1, gene2 in zip(p1, p2):
        if random.random() < swap_prob:
            child1.append(gene2)
            child2.append(gene1)
        else:
            child1.append(gene1)
            child2.append(gene2)
    return child1, child2

# Main section to test all crossovers
if __name__ == "__main__":
    p1 = [1, 1, 1, 1, 1, 1]
    p2 = [0, 0, 0, 0, 0, 0]

    print("Parent 1:", p1)
    print("Parent 2:", p2)

    # Single-point
    c1, c2 = single_point_crossover(p1, p2)
    print("\nSingle-Point Crossover:")
    print("Child1:", c1)
    print("Child2:", c2)

    # Multi-point
    c1, c2 = multi_point_crossover(p1, p2, points=3)
    print("\nMulti-Point Crossover:")
    print("Child1:", c1)
    print("Child2:", c2)

    # Uniform
    c1, c2 = uniform_crossover(p1, p2)
    print("\nUniform Crossover:")
    print("Child1:", c1)
    print("Child2:", c2)
        
        
        
        
        
        
        
        
import random

def bit_flip_mutation(individual, mutation_rate=0.05):
    mutated_individual = []  # To store the new mutated genes
    
    for gene in individual:
        rand_val = random.random()  # Generate a random number between 0 and 1
        
        if rand_val > mutation_rate:
            # With probability (1 - mutation_rate), keep the gene as is
            mutated_individual.append(gene)
        else:
            # With probability mutation_rate, flip the gene: 0 -> 1, 1 -> 0
            flipped_gene = 1 - gene
            mutated_individual.append(flipped_gene)
    
    return mutated_individual


if _name_ == "_main_":
    ind = [1, 0, 1, 0, 1, 0]
    print("Before:", ind)
    print("After :", bit_flip_mutation(ind, 0.3))
        