import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

data = pd.read_csv("USA_City_Adjacency_Matrix.csv")
data.set_index(data.columns[0], inplace=True)
cities = list(data.columns)
distance_matrix = data.to_numpy()

POPULATION_SIZE = 50
TOURNAMENT_SIZE = 5
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1
GENERATIONS = 100

def calculate_fitness(route):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i]][route[i + 1]]
    total_distance += distance_matrix[route[-1]][route[0]]
    return total_distance

def initialize_population():
    population = []
    for _ in range(POPULATION_SIZE):
        individual = list(range(1, len(cities)))
        random.shuffle(individual)
        individual = [0] + individual + [0]
        population.append(individual)
    return population

def tournament_selection(population):
    selected = []
    for _ in range(POPULATION_SIZE):
        tournament = random.sample(population, TOURNAMENT_SIZE)
        winner = min(tournament, key=calculate_fitness)
        selected.append(winner)
    return selected

def order_crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()

    size = len(parent1) - 2
    point1, point2 = sorted(random.sample(range(1, size + 1), 2))

    offspring1 = [-1] * len(parent1)
    offspring2 = [-1] * len(parent1)

    offspring1[0] = offspring2[0] = 0
    offspring1[-1] = offspring2[-1] = 0

    offspring1[point1:point2 + 1] = parent1[point1:point2 + 1]
    offspring2[point1:point2 + 1] = parent2[point1:point2 + 1]

    fill_remaining_genes(offspring1, parent2, point1, point2)
    fill_remaining_genes(offspring2, parent1, point1, point2)

    return offspring1, offspring2

def fill_remaining_genes(offspring, parent, point1, point2):
    size = len(parent)
    current_index = (point2 + 1) % (size - 1)

    for gene in parent:
        if gene not in offspring:
            while offspring[current_index] != -1:
                current_index = (current_index + 1) % (size - 1)
            offspring[current_index] = gene

def mutate(individual):
    if random.random() < MUTATION_RATE:
        idx1, idx2 = random.sample(range(1, len(individual) - 1), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

def genetic_algorithm():
    population = initialize_population()
    best_fitness_over_time = []

    for generation in range(GENERATIONS):
        fitness_scores = [calculate_fitness(individual) for individual in population]
        best_individual = population[np.argmin(fitness_scores)]
        best_fitness = min(fitness_scores)
        best_fitness_over_time.append(best_fitness)

        print(f"Generation {generation}: Best Fitness (Total Distance) = {best_fitness}")

        selected_population = tournament_selection(population)

        new_population = []
        for i in range(0, POPULATION_SIZE, 2):
            parent1 = selected_population[i]
            parent2 = selected_population[(i + 1) % POPULATION_SIZE]
            offspring1, offspring2 = order_crossover(parent1, parent2)
            mutate(offspring1)
            mutate(offspring2)
            new_population.extend([offspring1, offspring2])

        population = new_population

    final_fitness_scores = [calculate_fitness(individual) for individual in population]
    best_individual = population[np.argmin(final_fitness_scores)]
    best_fitness = min(final_fitness_scores)

    print("Best solution found:")
    route = [cities[i] for i in best_individual]
    print(route)
    print(f"Best Fitness (Total Distance) = {best_fitness}")

    save_solution_to_csv(route, best_fitness)
    plot_fitness_over_time(best_fitness_over_time)

    return best_individual

def save_solution_to_csv(route, best_fitness):
    df = pd.DataFrame({'Route': route})
    df['Total Distance (Fitness)'] = best_fitness
    df.to_csv("tsp_solution.csv", index=False)
    print("Best solution saved to tsp_solution.csv")

def plot_fitness_over_time(fitness_over_time):
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_over_time, marker='o', color='b', linestyle='-', linewidth=2, markersize=4)
    plt.title("Best Fitness Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness (Total Distance)")
    plt.grid(True)
    plt.savefig("fitness_over_time_tsp.png")
    print("Fitness over time plot saved as fitness_over_time.png")

best_solution = genetic_algorithm()
