
# Traveling Salesman Problem (TSP) Solver Using Genetic Algorithm

This project solves the Traveling Salesman Problem (TSP) using a Genetic Algorithm (GA). The goal is to find the shortest possible route that visits each city exactly once and returns to the origin city.

## Features

- Implements Genetic Algorithm to solve TSP.
- Supports customizable population size, mutation rate, crossover rate, and generations.
- Uses an adjacency matrix for city distances from a CSV file.
- Saves the best solution to a CSV file.
- Visualizes the improvement in fitness (total distance) over generations with a graph.

## Requirements

- Python 3.x
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/tsp-genetic-algorithm.git
   cd tsp-genetic-algorithm
   ```

2. **Install Dependencies**

   Install the required Python libraries:

   ```bash
   pip install pandas numpy matplotlib
   ```

3. **Set Up Input Data**

   Ensure you have an adjacency matrix CSV file (`USA_City_Adjacency_Matrix.csv`) with city distances. Place this file in the appropriate directory as referenced in the code.

4. **Modify Parameters**

   Customize parameters such as:
   - `POPULATION_SIZE`
   - `TOURNAMENT_SIZE`
   - `CROSSOVER_RATE`
   - `MUTATION_RATE`
   - `GENERATIONS`
   
   These can be adjusted directly in the script for experimentation.

## Usage

Run the script:

```bash
python GA_LS.py
```

The program will:
1. Load the city adjacency matrix from the CSV file.
2. Initialize a random population of potential solutions.
3. Evolve the population through selection, crossover, and mutation for a set number of generations.
4. Save the best solution to a CSV file (`tsp_solution.csv`).
5. Save a plot of the best fitness over generations (`fitness_over_time_tsp.png`).

## Output

- **CSV Output:** A file containing the best route and its total distance (fitness).
- **Fitness Plot:** A graphical representation of fitness improvement over generations.

## Example

Here’s a snippet of what the output might look like:

```
Generation 0: Best Fitness (Total Distance) = 1523.4
Generation 1: Best Fitness (Total Distance) = 1457.8
...
Generation 99: Best Fitness (Total Distance) = 1247.3

Best solution found:
['City1', 'City5', 'City3', 'City2', 'City4', 'City1']
Best Fitness (Total Distance) = 1247.3
```

## File Structure

- `GA_LS.py`: Main script for solving TSP using GA.
- `USA_City_Adjacency_Matrix.csv`: Input file with the city adjacency matrix.
- `tsp_solution.csv`: Output file with the best route and its total distance.
- `fitness_over_time_tsp.png`: Fitness improvement graph over generations.

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute.

## Contribution

Contributions are welcome! Feel free to fork the repository and submit a pull request with improvements or bug fixes.
