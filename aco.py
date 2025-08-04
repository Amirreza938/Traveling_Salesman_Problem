import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple, Optional
import time
import matplotlib
matplotlib.use('Agg')

class AntColonyOptimization:
    """
    Ant Colony Optimization implementation for solving the Traveling Salesman Problem (TSP).
    
    This implementation uses the Ant System (AS) algorithm with the following features:
    - Pheromone trail updates
    - Heuristic information (distance-based)
    - Elitist strategy for best solution reinforcement
    - Local pheromone update
    """
    
    def __init__(self, 
                 n_cities: int,
                 n_ants: int = 50,
                 n_iterations: int = 100,
                 decay: float = 0.1,
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 Q: float = 100.0,
                 elitist_factor: float = 2.0,
                 random_seed: Optional[int] = None):
        """
        Initialize the ACO algorithm.
        
        Args:
            n_cities: Number of cities in the TSP
            n_ants: Number of ants in the colony
            n_iterations: Number of iterations to run
            decay: Pheromone decay rate (evaporation)
            alpha: Pheromone importance factor
            beta: Heuristic importance factor
            Q: Pheromone deposit constant
            elitist_factor: Factor for elitist pheromone update
            random_seed: Random seed for reproducibility
        """
        self.n_cities = n_cities
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.Q = Q
        self.elitist_factor = elitist_factor
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Initialize cities with random coordinates
        self.cities = self._generate_cities()
        
        # Calculate distance matrix
        self.distance_matrix = self._calculate_distance_matrix()
        
        # Initialize pheromone matrix
        self.pheromone_matrix = np.ones((n_cities, n_cities)) / n_cities
        
        # Best solution tracking
        self.best_solution = None
        self.best_distance = float('inf')
        self.best_solutions_history = []
        self.avg_solutions_history = []
        
    def _generate_cities(self) -> np.ndarray:
        """Generate random city coordinates."""
        return np.random.rand(self.n_cities, 2) * 100
    
    def _calculate_distance_matrix(self) -> np.ndarray:
        """Calculate the distance matrix between all cities."""
        distances = np.zeros((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                if i != j:
                    distances[i][j] = np.sqrt(
                        (self.cities[i][0] - self.cities[j][0])**2 + 
                        (self.cities[i][1] - self.cities[j][1])**2
                    )
        return distances
    
    def _calculate_tour_length(self, tour: List[int]) -> float:
        """Calculate the total length of a tour."""
        total_distance = 0
        for i in range(len(tour)):
            current_city = tour[i]
            next_city = tour[(i + 1) % len(tour)]
            total_distance += self.distance_matrix[current_city][next_city]
        return total_distance
    
    def _select_next_city(self, ant_path: List[int], unvisited: List[int]) -> int:
        """
        Select the next city for an ant using the pheromone and heuristic information.
        
        Uses the probability formula: P(i,j) = [τ(i,j)^α * η(i,j)^β] / Σ[τ(i,k)^α * η(i,k)^β]
        where τ is pheromone and η is heuristic (1/distance)
        """
        current_city = ant_path[-1]
        
        # Calculate pheromone and heuristic values
        pheromone_values = self.pheromone_matrix[current_city][unvisited]
        heuristic_values = 1.0 / (self.distance_matrix[current_city][unvisited] + 1e-10)
        
        # Calculate probabilities
        probabilities = (pheromone_values ** self.alpha) * (heuristic_values ** self.beta)
        probabilities = probabilities / np.sum(probabilities)
        
        # Select next city based on probabilities
        next_city_idx = np.random.choice(len(unvisited), p=probabilities)
        return unvisited[next_city_idx]
    
    def _construct_solution(self) -> Tuple[List[int], float]:
        """Construct a solution for one ant."""
        # Start from a random city
        start_city = random.randint(0, self.n_cities - 1)
        ant_path = [start_city]
        unvisited = list(range(self.n_cities))
        unvisited.remove(start_city)
        
        # Build the tour
        while unvisited:
            next_city = self._select_next_city(ant_path, unvisited)
            ant_path.append(next_city)
            unvisited.remove(next_city)
        
        # Calculate tour length
        tour_length = self._calculate_tour_length(ant_path)
        
        return ant_path, tour_length
    
    def _update_pheromone(self, all_solutions: List[Tuple[List[int], float]]):
        """Update pheromone trails based on ant solutions."""
        # Evaporate pheromone
        self.pheromone_matrix *= (1 - self.decay)
        
        # Deposit pheromone for each ant
        for solution, distance in all_solutions:
            pheromone_deposit = self.Q / distance
            for i in range(len(solution)):
                current_city = solution[i]
                next_city = solution[(i + 1) % len(solution)]
                self.pheromone_matrix[current_city][next_city] += pheromone_deposit
                self.pheromone_matrix[next_city][current_city] += pheromone_deposit
        
        # Elitist update for best solution
        if self.best_solution is not None:
            elitist_deposit = self.Q / self.best_distance * self.elitist_factor
            for i in range(len(self.best_solution)):
                current_city = self.best_solution[i]
                next_city = self.best_solution[(i + 1) % len(self.best_solution)]
                self.pheromone_matrix[current_city][next_city] += elitist_deposit
                self.pheromone_matrix[next_city][current_city] += elitist_deposit
    
    def run(self) -> Tuple[List[int], float]:
        """
        Run the ACO algorithm.
        
        Returns:
            Tuple of (best_solution, best_distance)
        """
        print(f"Starting ACO with {self.n_ants} ants for {self.n_iterations} iterations...")
        start_time = time.time()
        
        for iteration in range(self.n_iterations):
            # Construct solutions for all ants
            all_solutions = []
            for _ in range(self.n_ants):
                solution, distance = self._construct_solution()
                all_solutions.append((solution, distance))
            
            # Find best solution in this iteration
            iteration_best_solution, iteration_best_distance = min(all_solutions, key=lambda x: x[1])
            
            # Update global best solution
            if iteration_best_distance < self.best_distance:
                self.best_solution = iteration_best_solution.copy()
                self.best_distance = iteration_best_distance
                print(f"Iteration {iteration + 1}: New best distance = {self.best_distance:.2f}")
            
            # Update pheromone trails
            self._update_pheromone(all_solutions)
            
            # Track history
            self.best_solutions_history.append(self.best_distance)
            avg_distance = np.mean([distance for _, distance in all_solutions])
            self.avg_solutions_history.append(avg_distance)
            
            # Progress update
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.n_iterations} - Best: {self.best_distance:.2f}, Avg: {avg_distance:.2f}")
        
        end_time = time.time()
        print(f"\nACO completed in {end_time - start_time:.2f} seconds")
        print(f"Best solution found: {self.best_distance:.2f}")
        print(f"Best tour: {self.best_solution}")
        
        return self.best_solution, self.best_distance
    
    def plot_results(self):
        """Plot the optimization results and the best tour."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot convergence
        ax1.plot(self.best_solutions_history, label='Best Solution', color='red', linewidth=2)
        ax1.plot(self.avg_solutions_history, label='Average Solution', color='blue', alpha=0.7)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Tour Length')
        ax1.set_title('ACO Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot best tour
        if self.best_solution:
            tour_cities = self.cities[self.best_solution + [self.best_solution[0]]]
            ax2.plot(tour_cities[:, 0], tour_cities[:, 1], 'b-', linewidth=2, alpha=0.7)
            ax2.scatter(tour_cities[:-1, 0], tour_cities[:-1, 1], c='red', s=100, zorder=5)
            ax2.scatter(tour_cities[0, 0], tour_cities[0, 1], c='green', s=150, marker='s', zorder=5, label='Start/End')
            
            # Add city labels
            for i, city in enumerate(self.best_solution):
                ax2.annotate(str(city), (self.cities[city, 0], self.cities[city, 1]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax2.set_xlabel('X Coordinate')
            ax2.set_ylabel('Y Coordinate')
            ax2.set_title(f'Best Tour (Length: {self.best_distance:.2f})')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("aco_result.png")  # Save the plot as a PNG file
        print("Plot saved as aco_result.png")
        # plt.show()  # Commented out to prevent showing the plot interactively
    
    def get_statistics(self) -> dict:
        """Get statistics about the optimization run."""
        return {
            'best_distance': self.best_distance,
            'best_solution': self.best_solution,
            'initial_avg_distance': self.avg_solutions_history[0] if self.avg_solutions_history else None,
            'final_avg_distance': self.avg_solutions_history[-1] if self.avg_solutions_history else None,
            'improvement_percentage': ((self.avg_solutions_history[0] - self.best_distance) / self.avg_solutions_history[0] * 100) if self.avg_solutions_history else None,
            'convergence_iteration': np.argmin(self.best_solutions_history) if self.best_solutions_history else None
        } 