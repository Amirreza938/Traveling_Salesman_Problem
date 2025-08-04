#!/usr/bin/env python3
"""
Main script to demonstrate Ant Colony Optimization (ACO) for solving the Traveling Salesman Problem (TSP).

This script showcases different configurations and problem sizes to demonstrate the effectiveness
of the ACO algorithm.
"""

from aco import AntColonyOptimization
import matplotlib.pyplot as plt
import numpy as np
import time

def run_basic_example():
    """Run a basic ACO example with default parameters."""
    print("=" * 60)
    print("BASIC ACO EXAMPLE - Traveling Salesman Problem")
    print("=" * 60)
    
    # Create ACO instance with 20 cities
    aco = AntColonyOptimization(
        n_cities=20,
        n_ants=30,
        n_iterations=50,
        random_seed=42
    )
    
    # Run the algorithm
    best_solution, best_distance = aco.run()
    
    # Display results
    print("\n" + "=" * 40)
    print("RESULTS")
    print("=" * 40)
    stats = aco.get_statistics()
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Plot results
    aco.plot_results()
    
    return aco

def run_parameter_comparison():
    """Compare different parameter settings."""
    print("\n" + "=" * 60)
    print("PARAMETER COMPARISON STUDY")
    print("=" * 60)
    
    # Define different parameter configurations
    configurations = [
        {
            'name': 'Conservative (High α, Low β)',
            'alpha': 2.0,
            'beta': 1.0,
            'decay': 0.1
        },
        {
            'name': 'Balanced (Default)',
            'alpha': 1.0,
            'beta': 2.0,
            'decay': 0.1
        },
        {
            'name': 'Explorative (Low α, High β)',
            'alpha': 0.5,
            'beta': 3.0,
            'decay': 0.1
        },
        {
            'name': 'Fast Evaporation',
            'alpha': 1.0,
            'beta': 2.0,
            'decay': 0.3
        }
    ]
    
    results = []
    
    for config in configurations:
        print(f"\nTesting: {config['name']}")
        print("-" * 40)
        
        aco = AntColonyOptimization(
            n_cities=15,
            n_ants=25,
            n_iterations=30,
            alpha=config['alpha'],
            beta=config['beta'],
            decay=config['decay'],
            random_seed=42
        )
        
        start_time = time.time()
        best_solution, best_distance = aco.run()
        end_time = time.time()
        
        results.append({
            'name': config['name'],
            'best_distance': best_distance,
            'execution_time': end_time - start_time,
            'final_avg': aco.avg_solutions_history[-1] if aco.avg_solutions_history else None
        })
    
    # Display comparison results
    print("\n" + "=" * 60)
    print("PARAMETER COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Configuration':<25} {'Best Distance':<15} {'Time (s)':<10} {'Final Avg':<15}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['name']:<25} {result['best_distance']:<15.2f} "
              f"{result['execution_time']:<10.2f} {result['final_avg']:<15.2f}")
    
    return results

def run_scalability_test():
    """Test ACO performance with different problem sizes."""
    print("\n" + "=" * 60)
    print("SCALABILITY TEST")
    print("=" * 60)
    
    problem_sizes = [10, 20, 30, 40, 50]
    results = []
    
    for n_cities in problem_sizes:
        print(f"\nTesting with {n_cities} cities...")
        
        # Adjust parameters based on problem size
        n_ants = min(50, n_cities * 2)
        n_iterations = min(100, max(20, 200 // n_cities))
        
        aco = AntColonyOptimization(
            n_cities=n_cities,
            n_ants=n_ants,
            n_iterations=n_iterations,
            random_seed=42
        )
        
        start_time = time.time()
        best_solution, best_distance = aco.run()
        end_time = time.time()
        
        results.append({
            'n_cities': n_cities,
            'best_distance': best_distance,
            'execution_time': end_time - start_time,
            'n_ants': n_ants,
            'n_iterations': n_iterations
        })
    
    # Plot scalability results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Execution time vs problem size
    cities = [r['n_cities'] for r in results]
    times = [r['execution_time'] for r in results]
    ax1.plot(cities, times, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Cities')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Scalability: Execution Time')
    ax1.grid(True, alpha=0.3)
    
    # Solution quality vs problem size
    distances = [r['best_distance'] for r in results]
    ax2.plot(cities, distances, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Cities')
    ax2.set_ylabel('Best Tour Length')
    ax2.set_title('Scalability: Solution Quality')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Display scalability results
    print("\n" + "=" * 50)
    print("SCALABILITY RESULTS")
    print("=" * 50)
    print(f"{'Cities':<8} {'Best Distance':<15} {'Time (s)':<10} {'Ants':<6} {'Iterations':<12}")
    print("-" * 55)
    
    for result in results:
        print(f"{result['n_cities']:<8} {result['best_distance']:<15.2f} "
              f"{result['execution_time']:<10.2f} {result['n_ants']:<6} {result['n_iterations']:<12}")
    
    return results

def run_convergence_analysis():
    """Analyze convergence behavior with different iteration counts."""
    print("\n" + "=" * 60)
    print("CONVERGENCE ANALYSIS")
    print("=" * 60)
    
    iteration_counts = [20, 50, 100, 200]
    convergence_data = []
    
    for n_iterations in iteration_counts:
        print(f"\nTesting with {n_iterations} iterations...")
        
        aco = AntColonyOptimization(
            n_cities=25,
            n_ants=40,
            n_iterations=n_iterations,
            random_seed=42
        )
        
        best_solution, best_distance = aco.run()
        
        convergence_data.append({
            'n_iterations': n_iterations,
            'best_distance': best_distance,
            'convergence_history': aco.best_solutions_history.copy(),
            'avg_history': aco.avg_solutions_history.copy()
        })
    
    # Plot convergence comparison
    plt.figure(figsize=(12, 8))
    
    for data in convergence_data:
        iterations = range(1, data['n_iterations'] + 1)
        plt.plot(iterations, data['convergence_history'], 
                label=f"{data['n_iterations']} iterations (Best: {data['best_distance']:.2f})",
                linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Best Tour Length')
    plt.title('Convergence Analysis: Best Solution vs Iterations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return convergence_data

def main():
    """Main function to run all demonstrations."""
    print("ANT COLONY OPTIMIZATION DEMONSTRATION")
    print("Solving the Traveling Salesman Problem (TSP)")
    print("=" * 60)
    
    try:
        # Run basic example
        basic_aco = run_basic_example()
        
        # Run parameter comparison
        param_results = run_parameter_comparison()
        
        # Run scalability test
        scalability_results = run_scalability_test()
        
        # Run convergence analysis
        convergence_results = run_convergence_analysis()
        
        print("\n" + "=" * 60)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nSummary:")
        print("- Basic ACO example with 20 cities")
        print("- Parameter comparison with 4 different configurations")
        print("- Scalability test with 5 different problem sizes")
        print("- Convergence analysis with 4 different iteration counts")
        print("\nThe ACO algorithm successfully demonstrates:")
        print("✓ Pheromone trail optimization")
        print("✓ Heuristic-based city selection")
        print("✓ Convergence to good solutions")
        print("✓ Scalability with problem size")
        print("✓ Parameter sensitivity analysis")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 