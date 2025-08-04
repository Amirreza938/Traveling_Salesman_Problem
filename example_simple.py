#!/usr/bin/env python3
"""
Simple example demonstrating Ant Colony Optimization for the Traveling Salesman Problem.

This script provides a quick way to test the ACO algorithm with minimal setup.
"""

from aco import AntColonyOptimization
import matplotlib.pyplot as plt

def simple_example():
    """Run a simple ACO example."""
    print("Simple Ant Colony Optimization Example")
    print("=" * 50)
    
    # Create a small TSP problem with 15 cities
    aco = AntColonyOptimization(
        n_cities=15,
        n_ants=20,
        n_iterations=30,
        random_seed=42
    )
    
    print(f"Problem: {aco.n_cities} cities")
    print(f"Colony: {aco.n_ants} ants")
    print(f"Iterations: {aco.n_iterations}")
    print("\nStarting optimization...")
    
    # Run the algorithm
    best_solution, best_distance = aco.run()
    
    # Display results
    print(f"\nResults:")
    print(f"Best tour length: {best_distance:.2f}")
    print(f"Best tour: {best_solution}")
    
    # Show statistics
    stats = aco.get_statistics()
    print(f"\nStatistics:")
    print(f"Initial average distance: {stats['initial_avg_distance']:.2f}")
    print(f"Final average distance: {stats['final_avg_distance']:.2f}")
    print(f"Improvement: {stats['improvement_percentage']:.1f}%")
    
    # Plot the results
    print("\nGenerating plots...")
    aco.plot_results()
    
    return aco

def quick_test():
    """Run a very quick test with minimal parameters."""
    print("⚡ Quick ACO Test")
    print("=" * 30)
    
    aco = AntColonyOptimization(
        n_cities=10,
        n_ants=10,
        n_iterations=15,
        random_seed=42
    )
    
    best_solution, best_distance = aco.run()
    
    print(f"Quick test completed!")
    print(f"Best distance: {best_distance:.2f}")
    print(f"Tour: {best_solution}")
    
    return aco

if __name__ == "__main__":
    # Run the simple example
    aco = simple_example()
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("=" * 50) 