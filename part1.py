import time
import sys
import numpy as np
import os
import random

#=====ALGORITHM 1: NEAREST NEIGHBOR (NN)======
def nearest_neighbor(adj_matrix):

    num_cities = len(adj_matrix)
    visited = [False]*num_cities #list to track visited cities. all start as false(unvisited)
    path = []                    #list to store the order of visited cities
    total_distance = 0.0         #total distance of the path

    #start at first city (city 0)
    current_city = 0
    path.append(current_city)
    visited[current_city] = True #mark city 0 as visited

    # visit all the other cities
    for _ in range(num_cities - 1):
        min_distance = float('inf')  # start with a very large number
        next_city = -1  # this will store the index of the next city we go to

        # look at all cities to find the nearest unvisited neighbor
        for neighbor in range(num_cities):
            # if the city is unvisited and isn't the current city
            if not visited[neighbor]:

                distance = adj_matrix[current_city][neighbor]

                if distance < min_distance:
                    min_distance = distance
                    next_city = neighbor

        # now we found the next city to visit
        total_distance += min_distance
        current_city = next_city
        path.append(current_city)
        visited[current_city] = True  # mark this new city as visited

    # after visiting all cities, return to the start (City 0)
    return_distance = adj_matrix[current_city][0]
    total_distance += return_distance
    path.append(0) #add the start city again to show the return

    return total_distance, path

#=====ALGORITHM 2: NEAREST NEIGHBOR WITH 2-OPT======
def calculate_path_distance(path, adj_matrix):
    #calculate total distance of a path
    total_distance = 0.0
    for i in range(len(path) - 1):
        total_distance += adj_matrix[path[i]][path[i+1]]
    return total_distance

def two_opt_swap(path, i, j):
    #reverse the segment between i and j
    new_path = path[:i] + list(reversed(path[i:j+1])) + path[j+1:]
    return new_path

def nearest_neighbor_2opt(adj_matrix):
    #NN with 2-opt optimization
    # first get initial solution from NN
    score, path = nearest_neighbor(adj_matrix)
    n = len(path)
    
    improved = True
    while improved:
        improved = False
        best_score = score
        
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                # don't swap adjacent edges
                if j == i + 1:
                    continue
                    
                # calculate current edges to be removed
                old_edge1 = adj_matrix[path[i-1]][path[i]]
                old_edge2 = adj_matrix[path[j]][path[j+1]]
                
                # calculate new edges that would be added
                new_edge1 = adj_matrix[path[i-1]][path[j]]
                new_edge2 = adj_matrix[path[i]][path[j+1]]
                
                # check if swap improves the solution
                if new_edge1 + new_edge2 < old_edge1 + old_edge2:
                    new_path = two_opt_swap(path, i, j)
                    new_score = calculate_path_distance(new_path, adj_matrix)
                    
                    if new_score < best_score:
                        path = new_path
                        score = new_score
                        best_score = new_score
                        improved = True
                        break  # restart the search after improvement
            
            if improved:
                break
                
    return score, path

#=====ALGORITHM 3: REPEATED RANDOM NEAREST NEIGHBOR (RRNN) WITH 2-OPT======
def randomized_nearest_neighbor(adj_matrix, k):
    #Randomized version of nearest neighbor - chooses randomly from k closest cities
    num_cities = len(adj_matrix)
    visited = [False] * num_cities
    path = []
    total_distance = 0.0

    current_city = 0
    path.append(current_city)
    visited[current_city] = True

    for _ in range(num_cities - 1):
        distances = []
        for neighbor in range(num_cities):
            if not visited[neighbor]:
                distance = adj_matrix[current_city][neighbor]
                distances.append((distance, neighbor))
        
        distances.sort(key=lambda x: x[0])
        k_closest = distances[:min(k, len(distances))]
        
        if k_closest:
            chosen_distance, next_city = random.choice(k_closest)
            total_distance += chosen_distance
            current_city = next_city
            path.append(current_city)
            visited[current_city] = True
        else:
            break

    return_distance = adj_matrix[current_city][0]
    total_distance += return_distance
    path.append(0)
    
    return total_distance, path

def randomized_nearest_neighbor_2opt(adj_matrix, k, num_repeats):
    #RRNN with 2-opt optimization - runs multiple times and returns best solution
    best_score = float('inf')
    best_path = []
    
    for repeat in range(num_repeats):
        score, path = randomized_nearest_neighbor(adj_matrix, k)
        
        n = len(path)
        improved = True
        
        while improved:
            improved = False
            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):
                    if j == i + 1:
                        continue
                    
                    old_edge1 = adj_matrix[path[i-1]][path[i]]
                    old_edge2 = adj_matrix[path[j]][path[j+1]]
                    new_edge1 = adj_matrix[path[i-1]][path[j]]
                    new_edge2 = adj_matrix[path[i]][path[j+1]]
                    
                    if new_edge1 + new_edge2 < old_edge1 + old_edge2:
                        new_path = two_opt_swap(path, i, j)
                        new_score = calculate_path_distance(new_path, adj_matrix)
                        
                        if new_score < score:
                            path = new_path
                            score = new_score
                            improved = True
                            break
                if improved:
                    break
        
        if score < best_score:
            best_score = score
            best_path = path
    
    return best_score, best_path

# ==================== PROCESSING FUNCTIONS ====================
def run_algorithm_with_timing(algorithm_func, matrix, *args):
    """Run any algorithm with timing"""
    num_runs = 1
    start_wall = time.time()
    start_cpu = time.process_time()
    
    if args:
        total_dist, path = algorithm_func(matrix, *args)
    else:
        total_dist, path = algorithm_func(matrix)
    
    end_wall = time.time()
    end_cpu = time.process_time()
    
    wall_time = end_wall - start_wall
    cpu_time = end_cpu - start_cpu
    
    if cpu_time < 0.000001:
        num_runs = 10
        start_wall = time.time()
        start_cpu = time.process_time()
        
        for _ in range(num_runs):
            if args:
                total_dist, path = algorithm_func(matrix, *args)
            else:
                total_dist, path = algorithm_func(matrix)
            
        end_wall = time.time()
        end_cpu = time.process_time()
        
        wall_time = (end_wall - start_wall) / num_runs
        cpu_time = (end_cpu - start_cpu) / num_runs
        
        print(f"    (Ran {num_runs} times due to small CPU time)")
    
    return total_dist, wall_time, cpu_time, path

def get_sizes_from_args():
    #get sizes from command line or use defaults
    DEFAULT_SIZES = [5, 10, 15, 20, 25, 30]
    
    if len(sys.argv) > 1:
        try:
            sizes = [int(arg) for arg in sys.argv[1:]]
            print(f"Processing selected sizes: {sizes}")
            return sizes
        except ValueError:
            print("Invalid size argument. Using default sizes.")
            return DEFAULT_SIZES
    else:
        print("No sizes specified. Using default sizes: ", DEFAULT_SIZES)
        return DEFAULT_SIZES

def find_global_optimal_hyperparameters(folder_path, sizes_to_process, k_values, num_repeats_values):
    #Find one set of optimal hyperparameters that works well across all sizes"""
    print("Finding global optimal hyperparameters for RRNN...")
    
    test_matrices = []
    
    for size in sizes_to_process:
        matrices_found = 0
        for i in range(10):
            filename = os.path.join(folder_path, f"{size}_random_adj_mat_{i}.txt")
            if os.path.exists(filename):
                test_matrices.append((size, np.loadtxt(filename)))
                matrices_found += 1
                if matrices_found >= 2:
                    break
    
    print(f"  Testing on {len(test_matrices)} matrices from sizes: {sorted(set(size for size, _ in test_matrices))}")
    
    k_results = {}
    test_num_repeats = 10
    
    for k in k_values:
        k_scores = []
        for size, matrix in test_matrices:
            scores = []
            for _ in range(2):
                score, _ = randomized_nearest_neighbor_2opt(matrix, k, test_num_repeats)
                scores.append(score)
            k_scores.extend(scores)
        
        median_score = np.median(k_scores)
        k_results[k] = median_score
        print(f"    k={k}: Global Median Score={median_score:.6f}")
    
    best_k = min(k_results, key=k_results.get)
    print(f"  Best global k: {best_k} (score: {k_results[best_k]:.6f})")
    
    repeats_results = {}
    
    for num_repeats in num_repeats_values:
        repeats_scores = []
        for size, matrix in test_matrices:
            scores = []
            for _ in range(2):
                score, _ = randomized_nearest_neighbor_2opt(matrix, best_k, num_repeats)
                scores.append(score)
            repeats_scores.extend(scores)
        
        median_score = np.median(repeats_scores)
        repeats_results[num_repeats] = median_score
        print(f"    num_repeats={num_repeats}: Global Median Score={median_score:.6f}")
    
    best_num_repeats = min(repeats_results, key=repeats_results.get)

    # Look for significant improvements and don't stop too early
    sorted_repeats = sorted(num_repeats_values)
    best_idx = 0
    for i in range(1, len(sorted_repeats)):
        current = sorted_repeats[i]
        prev = sorted_repeats[i-1]
        improvement = repeats_results[prev] - repeats_results[current]
        
        # Continue if we're still seeing good improvements
        if improvement > 0.02:  # More aggressive threshold
            best_idx = i
        # Also continue if we haven't tried many values yet
        elif i < 3:  # Don't stop too early
            best_idx = i
        else:
            break

    best_num_repeats = sorted_repeats[best_idx]
    
    print(f"  Best global num_repeats: {best_num_repeats} (score: {repeats_results[best_num_repeats]:.6f})")
    print(f"  Final global hyperparameters: k={best_k}, num_repeats={best_num_repeats}")
    
    return best_k, best_num_repeats, k_results, repeats_results

def process_all_matrices(folder_path="mats_911", sizes_to_process=None):
    #Process all matrices for NN, NN-2opt, and RRNN algorithms with global hyperparameters
    if sizes_to_process is None:
        sizes_to_process = [5, 10, 15, 20, 25, 30]
    
    k_values = [1, 2, 3, 5, 8, 9]
    num_repeats_values = [1, 3, 5, 10, 20, 30]
    
    algorithms = {
        'NN': nearest_neighbor,
        'NN_2opt': nearest_neighbor_2opt
    }
    
    all_results = []
    
    best_k, best_num_repeats, k_results, repeats_results = find_global_optimal_hyperparameters(
        folder_path, sizes_to_process, k_values, num_repeats_values)
    
    print(f"\nUsing global hyperparameters for all sizes: k={best_k}, num_repeats={best_num_repeats}")
    
    hyperparam_results = {
        'k_values': k_values,
        'k_scores': [k_results[k] for k in k_values],
        'repeats_values': num_repeats_values,
        'repeats_scores': [repeats_results[r] for r in num_repeats_values]
    }
    
    for size in sizes_to_process:
        print(f"\nProcessing size {size}...")
        size_results = {'size': size}
        
        for algo_name, algo_func in algorithms.items():
            print(f"  Running {algo_name}...")
            scores = []
            wall_times = []
            cpu_times = []
            paths = []
            matrices_processed = 0
            
            for i in range(10):
                filename = os.path.join(folder_path, f"{size}_random_adj_mat_{i}.txt")
                if not os.path.exists(filename):
                    continue
                    
                try:
                    matrix = np.loadtxt(filename)
                    score, wall_time, cpu_time, path = run_algorithm_with_timing(algo_func, matrix)
                    
                    scores.append(score)
                    wall_times.append(wall_time)
                    cpu_times.append(cpu_time)
                    paths.append(path)
                    matrices_processed += 1
                    
                    print(f"    Matrix {i}: Score={score:.6f}, CPU Time={cpu_time:.6f}s")
                    
                except Exception as e:
                    print(f"    Error processing {filename}: {e}")
            
            if scores:
                best_idx = np.argmin(scores)
                size_results.update({
                    f'{algo_name}_median_score': np.median(scores),
                    f'{algo_name}_median_wall_time': np.median(wall_times),
                    f'{algo_name}_median_cpu_time': np.median(cpu_times),
                    f'{algo_name}_min_score': scores[best_idx],
                    f'{algo_name}_max_score': np.max(scores),
                    f'{algo_name}_best_path': paths[best_idx],
                    f'{algo_name}_matrices_processed': matrices_processed
                })
        
        print(f"  Running RRNN with global hyperparameters: k={best_k}, num_repeats={best_num_repeats}...")
        rrn_scores = []
        rrn_wall_times = []
        rrn_cpu_times = []
        rrn_paths = []
        rrn_matrices_processed = 0
        
        for i in range(10):
            filename = os.path.join(folder_path, f"{size}_random_adj_mat_{i}.txt")
            if not os.path.exists(filename):
                continue
                
            try:
                matrix = np.loadtxt(filename)
                
                num_runs = 1
                start_wall = time.time()
                start_cpu = time.process_time()
                
                score, path = randomized_nearest_neighbor_2opt(matrix, best_k, best_num_repeats)
                
                end_wall = time.time()
                end_cpu = time.process_time()
                
                wall_time = end_wall - start_wall
                cpu_time = end_cpu - start_cpu
                
                if cpu_time < 0.000001:
                    num_runs = 3
                    start_wall = time.time()
                    start_cpu = time.process_time()
                    
                    for _ in range(num_runs):
                        score, path = randomized_nearest_neighbor_2opt(matrix, best_k, best_num_repeats)
                        
                    end_wall = time.time()
                    end_cpu = time.process_time()
                    
                    wall_time = (end_wall - start_wall) / num_runs
                    cpu_time = (end_cpu - start_cpu) / num_runs
                
                rrn_scores.append(score)
                rrn_wall_times.append(wall_time)
                rrn_cpu_times.append(cpu_time)
                rrn_paths.append(path)
                rrn_matrices_processed += 1
                
                print(f"    Matrix {i}: Score={score:.6f}, CPU Time={cpu_time:.6f}s")
                
            except Exception as e:
                print(f"    Error processing {filename}: {e}")
        
        if rrn_scores:
            best_idx = np.argmin(rrn_scores)
            size_results.update({
                'RRNN_median_score': np.median(rrn_scores),
                'RRNN_median_wall_time': np.median(rrn_wall_times),
                'RRNN_median_cpu_time': np.median(rrn_cpu_times),
                'RRNN_min_score': rrn_scores[best_idx],
                'RRNN_max_score': np.max(rrn_scores),
                'RRNN_best_path': rrn_paths[best_idx],
                'RRNN_matrices_processed': rrn_matrices_processed
            })
        
        all_results.append(size_results)
        print(f"  Size {size} completed")
    
    return all_results, hyperparam_results, (best_k, best_num_repeats)

#========================PLOTTING=======================================
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")  # clean style

def plot_rrnn_hyperparams(hyperparam_results, best_k, best_repeats):
    # k vs score
    plt.figure(figsize=(7,5))
    plt.plot(hyperparam_results['k_values'], hyperparam_results['k_scores'],
             marker='o', markersize=10, linewidth=2, label="Median Score")
    plt.axvline(best_k, color='r', linestyle='--', linewidth=1.5, label=f'Optimal k={best_k}')
    plt.xlabel("k (nearest neighbors considered)", fontsize=12, fontweight='bold')
    plt.ylabel("Median Tour Score", fontsize=12, fontweight='bold')
    plt.title("RRNN Hyperparameter Tuning: k", fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig("rrnn_k_vs_score.png", dpi=300)

    # num_repeats vs score
    plt.figure(figsize=(7,5))
    plt.plot(hyperparam_results['repeats_values'], hyperparam_results['repeats_scores'],
             marker='s', markersize=8, linewidth=2, label="Median Score")
    plt.axvline(best_repeats, color='r', linestyle='--', linewidth=1.5, label=f'Optimal repeats={best_repeats}')
    plt.xlabel("num_repeats", fontsize=12, fontweight='bold')
    plt.ylabel("Median Tour Score", fontsize=12, fontweight='bold')
    plt.title("RRNN Hyperparameter Tuning: num_repeats", fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig("rrnn_repeats_vs_score.png", dpi=300)

def plot_comparisons(results):
    sizes = [r['size'] for r in results]

    # Wall time
    plt.figure(figsize=(7,5))
    plt.plot(sizes, [r['NN_median_wall_time'] for r in results],
             marker='o', markersize=7, linewidth=2, label="NN")
    plt.plot(sizes, [r['NN_2opt_median_wall_time'] for r in results],
             marker='s', markersize=7, linewidth=2, label="NN2O")
    plt.plot(sizes, [r['RRNN_median_wall_time'] for r in results],
             marker='^', markersize=7, linewidth=2, label="RRNN")
    plt.xlabel("Number of Cities", fontsize=12, fontweight='bold')
    plt.ylabel("Runtime (s)", fontsize=12, fontweight='bold')
    plt.title("Runtime Comparison Across Algorithms (NN, NN2O, RNN)", fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig("run_time_vs_size.png", dpi=300)

    # CPU time
    plt.figure(figsize=(7,5))
    plt.plot(sizes, [r['NN_median_cpu_time'] for r in results],
             marker='o', markersize=7, linewidth=2, label="NN")
    plt.plot(sizes, [r['NN_2opt_median_cpu_time'] for r in results],
             marker='s', markersize=7, linewidth=2, label="NN2O")
    plt.plot(sizes, [r['RRNN_median_cpu_time'] for r in results],
             marker='^', markersize=7, linewidth=2, label="RRNN")
    plt.xlabel("Number of Cities", fontsize=12, fontweight='bold')
    plt.ylabel("CPU Time (s)", fontsize=12, fontweight='bold')
    plt.title("CPU Time Comparison Across Algorithms (NN, NN2O, RNN)", fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig("cpu_time_vs_size.png", dpi=300)

    # Scores
    plt.figure(figsize=(7,5))
    plt.plot(sizes, [r['NN_median_score'] for r in results],
             marker='o', markersize=7, linewidth=2, label="NN")
    plt.plot(sizes, [r['NN_2opt_median_score'] for r in results],
             marker='s', markersize=7, linewidth=2, label="NN2O")
    plt.plot(sizes, [r['RRNN_median_score'] for r in results],
             marker='^', markersize=7, linewidth=2, label="RRNN")
    plt.xlabel("Number of Cities", fontsize=12, fontweight='bold')
    plt.ylabel("Score", fontsize=12, fontweight='bold')
    plt.title("Cost Comparison Across Algorithms (NN, NN2O, RNN)", fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig("scores_vs_size.png", dpi=300)

# Main output section
# Detailed results section to include best paths
if __name__ == "__main__":
    sizes_to_process = get_sizes_from_args()
    
    folder_path = "mats_911"
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found!")
        sys.exit(1)
    
    print(f"Processing matrices in folder: {folder_path}")
    print("="*60)
    
    results, hyperparam_results, global_hyperparams = process_all_matrices(folder_path, sizes_to_process)
    best_k, best_num_repeats = global_hyperparams
    
    # Output results
    print("\n" + "="*180)
    print("FINAL RESULTS SUMMARY - NN vs NN-2OPT vs RRNN")
    print("="*180)
    print(f"Global Hyperparameters: k={best_k}, num_repeats={best_num_repeats}")
    print("="*180)
    print("Size | NN Median Score | NN-2opt Median Score | RRNN Median Score | NN Wall Time | NN-2opt Wall Time | RRNN Wall Time | NN CPU Time | NN-2opt CPU Time | RRNN CPU Time")
    print("-" * 180)
    
    for result in results:
        print(f"{result['size']:4} | {result['NN_median_score']:15.6f} | {result['NN_2opt_median_score']:18.6f} | "
              f"{result['RRNN_median_score']:17.6f} | {result['NN_median_wall_time']:11.6f}s | "
              f"{result['NN_2opt_median_wall_time']:14.6f}s | {result['RRNN_median_wall_time']:13.6f}s | "
              f"{result['NN_median_cpu_time']:10.6f}s | {result['NN_2opt_median_cpu_time']:13.6f}s | "
              f"{result['RRNN_median_cpu_time']:12.6f}s")

    print("\n" + "="*180)
    print("DETAILED RESULTS BY SIZE:")
    print("="*180)
    
    for result in results:
        print(f"\nSize {result['size']}:")
        print(f"  NN -      Median Score: {result['NN_median_score']:.6f}, Wall Time: {result['NN_median_wall_time']:.6f}s, CPU Time: {result['NN_median_cpu_time']:.6f}s")
        print(f"           Best Score: {result['NN_min_score']:.6f}, Best Path: {result['NN_best_path']}")
        
        print(f"  NN-2opt - Median Score: {result['NN_2opt_median_score']:.6f}, Wall Time: {result['NN_2opt_median_wall_time']:.6f}s, CPU Time: {result['NN_2opt_median_cpu_time']:.6f}s")
        print(f"           Best Score: {result['NN_2opt_min_score']:.6f}, Best Path: {result['NN_2opt_best_path']}")
        
        print(f"  RRNN -    Median Score: {result['RRNN_median_score']:.6f}, Wall Time: {result['RRNN_median_wall_time']:.6f}s, CPU Time: {result['RRNN_median_cpu_time']:.6f}s")
        print(f"           Best Score: {result['RRNN_min_score']:.6f}, Best Path: {result['RRNN_best_path']}")
        
        print(f"  Improvement over NN: {result['NN_median_score'] - result['RRNN_median_score']:.6f}")
        print(f"  Improvement over NN-2opt: {result['NN_2opt_median_score'] - result['RRNN_median_score']:.6f}")

    # Hyperparameter results
    print("\n" + "="*140)
    print("HYPERPARAMETER TUNING RESULTS (for plots):")
    print("="*140)
    
    print("k values vs Median Scores:")
    for k, score in zip(hyperparam_results['k_values'], hyperparam_results['k_scores']):
        optimal_indicator = "  <-- OPTIMAL" if k == best_k else ""
        print(f"  k={k}: {score:.6f}{optimal_indicator}")
    
    print(f"\nSelected k = {best_k} as optimal")
    
    print("\nnum_repeats values vs Median Scores:")
    for repeats, score in zip(hyperparam_results['repeats_values'], hyperparam_results['repeats_scores']):
        optimal_indicator = "  <-- OPTIMAL" if repeats == best_num_repeats else ""
        print(f"  num_repeats={repeats}: {score:.6f}{optimal_indicator}")
    
    print(f"\nSelected num_repeats = {best_num_repeats} as optimal")

    print("\nProcess Complete!! :)\n")

    # Generate and save figures
    plot_rrnn_hyperparams(hyperparam_results, best_k, best_num_repeats)
    plot_comparisons(results)