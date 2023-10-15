import random
import math
import subprocess
import argparse
import psutil
import numpy as np

# Define the objective function that calls an external script
def objective_function(x, y, z):
    command = ["python", "PyG/gnn_train.py", 
               "--model", arguments.model, 
               "--sampler", arguments.sampler , 
               "--dataset", arguments.dataset, 
               '--cpu_process', str(int(x)), 
               '--n_sampler', str(int(y)), 
               '--n_trainer', str(int(z)), 
               '--batch_num', str(batch_num)]
    print(command)
    if x*(y+z) > total_cpu:
        print("[ERR] out of cpu_cores")
        return 100
    try:
        # Execute the external script and capture its output
        # print(command)
        result = subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True, timeout=600)
        # print(result)

        output_lines = result.split("\n")
        # Search for the line containing "total_time" and extract the value
        objective_values = []
        for line in output_lines:
            if "total_time" in line:
                objective_value = float(line.split()[1])
                objective_values.append(objective_value)
                break
        if objective_values == []:
            objective_value = max_val
        else:
            objective_value = np.mean(objective_values)
        return objective_value
    
    except:
        # Handle errors if the external script fails
        print("External script failed with error")
        return max_val

# Simulated Annealing algorithm
def simulated_annealing(initial_x, initial_y, initial_z, initial_temperature, cooling_rate, max_iterations):
    current_x, current_y, current_z = initial_x, initial_y, initial_z

    current_value = objective_function(current_x, current_y, current_z)

    best_x, best_y, best_z = current_x, current_y, current_z
    best_value = current_value

    temperature = initial_temperature

    for iteration in range(max_iterations):
        # Generate neighboring integer coordinates
        # TODO: You need to change the range of the neighbor points to adapt to your CPU configuration
        neighbor_x = random.randint(2, 4)
        neighbor_y = random.randint(1, 4)
        neighbor_z = random.randint(1, 15)

        # Calculate the value for the neighbor point
        neighbor_value = objective_function(neighbor_x, neighbor_y, neighbor_z)

        # Calculate the change in value for the neighbor point
        delta_value = neighbor_value - current_value

        # If the neighbor point is better or accepted with a probability, update the current point
        if delta_value < 0 or random.random() < math.exp(-delta_value / temperature):
            current_x, current_y, current_z = neighbor_x, neighbor_y, neighbor_z
            current_value = neighbor_value

            # Update the best point if necessary
            if current_value < best_value:
                best_x, best_y, best_z = current_x, current_y, current_z
                best_value = current_value

        # Reduce the temperature
        temperature *= cooling_rate
        print(round(temperature,2), round(neighbor_value,2), current_x, current_y, current_z)
        with open("PyG/Result/Random_{}_{}.txt".format(arguments.dataset, arguments.model), "a") as text_file:
            text_file.write(str(round(temperature,2)) + " " + str(round(neighbor_value,2)) + " " + str(current_x) + " " + str(current_y) + " " + str(current_z) + "\n")

    return best_x, best_y, best_z, best_value

if __name__ == "__main__":
    initial_x = 2
    initial_y = 1
    initial_z = 8

    initial_temperature = 100.0
    cooling_rate = 0.95
    max_iterations = 30

    total_cpu = psutil.cpu_count(logical=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        default='ogbn-products',
                        choices=["ogbn-papers100M", "ogbn-products", "reddit", "flickr"])
    parser.add_argument(
        "--model",
        type=str,
        default= "sage",
        help="model",
        choices=["sage", "gcn"],
    )

    parser.add_argument(
        "--sampler",
        type=str,
        default= "neighbor",
        help="sampler",
        choices=["shadow", "neighbor"],
    )

    parser.add_argument(
        "--batch_num",
        type=str,
        default = '0',
    )

    arguments = parser.parse_args()
    batch_num = int(arguments.batch_num)
    command = ["python", "PyG/gnn_train.py", "--model", arguments.model, "--sampler", arguments.sampler, "--dataset", arguments.dataset, '--cpu_process', str(int(1)), '--n_sampler', str(int(2)), '--n_trainer', str(int(8)), '--batch_num', str(batch_num)]
    try:
        result = subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True, timeout=600)
        # print(result)
        output_lines = result.split("\n")
        # Search for the line containing "total_time" and extract the value
        max_val = 10000000
        for line in output_lines:
            if "total_time" in line:
                max_val = float(line.split()[1])
                break
    except:
        print("External script failed with Error")
        max_val = 10000
    
        

    best_x, best_y, best_z, best_value = simulated_annealing(initial_x, initial_y, initial_z, initial_temperature, cooling_rate, max_iterations)

    print("Best (x, y, z):", (best_x, best_y, best_z))
    print("Best Value:", best_value)
    with open("PyG/Result/Random_{}_{}.txt".format(arguments.dataset, arguments.model), "a") as text_file:
        text_file.write("Best (x, y, z):" + str((best_x, best_y, best_z)) + "\n")
        text_file.write("Best Value:" + str(best_value) + "\n")
    outp = [arguments.dataset,best_x,best_y,best_value]
