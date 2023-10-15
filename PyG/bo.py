# To run this demo, you need to rewrite all the 'np.int' in the file '/skopt/space/transformers.py' into 'np.int_' first. 
import numpy as np
from skopt import gp_minimize
import subprocess
from skopt.plots import plot_convergence

import psutil
import argparse
import matplotlib.pyplot as plt
import os

total_cpu = psutil.cpu_count(logical=False)
# define the acquisition function, can be choose from ['LCB', 'EI', 'PI']
acq_func = 'EI'  


# define the objective function
def objective_function(x):
    n_process = x[0]
    n_sampler = x[1]
    n_trainer = x[2]
    # if x in evaluated:
    #     print("already evaluated:", x)
    #     return evaluated[x]
    if n_process*(n_sampler+n_trainer) > total_cpu:
        return max_val
    
    command = ["python", "PyG/gnn_train.py", 
               "--model", arguments.model, 
               "--sampler", arguments.sampler, 
               "--dataset", arguments.dataset, 
               '--cpu_process', str(int(n_process)), 
               '--n_sampler', str(int(n_sampler)), 
               '--n_trainer', str(int(n_trainer)), 
               '--batch_num', str(batch_num), 
               ]
    print(n_process, n_sampler, n_trainer)
    try:
        # Execute the external script and capture its output
        # result = subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True, timeout=600)
        # result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=timeout)
        # result = result.stdout
        result = subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True, timeout=timeout)
        # print(result)
        output_lines = result.split("\n")
        # Search for the line containing "total_time" and extract the value
        objective_value = max_val
        for line in output_lines:
            if "total_time" in line:
                objective_value = float(line.split()[1])
                print("objective_value:", objective_value)
                break
        return objective_value
    except:
        print("External script failed with error")
    return max_val



# Define the searching space of the parameters
# TODO: You need to change the range of the searching space to adapt to your CPU configuration
space = [(2, 4), (1, 4), (1,32)] 

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    type=str,
                    default='flickr',
                    choices=["ogbn-products", "ogbn-papers100M", "reddit", "yelp", "flickr"])

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

command = ["python", "PyG/gnn_train.py", 
           "--dataset", arguments.dataset , 
           "--model", arguments.model, 
           '--sampler', arguments.sampler, 
           '--cpu_process', str(2), 
           '--n_sampler', str(1), 
           '--n_trainer', str(1),
           '--batch_num', str(batch_num),
           ]
# command = ["python", "PyG/demo.py"]
timeout = 1200 if arguments.dataset == 'ogbn-papers100M' else 600 
print("Set timeout: ", timeout)
print("Begin the first run, command: ", command)
# result = subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True, timeout=600)
try:
    
    # result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=timeout)
    result = subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True, timeout=600)
    # print(result)
    max_val = 1000000
    output_lines = result.split("\n")
    for line in output_lines:
        if "total_time" in line:
            max_val = float(line.split()[1])
            break
except subprocess.TimeoutExpired as e:
    print("External script timeout:", e)
    max_val = 100000
print("upper bound:", max_val)

# Run the Bayesian optimization algorithm, using the Gaussian process as the surrogate model
result = gp_minimize(objective_function, space, n_calls=30, random_state=3, acq_func=acq_func)
print("Best parameter:", result.x)
print("Minimum output:", result.fun)

if not os.path.exists("PyG/Result"):
    os.mkdir("PyG/Result")

with open("PyG/Result/Bo_{}_{}.txt".format(arguments.dataset, arguments.model), "a") as text_file:
    text_file.write("Best parameter:" + str(result.x) + "\n")
    text_file.write("Minimum output:" + str(result.fun) + "\n")
    text_file.write("Parameters of each iteration:" + str(result.x_iters) + "\n")
    text_file.write("Output of each iteration:" + str(result.func_vals) + "\n")

plot_convergence(result)
plt.savefig('PyG/Result/convergence_plot_{}.png'.format(arguments.dataset))
