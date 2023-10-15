import math
import psutil

datasets = ['ogbn-products','flickr', 'reddit']
# datasets = ['ogbn-products']
models = ['sage','gcn']
cpu_processes = [i for i in range(2, 5)]
# TODO: You need to set the step size and the range of searching space to adapt to your CPU configuration
step = 2
n_samplers = [i for i in range(1, 5)] 
n_trainers = [step * i for i in range(1, 12)]

max_cores = psutil.cpu_count(logical=False)


lines = []
for dataset in datasets:
    for model in models:
        sampler = 'neighbor' if model == 'sage' else 'shadow'
        for cpu_process in cpu_processes:
            for n_sampler in n_samplers:
                if cpu_process * n_sampler > max_cores:
                    continue
                max_trainer = math.floor((max_cores - cpu_process * n_sampler) / cpu_process)
                # print(max_trainer)
                for n_trainer in n_trainers[:max_trainer]:
                    line = f"timeout 5400 python -W ignore PyG/gnn_train.py --dataset {dataset} --model {model} --sampler {sampler} --record "
                    line += f"--cpu_process {cpu_process} --n_sampler {n_sampler} --n_trainer {n_trainer}"
                    lines.append(line)

with open('PyG/grid_search.sh', 'w') as f:
    f.write('#!/bin/bash\n')
    f.write('\n'.join(lines))