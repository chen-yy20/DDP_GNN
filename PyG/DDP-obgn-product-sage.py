# Reaches around 0.7930 test accuracy.

import argparse
import time

import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from tqdm import tqdm

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv

# new imports
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import psutil
import os
from torch.profiler import profile, ProfilerActivity
import os
# import merge
import subprocess

def get_mask(comp_core):
    mask = 0
    for core in comp_core:
        mask += 2 ** core
    return hex(mask)

def get_core_num(process_num, rank, n, load_core_num=4):
    
    if process_num == 1:
        load_core = list(range(0,load_core_num))
        comp_core = list(range(load_core_num,n))
        all_core = list(range(0,n))
        

    elif process_num == 2:
        if rank == 0:
            load_core = list(range(0,load_core_num))
            comp_core = list(range(load_core_num,n//2))
            all_core = list(range(0,n//2))
            # load_core = [0]
            # comp_core = [1]
            # all_core = [0,1]
        else:
            load_core = list(range(n//2,n//2+load_core_num))
            comp_core = list(range(n//2+load_core_num,n))
            all_core = list(range(n//2,n))
            # load_core = [16,17]
            # comp_core = [19]
            # all_core = [16,17,19]

    elif process_num == 4:
        if rank == 0:
            load_core = list(range(0,load_core_num))
            comp_core = list(range(load_core_num,n//4))
            all_core = list(range(0,n//4))
        elif rank == 1:
            load_core = list(range(n//4,n//4+load_core_num))
            comp_core = list(range(n//4+load_core_num,n//2))
            all_core = list(range(n//4,n//2))
        elif rank == 2:
            load_core = list(range(n//2,n//2+load_core_num))
            comp_core = list(range(n//2+load_core_num,n//4*3))
            all_core = list(range(n//2,n//4*3))
        else:
            load_core = list(range(n//4*3,n//4*3+load_core_num))
            comp_core = list(range(n//4*3+load_core_num,n))
            all_core = list(range(n//4*3,n))
    return load_core, comp_core, all_core

def set_specific_core_num():
    if process_num == 1:
        comp_core = []
        load_core = []
        all_core = load_core + comp_core
    elif process_num == 2:
        if rank == 0:
            load_core = [0]
            comp_core = [1]
            all_core = load_core + comp_core
        else:
            load_core = [16]
            comp_core = [19]
            all_core = load_core + comp_core
    elif process_num == 4:
        if rank == 0:
            comp_core = [0,1,2]
            load_core = [16]
            all_core = load_core + comp_core
        elif rank == 1:
            comp_core = [4,5,6]
            load_core = [17]
            all_core = load_core + comp_core
        elif rank == 2:
            comp_core = [8,9,10]
            load_core = [18]
            all_core = load_core + comp_core
        else:
            comp_core = [12,13,14]
            load_core = [19]
            all_core = load_core + comp_core
    return load_core, comp_core, all_core


parser = argparse.ArgumentParser()

parser.add_argument(
    "--process",
    default= "1",
    choices=["1", "2", "4"],
)

parser.add_argument(
    "--l_core",
    default= "4",
    help="load core number"
)

parser.add_argument(
    "--run_times",
    default= "1",
    help="repeat times to run the experiment",
)
    


args = parser.parse_args()
args.mode = "cpu"
print(f"Training in {args.mode} mode.")
times = int(args.run_times)

process_num = int(args.process)
load_core_num = int(args.l_core)
device = torch.device('cpu')
print("device: ", device, "process_num: ", process_num, "load_core_num: ", load_core_num)
# device = torch.device('cpu')
dataset = PygNodePropPredDataset(name = 'ogbn-products')
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-products')
data = dataset[0].to(device, 'x', 'y')
trace_list = []


subgraph_loader = NeighborLoader(
    data,
    input_nodes=None,
    num_neighbors=[-1],
    batch_size=4096,
    num_workers=4,
    persistent_workers=True,
)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.num_layers):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id].to(device)
                edge_index = batch.edge_index.to(device)
                x = self.convs[i](x, edge_index)
                x = x[:batch.batch_size]
                if i != self.num_layers - 1:
                    x = x.relu()
                xs.append(x.cpu())

                pbar.update(batch.batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all

# 查看physical core 属于哪个NUMA节点
# cat /proc/cpuinfo | grep -E "physical id|core id"


def run(rank, times, model, process_num):
    # 物理cpu的数量
    n = psutil.cpu_count(logical=False) 
    dist.init_process_group('gloo', rank=rank, world_size=process_num)   
    train_idx = split_idx['train'].to(device)

    # 在DDP_time中追加写入一空行
    with open('./DDP_profile/DDP_time.txt', 'a') as f:
        f.write('\n')
        if rank == 0:
            f.write(f'Load_num:{load_core_num}| Process_num:{process_num}\n')
            f.write("=========================================================\n")
    
    for run in range(times):
        model.reset_parameters()
        model = DistributedDataParallel(model)
        if rank == 0:
            print(f'\nRun {run:02d}:\n')
        test_loading_time(rank, train_idx, process_num, load_core_num)
        train(model, rank, train_idx)
        # test_loading_time(rank, train_idx, process_num, load_core_num)

def test_loading_time(rank, train_idx, process_num, load_core_num):
    n = psutil.cpu_count(logical=False)
    load_core, comp_core, all_core = get_core_num(process_num, rank, n, load_core_num)
    # load_core, comp_core, all_core = set_specific_core_num()
    print("rank {}: testing loading time".format(rank))
    train_sampler = DistributedSampler(
            train_idx,
            num_replicas=process_num,
            rank=rank
        )
    train_loader = NeighborLoader(
        data,
        input_nodes=split_idx['train'],
        num_neighbors=[15, 10, 5],
        batch_size=4096//process_num,
        num_workers=len(load_core),
        persistent_workers=True,
        sampler=train_sampler
    )
    start_time = time.time()
    with train_loader.enable_cpu_affinity(loader_cores = load_core):
        for idx,batch in enumerate(train_loader):
            # if idx % 10 == 0:
                # print("rank {}: loading {}th batch".format(rank, idx))
            pass
    
    end_time = time.time()
    # 在DDP_time.txt中追加写入每个进程的加载时间
    avg_load_time = (end_time - start_time) / len(train_loader)
    with open('./DDP_profile/DDP_time.txt', 'a') as f:
        f.write(f'Rank {rank} Load Time: {avg_load_time}\n')
    print(f'Load_num:{load_core_num}| Process_num:{process_num} | Rank {rank} Load Time: {avg_load_time}\n')
        

def train(model, rank, train_idx):
    n = psutil.cpu_count(logical=False)
    load_core, comp_core, all_core = get_core_num(process_num, rank, n, load_core_num)

    # set compute cores 【taskset】
    torch.set_num_threads(len(comp_core))
    pid = os.getpid()
    print("[TASKSET] rank {}, pid: {}".format(rank, pid))
    core_mask = get_mask(comp_core)
    subprocess.run(["taskset", "-a","-p", str(core_mask), str(pid)])
    
    print("[TASKSET] rank {}, using compute core: {}".format(rank, comp_core))


    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        train_sampler = DistributedSampler(
            train_idx,
            num_replicas=process_num,
            rank=rank
        )
        train_loader = NeighborLoader(
            data,
            input_nodes=split_idx['train'],
            num_neighbors=[15, 10, 5],
            batch_size=4096//process_num,
            num_workers=len(load_core),
            persistent_workers=True, # 是否牺牲内存换取性能
            sampler=train_sampler
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

        best_val_acc = final_test_acc = 0.0
        test_accs = []
        trace_name = "DDP_profile/trace_p{}_l{}_r{}.json".format(process_num, load_core_num, rank)
        for epoch in range(1):
            train_sampler.set_epoch(epoch)
            start_time = time.time()
            model.train()
            if rank == 0:
                pbar = tqdm(total=split_idx['train'].size(0)//process_num)
                pbar.set_description(f'Rank {rank} Epoch {epoch:02d}')

            total_loss = total_correct = total_cnt =  0
            with train_loader.enable_cpu_affinity(loader_cores = load_core):
                print("[LOADER] rank {}: loading data using core {} ".format(rank,load_core))
                for batch in train_loader:
                    optimizer.zero_grad()
                    out = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
                    y = batch.y[:batch.batch_size].squeeze()
                    loss = F.cross_entropy(out, y)
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss)
                    total_correct += int(out.argmax(dim=-1).eq(y).sum())
                    total_cnt += batch.batch_size
                    if rank == 0:
                        pbar.update(batch.batch_size)
            if rank == 0:
                pbar.close()

            end_time = time.time()
            with open('./DDP_profile/DDP_time.txt', 'a') as f:
                f.write(f'Rank {rank} Epoch {epoch} Train_time: {end_time - start_time}\n')

            loss = total_loss / len(train_loader)
            approx_acc = total_correct / total_cnt
            print(f'Rank {rank}|Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {approx_acc:.4f}')
            
            # if epoch % 2 == 0:
            # train_acc, val_acc, test_acc = test()
            # print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
            #     f'Test: {test_acc:.4f}')

            # if val_acc > best_val_acc:
            #     best_val_acc = val_acc
            #     final_test_acc = test_acc
            #     test_accs.append(final_test_acc)
    prof.export_chrome_trace(trace_name)
    trace_list.append(trace_name)
    # test_acc = torch.tensor(test_accs)
    # print(test_accs, test_acc)
    # print('============================')
    # print(f'Final Test: {test_acc.mean():.4f} ± {test_acc.std():.4f}')
    



@torch.no_grad()
def test():
    model.eval()

    out = model.inference(data.x)

    y_true = data.y.cpu()
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    val_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],

        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, val_acc, test_acc

model = SAGE(dataset.num_features, 256, dataset.num_classes, num_layers=3)
model = model.to(device)

# model training
master_addr = '127.0.0.1'
master_port = '29500'

processes = []
try:
    mp.set_start_method('fork')
    print("set start method to fork")
except RuntimeError:
    pass
        
for rank in range(process_num):
    p = mp.Process(target=run, args=(rank, times, model, process_num))
    p.start()
    processes.append(p)
for p in processes:
    p.join()

print("program finished.")






