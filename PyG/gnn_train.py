# Reaches around 0.7930 test accuracy.

import argparse
import time

import torch
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset

from torch_geometric.loader import NeighborLoader, ShaDowKHopSampler, GraphSAINTNodeSampler
from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.datasets import Reddit2, Flickr

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
import dgl.multiprocessing as dmp
from torch.nn.parallel import DistributedDataParallel
import psutil
import os
import subprocess
import random
import csv

# replace node_index with node_mask
def get_split_idx(train_mask, val_mask, test_mask):
    train_idx = torch.where(train_mask)[0]
    val_idx = torch.where(val_mask)[0]
    test_idx = torch.where(test_mask)[0]
    return {'train': train_idx, 'valid': val_idx, 'test': test_idx}

def get_mask(comp_core):
    mask = 0
    for core in comp_core:
        mask += 2 ** core
    return hex(mask)

def is_cpu_proc(num_cpu_proc, rank=None):
    if rank is None:
        rank = dist.get_rank()
    return rank < num_cpu_proc

def assign_cores(num_cpu_proc,n_samp,n_train):
    assert is_cpu_proc(num_cpu_proc), "For CPU Comp process only"
    rank = dist.get_rank()
    load_core, comp_core = [], []
    n = psutil.cpu_count(logical=False)
    size = num_cpu_proc
    num_of_samplers = n_samp
    load_core = list(range(n//size*rank,n//size*rank+num_of_samplers))
    comp_core = list(range(n//size*rank+num_of_samplers,n//size*rank+num_of_samplers+n_train))

    return load_core, comp_core
# GraphSAGE & GCN  model
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, model_name):
        super().__init__()

        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        if model_name == "sage":
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels))
        elif model_name == "gcn":
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels))
        else:
            raise NotImplementedError

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




def run(rank, times, process_num, model_name):
    try:
        dist.init_process_group('gloo', rank=rank, world_size=process_num)  
    except:
        print("Port setting error.")
        return
    device = torch.device('cpu' if is_cpu_proc(process_num, rank) 
                            else 'cuda')
    if not is_cpu_proc(process_num):
        torch.cuda.set_device(device)
    train_idx = split_idx['train'].to(device)

    for run in range(times):
        model = GNN(dataset.num_features, 128, dataset.num_classes, num_layers=3, model_name=model_name).to(device)
        model = DistributedDataParallel(model)
        train(model, rank, train_idx)
    dist.barrier()

        
def train(model, rank, train_idx):
    model.train()
    load_core, comp_core = assign_cores(process_num, load_core_num, compute_core_num)
    # set compute cores 【taskset】
    torch.set_num_threads(len(comp_core))
    pid = os.getpid()
    # print("[TASKSET] rank {}, pid: {}".format(rank, pid))
    core_mask = get_mask(comp_core)
    subprocess.run(["taskset", "-a","-p", str(core_mask), str(pid)])
    print("[TASKSET] rank {}, using compute core: {}".format(rank, comp_core))

    if sampler == "neighbor":
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
            persistent_workers=True, # using memory cache to speed up
            sampler=train_sampler
        )
    elif sampler == "shadow":
        train_loader = ShaDowKHopSampler(
            data,
            node_idx=split_idx['train'],
            depth=2,
            num_neighbors=5,
            batch_size=4096//process_num,
            num_workers=len(load_core),
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    for epoch in range(1):
        total_loss = total_correct = total_cnt =  0
        if sampler == "neighbor":
            train_sampler.set_epoch(epoch)
            # set loader cores
            with train_loader.enable_cpu_affinity(loader_cores = load_core):
                # print("[LOADER] rank {}: loading data using core {} ".format(rank,load_core))
                for batch in train_loader:
                    if batch_num != 0 and total_cnt >= batch_num:
                        break
                    optimizer.zero_grad()
                    out = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
                    y = batch.y[:batch.batch_size].squeeze().long()
                    loss = F.cross_entropy(out, y)
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss)   
                    total_correct += int(out.argmax(dim=-1).eq(y).sum())
                    total_cnt += 1
                    # if rank == 0:
                    #     print("{}/{}".format(total_cnt, batch_num))

        elif sampler == "shadow":
            for batch in train_loader:
                if batch_num != 0 and total_cnt >= batch_num:
                    break
                batchsize = len(batch.y)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index.to(device))[:batchsize]
                y = batch.y[:batchsize].squeeze().long()
                loss = F.cross_entropy(out, y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss)   
                total_correct += int(out.argmax(dim=-1).eq(y).sum())
                total_cnt += 1
                # if rank == 0:
                #     print("{}/{}".format(total_cnt, batch_num))

        loss = total_loss / len(train_loader)
        # approx_acc = total_correct / total_cnt
        # print(f'Rank {rank}|Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {approx_acc:.4f}')
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cpu_process",
        default= "2",
    )

    parser.add_argument(
        "--n_sampler",
        default= "2",
        help="loader core number"
    )

    parser.add_argument(
        "--n_trainer",
        default= "8",
        help="trainer core number"
    )

    parser.add_argument(
        "--run_times",
        default= "1",
        help="repeat times to run the experiment",
    )

    parser.add_argument(
        "--dataset",
        default= "ogbn-products",
        help="dataset name",
        choices=["ogbn-products", "ogbn-papers100M", "flickr",  "reddit"],
    )

    parser.add_argument(
        "--model",
        type = str,
        default= "sage",
        choices=["sage", "gcn"],
    )

    parser.add_argument(
        "--sampler",
        type = str,
        default= "neighbor",
        choices = ["neighbor", "shadow"]
    )

    parser.add_argument(
        "--batch_num",
        type = str,
        help = "Number of batches for training ",
        default= "0",
    )

    parser.add_argument(
        "--record",
        action = "store_true",
    )



    args = parser.parse_args()
    args.mode = "cpu"
    # print(f"Training in {args.mode} mode.")
    times = int(args.run_times)
    batch_num = int(args.batch_num)
    record = args.record

    process_num = int(args.cpu_process)
    load_core_num = int(args.n_sampler)
    compute_core_num = int(args.n_trainer)
    model = args.model
    sampler = args.sampler
    max_core_num = psutil.cpu_count(logical=False)
    assert(process_num * (load_core_num + compute_core_num) <= max_core_num)

    device = torch.device('cpu')
    print("device: ", device, "process_num: ", process_num, "load_core_num: ", load_core_num)
    # device = torch.device('cpu')
    if args.dataset == "ogbn-products":
        dataset = PygNodePropPredDataset(name = 'ogbn-products', root = './PyG/data/ogbn')
        split_idx = dataset.get_idx_split()
        # evaluator = Evaluator(name='ogbn-products')
    elif args.dataset == "ogbn-papers100M":
        dataset = PygNodePropPredDataset(name = 'ogbn-papers100M', root = './PyG/data/ogbn-papers100M')
        split_idx = dataset.get_idx_split()
    elif args.dataset == "flickr":
        dataset = Flickr(root='./PyG/data/flickr')
        split_idx = get_split_idx(dataset[0].train_mask, dataset[0].val_mask, dataset[0].test_mask) 
    elif args.dataset == "reddit":
        dataset = Reddit2(root='./PyG/data/reddit')
        split_idx = get_split_idx(dataset[0].train_mask, dataset[0].val_mask, dataset[0].test_mask)

    data = dataset[0].to(device, 'x', 'y')

    # model = SAGE(dataset.num_features, 256, dataset.num_classes, num_layers=3)
    # model = model.to(device)
    # port_list = np.arange(29510,29510+30)
    # port = np.random.choice(port_list)
    port = random.randint(29500,30000)
    print(port)
    # multi-processes training
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)

    processes = []
    try:
        mp.set_start_method('fork', force=True)
        print("set start method to fork")
    except RuntimeError:
        pass
   
    tik = time.time()
    for rank in range(process_num):
        p = dmp.Process(target=run, args=(rank, times, process_num, model))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    t = time.time() - tik
    print('total_time:', t)
    if record:
        meta_data = []
        meta_data.append(t)
        meta_data.append(args.cpu_process)
        meta_data.append(args.n_sampler)
        meta_data.append(args.n_trainer)

        with open('PyG/Result/Grid_{}_{}_{}.csv'.format(args.dataset,args.model,args.batch_num), 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(meta_data)
    print("program finished.")
