import argparse
import time

import torch
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset

from torch_geometric.loader import NeighborLoader, ShaDowKHopSampler
from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.datasets import Reddit2, Yelp, Flickr
import os
import subprocess
import psutil

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

# GraphSAGE model
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


def run(times, model, model_name):
    device = torch.device('cpu')
    train_idx = split_idx['train'].to(device)

    for run in range(times):  
        model.reset_parameters()
        train(model, train_idx, model_name)

        
def train(model, train_idx, model_name): 
    model.train()
    # set trainer core
    torch.set_num_threads(len(trainer_core))
    pid = os.getpid()
    core_mask = get_mask(trainer_core)
    subprocess.run(["taskset", "-a","-p", str(core_mask), str(pid)])

    if sampler == "neighbor":
        train_loader = NeighborLoader(
            data,
            input_nodes = train_idx,
            num_neighbors=[15, 10, 5],
            batch_size=4096,
            num_workers=len(loader_core),
            persistent_workers=True,
        )
    elif sampler == "shadow":
        train_loader = ShaDowKHopSampler(
            data,
            depth=3,
            num_neighbors=5,
            node_idx=train_idx,
            batch_size=4096,
            num_workers=len(loader_core)
        )



    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
  
    for epoch in range(1):
        start_time = time.time()
        
        total_loss = total_correct = total_cnt =  0
        if sampler == "neighbor":
            with train_loader.enable_cpu_affinity(loader_core):
                for batch in train_loader:
                    optimizer.zero_grad()
                    out = model(batch.x, batch.edge_index)[:batch.batch_size]
                    y = batch.y[:batch.batch_size].squeeze().long()
                    loss = F.cross_entropy(out, y)
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss)   
                    # yelp is multi-label classification
                    if (args.dataset == 'yelp'):
                        # True: larger than 0.5, False: less than 0.5
                        out = out > 0.5
                        total_correct += int(out.eq(y).sum())
                        total_cnt += batch.batch_size * dataset.num_classes
                    else:
                        total_correct += int(out.argmax(dim=-1).eq(y).sum())
                        total_cnt += batch.batch_size
        elif sampler == "shadow":
            for batch in train_loader:
                batchsize = len(batch.y)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index.to(device))[:batchsize]
                y = batch.y[:batchsize].squeeze().long()
                loss = F.cross_entropy(out, y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss)   
                # yelp is multi-label classification
                if (args.dataset == 'yelp'):
                    # True: larger than 0.5, False: less than 0.5
                    out = out > 0.5
                    total_correct += int(out.eq(y).sum())
                    total_cnt += batchsize * dataset.num_classes
                else:
                    total_correct += int(out.argmax(dim=-1).eq(y).sum())
                    total_cnt += batchsize

        end_time = time.time()
        loss = total_loss / len(train_loader)
        approx_acc = total_correct / total_cnt
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {approx_acc:.4f}')
        print("total_time: ", end_time - start_time)   
        with open("PyG/Result/Scal_{}_{}.txt".format(args.dataset, args.model), "a") as text_file:
            text_file.write("\n Core num:" + str(args.trainer) + "  ")
            text_file.write("Training time:" + str(end_time - start_time) + "\n")
           

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_times",
        default= "1",
        help="repeat times to run the experiment",
    )

    parser.add_argument(
        "--dataset",
        default= "ogbn-products",
        help="dataset name",
        choices=["ogbn-products", "flickr", "yelp", "reddit", "ogbn-papers100M"],
    )

    parser.add_argument(
        "--model",
        default= "sage",
        help="model",
        choices=["sage", "gcn"],
    )
    
    parser.add_argument(
        "--sampler",
        default= "neighbor",
        help="sampler",
        choices=["shadow", "neighbor"],
    )

    parser.add_argument(
        "--trainer",
        default= 28,
        help="trainer core",
    )




    args = parser.parse_args()
    args.mode = "cpu"
    # print(f"Training in {args.mode} mode.")
    times = int(args.run_times)
    model_name = args.model
    sampler = args.sampler
    socket_core_num = 32

    trainer_num = int(args.trainer)
    max_core_num = psutil.cpu_count(logical=False) -2
    assert(trainer_num <= max_core_num)
    trainer_core = [i for i in range(0, trainer_num)]
    loader_core = [i for i in range(trainer_num, trainer_num+2)]
    print("loader_core: ", loader_core)
    print("trainer_core: ", trainer_core)
    

    device = torch.device('cpu')

    
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
    elif args.dataset == "yelp":
        dataset = Yelp(root='./PyG/data/yelp')
        split_idx = get_split_idx(dataset[0].train_mask, dataset[0].val_mask, dataset[0].test_mask)
    elif args.dataset == "reddit":
        dataset = Reddit2(root='./PyG/data/reddit')
        split_idx = get_split_idx(dataset[0].train_mask, dataset[0].val_mask, dataset[0].test_mask)

    data = dataset[0].to(device, 'x', 'y')


    model = GNN(dataset.num_features, 128, dataset.num_classes, num_layers=3, model_name=model_name)
    model = model.to(device)
   
    run(times, model, model_name)
    







