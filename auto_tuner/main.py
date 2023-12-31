import argparse
import math
import os
from contextlib import nullcontext

import dgl
import dgl.nn as dglnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
from dgl.dataloading.dataloader import _divide_by_worker, _TensorizedDatasetIter
from dgl.multiprocessing import call_once_and_share

from tqdm import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler, ShaDowKHopSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset
from ogb.lsc import MAG240MDataset, MAG240MEvaluator
from torch.profiler import profile, record_function, ProfilerActivity
from torchmetrics.classification import MulticlassAccuracy
import time
import torch.distributed as dist
import torch.multiprocessing as mp
import dgl.multiprocessing as dmp
from torch.nn.parallel import DistributedDataParallel
import psutil

from load_mag_to_shm import fetch_datas_from_shm
from manager import ResourceManager
# from utils import merge_trace_files

TRACE_NAME = 'mixture_product_{}.json'
OUTPUT_TRACE_NAME = "combine.json"


class GNN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers=3, model_name='sage'):
        super().__init__()
        self.layers = nn.ModuleList()

        # GraphSAGE-mean
        if model_name.lower() == 'sage':
            self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
            for i in range(num_layers-2):
                self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
            self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
        # GCN
        elif model_name.lower() == 'gcn':
            kwargs = {'norm': 'both', 'weight': True, 'bias': True, 'allow_zero_in_degree': True}
            self.layers.append(dglnn.GraphConv(in_size, hid_size, **kwargs))
            for i in range(num_layers - 2):
                self.layers.append(dglnn.GraphConv(hid_size, hid_size, **kwargs))
            self.layers.append(dglnn.GraphConv(hid_size, out_size, **kwargs))
        else:
            raise NotImplementedError

        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        if hasattr(blocks, '__len__'):
            for l, (layer, block) in enumerate(zip(self.layers, blocks)):
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
        else:
            for l, layer in enumerate(self.layers):
                h = layer(blocks, h)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
        return h


class UnevenDDPTensorizedDataset(torch.utils.data.IterableDataset):
    """Custom Dataset wrapper that returns a minibatch as tensors or dicts of tensors.
    When the dataset is on the GPU, this significantly reduces the overhead.

    This class additionally saves the index tensor in shared memory and therefore
    avoids duplicating the same index tensor during shuffling.
    """

    def __init__(self, indices, total_batch_size, sub_batch_sizes, drop_last, shuffle):
        self.rank = dist.get_rank()
        self.seed = 0
        self.epoch = 0
        self._mapping_keys = None
        self.drop_last = drop_last
        self._shuffle = shuffle

        # batch size
        self.prefix_sum_batch_size = sum(sub_batch_sizes[:self.rank])
        self.batch_size = sub_batch_sizes[self.rank]

        len_indices = len(indices)
        if self.drop_last and len_indices % total_batch_size != 0:
            self.num_batches = math.ceil((len_indices - total_batch_size) / total_batch_size)
        else:
            self.num_batches = math.ceil(len_indices / total_batch_size)
        self.total_size = self.num_batches * total_batch_size
        # If drop_last is False, we create a shared memory array larger than the number
        # of indices since we will need to pad it after shuffling to make it evenly
        # divisible before every epoch.  If drop_last is True, we create an array
        # with the same size as the indices so we can trim it later.
        self.shared_mem_size = self.total_size if not self.drop_last else len_indices
        self.num_indices = len_indices

        self._id_tensor = indices
        # self._device = self._id_tensor.device
        self.device = self._id_tensor.device

        self._indices = call_once_and_share(
            self._create_shared_indices, (self.shared_mem_size,), torch.int64)

    def update_batch_size(self, sub_batch_sizes):
        self.prefix_sum_batch_size = sum(sub_batch_sizes[:self.rank])
        self.batch_size = sub_batch_sizes[self.rank]

    def _create_shared_indices(self):
        indices = torch.empty(self.shared_mem_size, dtype=torch.int64)
        num_ids = self._id_tensor.shape[0]
        torch.arange(num_ids, out=indices[:num_ids])
        torch.arange(self.shared_mem_size - num_ids, out=indices[num_ids:])
        return indices

    def shuffle(self):
        """Shuffles the dataset."""
        # Only rank 0 does the actual shuffling.  The other ranks wait for it.
        if self.rank == 0:
            np.random.shuffle(self._indices[:self.num_indices].numpy())
            if not self.drop_last:
                # pad extra
                self._indices[self.num_indices:] = \
                    self._indices[:self.total_size - self.num_indices]
        dist.barrier()

    def __iter__(self):
        start = self.prefix_sum_batch_size * self.num_batches
        end = start + self.batch_size * self.num_batches
        indices = _divide_by_worker(self._indices[start:end], self.batch_size, self.drop_last)
        id_tensor = self._id_tensor[indices]
        return _TensorizedDatasetIter(
            id_tensor, self.batch_size, self.drop_last, self._mapping_keys, self._shuffle)

    def __len__(self):
        return self.total_size


def is_cpu_proc(num_cpu_proc, rank=None):
    if rank is None:
        rank = dist.get_rank()
    return rank < num_cpu_proc


def get_subbatch_size(args, rank=None, cpu_gpu_ratio=None) -> int:
    if rank is None:
        rank = dist.get_rank()
    if cpu_gpu_ratio is None:
        cpu_gpu_ratio = args.cpu_gpu_ratio
    world_size = dist.get_world_size()
    cpu_batch_size = int(args.batch_size * cpu_gpu_ratio)
    if is_cpu_proc(args.cpu_process, rank):
        return cpu_batch_size // args.cpu_process + \
            (cpu_batch_size % args.cpu_process if rank == args.cpu_process - 1 else 0)
    else:
        return (args.batch_size - cpu_batch_size) // args.gpu_process + \
            ((args.batch_size - cpu_batch_size) % args.gpu_process if rank == world_size - 1 else 0)


def device_mapping(num_cpu_proc):
    assert not is_cpu_proc(num_cpu_proc), "For GPU Comp process only"
    return dist.get_rank() - num_cpu_proc


def get_device(args):
    device = torch.device("cpu" if is_cpu_proc(args.cpu_process)
                          else "cuda:{}".format(device_mapping(args.cpu_process)))
    if not is_cpu_proc(args.cpu_process):
        torch.cuda.set_device(device)
    return device


def get_sample_workers(args):
    if is_cpu_proc(args.cpu_process):
        return 4 // args.cpu_process
    else:
        return 0


def _train(loader, model, opt, **kwargs):
    model.train()
    total_loss = 0
    if kwargs['rank'] == 0:
        pbar = tqdm(total=kwargs['train_size'])
        epoch = kwargs['epoch']
        pbar.set_description(f'Epoch {epoch:02d}')

    process = kwargs['process']
    device = torch.device("cpu" if is_cpu_proc(process)
                          else "cuda:{}".format(device_mapping(process)))
    for it, (input_nodes, output_nodes, blocks) in enumerate(loader):
        # if it + 1 == 50: break
        if hasattr(blocks, '__len__'):
            x = blocks[0].srcdata["feat"].to(torch.float32)
            y = blocks[-1].dstdata["label"]
        else:
            x = blocks.srcdata["feat"].to(torch.float32)
            y = blocks.dstdata["label"]
        # y_hat = model(blocks, x)
        if kwargs['device'] == "cpu":  # for papers100M
            y = y.type(torch.LongTensor)
            y_hat = model(blocks, x)
        else:
            y = y.type(torch.LongTensor).to(device)
            y_hat = model(blocks, x).to(device)
        loss = F.cross_entropy(y_hat[:output_nodes.shape[0]],
                               y[:output_nodes.shape[0]])
        opt.zero_grad()
        loss.backward()
        opt.step()

        del input_nodes, output_nodes, blocks
        torch.cuda.empty_cache()

        total_loss += loss.item()  # avoid cuda memory accumulation
        if kwargs['rank'] == 0:
            pbar.update(kwargs['batch_size'])
    if kwargs['rank'] == 0:
        pbar.close()
    return total_loss


def hybrid_train(args, config, func, params):
    # for log only
    rank = params['rank']
    epoch = params['epoch']
    num_batches = params['num_batches']
    loader: DataLoader = params['loader']

    # update cpu_gpu_ratio
    sub_batch_sizes = [get_subbatch_size(args, r, config['cpu_gpu_ratio'])
                       for r in range(dist.get_world_size())]
    loader.indices.update_batch_size(sub_batch_sizes)
    if rank == 0:
        print(f'\nEpoch {epoch}, CPU/GPU workload ratio {config["cpu_gpu_ratio"]:.3f}')
        print('SubBatch sizes:', sub_batch_sizes, '\n')

    # start training
    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True
    ) if True else nullcontext() as prof:
        _tik = time.time()
        cpu_runtime = gpu_runtime = 0
        if is_cpu_proc(args.cpu_process):
            with loader.enable_cpu_affinity(loader_cores=config['load_core'],
                                            compute_cores=config['comp_core']):
                loss = func(**params)
                cpu_runtime = time.time() - _tik
        else:
            loss = func(**params)
            gpu_runtime = time.time() - _tik
        if rank == 0:
            print(f'\nTraining loss: {loss / num_batches:.4f}')
            print(f'Epoch Time: {time.time() - _tik:.3f}s\n')
    if epoch == 0:
        prof.export_chrome_trace(TRACE_NAME.format(rank))
    return prof, cpu_runtime, gpu_runtime


def train(rank, world_size, args, g, data):
    num_classes, train_idx = data
    dist.init_process_group('gloo', rank=rank, world_size=world_size)
    device = get_device(args)
    model = GNN(g.ndata["feat"].size(-1), 128, num_classes, args.layer, model_name=args.model)
    model = model.to(device)
    model = DistributedDataParallel(model)

    # create loader
    drop_last, shuffle = True, True
    sub_batch_sizes = [get_subbatch_size(args, r) for r in range(world_size)]
    train_indices = UnevenDDPTensorizedDataset(
        train_idx.to(device),
        args.batch_size,
        sub_batch_sizes,
        drop_last,
        shuffle
    )
    if args.sampler.lower() == 'neighbor':
        sampler = NeighborSampler(
            [15, 10, 5],
            prefetch_node_feats=["feat"],
            prefetch_labels=["label"],
        )
        assert len(sampler.fanouts) == args.layer
    elif args.sampler.lower() == 'shadow':
        sampler = ShaDowKHopSampler(  # CPU sampling is 2x faster than GPU sampling
            [10, 5],
            output_device=device,  # comment out in CPU sampling version
            prefetch_node_feats=["feat"],
        )
    else:
        raise NotImplementedError
    

    # training loop
    params = {
        # training
        # 'loader': train_loader,
        'model': model,
        'opt': torch.optim.Adam(model.parameters(), lr=1e-3),
        # logging
        'rank': rank,
        'train_size': len(train_indices),
        'batch_size': args.batch_size,
        'num_batches': train_indices.num_batches,
        'device': device,
        'process': args.cpu_process,
        'epoch': 0,
    }

    manager = ResourceManager(args, is_cpu_proc(args.cpu_process))
    
    for epoch in range(1):

        conf = manager.config()
        train_loader = DataLoader(
            g,
            train_indices,
            sampler,
            device=device,
            use_ddp=True,
            use_uva=device.type == 'cuda',
            drop_last=drop_last,
            shuffle=shuffle,
            num_workers=0
        )
        
        params['epoch'] = epoch
        params['loader'] = train_loader

        prof = hybrid_train(args, manager.config(), _train, params)
        manager.update(prof)

        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        type=str,
                        default='/home/jason/DDP_GNN/dataset/')
    parser.add_argument('--dataset',
                        type=str,
                        default='ogbn-products',
                        choices=["ogbn-papers100M", "ogbn-products", "mag240M"])
    parser.add_argument("--cpu_process",
                        type=int,
                        default=2,
                        choices=[0, 1, 2, 4])
    parser.add_argument("--gpu_process",
                        type=int,
                        default=1)
    parser.add_argument("--cpu_gpu_ratio",
                        type=float,
                        default=0.5)
    parser.add_argument("--batch_size",
                        type=int,
                        default=1024 * 4)
    parser.add_argument('--sampler',
                        type=str,
                        default='neighbor',
                        choices=["neighbor", "shadow"])
    parser.add_argument('--model',
                        type=str,
                        default='sage',
                        choices=["sage", "gcn"])
    parser.add_argument('--layer',
                        type=int,
                        default=3)
    arguments = parser.parse_args()

    # Assure Consistency
    if arguments.cpu_gpu_ratio == 0 or arguments.cpu_process == 0:
        arguments.cpu_gpu_ratio = 0
        arguments.cpu_process = 0
    if arguments.cpu_gpu_ratio == 1 or arguments.gpu_process == 0:
        arguments.cpu_gpu_ratio = 1
        arguments.gpu_process = 0
    nprocs = arguments.cpu_process + arguments.gpu_process
    assert nprocs > 0
    print(f'\nUse {arguments.cpu_process} CPU Comp processes and {arguments.gpu_process} GPUs\n'
          f'The batch size is {arguments.batch_size} with {arguments.cpu_gpu_ratio} cpu/gpu workload ratio\n'
          f'Sampler: {arguments.sampler}, Model: {arguments.model}, Layer: {arguments.layer}\n')

    # load and preprocess dataset
    print('Use Dataset:', arguments.dataset)
    tik = time.time()
    if arguments.dataset == 'mag240M':
        dataset = MAG240MDataset(root='../HiPC')
        print('Start Loading Graph Structure')
        (g,), _ = dgl.load_graphs('../HiPC/graph.dgl')
        g = g.formats(["csc"])
        print('Graph Structure Loading Finished!')
        paper_offset = dataset.num_authors + dataset.num_institutions
        dataset.train_idx = torch.from_numpy(dataset.get_idx_split("train")) + paper_offset
        g.ndata["feat"] = fetch_datas_from_shm()
        g.ndata["label"] = torch.cat([torch.empty((paper_offset,), dtype=torch.long),
                                      torch.LongTensor(dataset.paper_label[:])])
        print('Graph Feature/Label Loading Finished!')
    else:
        dataset = AsNodePredDataset(DglNodePropPredDataset(arguments.dataset, arguments.data_path))
        g = dataset[0]

    """
    Note 1: This func avoid creating certain graph formats in each sub-process to save memory
    Note 2: This func will init CUDA. It is not possible to use CUDA in a child process 
            created by fork(), if CUDA has been initialized in the parent process. 
    """
    # g.create_formats_()

    data = (
        dataset.num_classes,
        dataset.train_idx,
    )
    tok = time.time()
    print(f"Data loading finished, Elapsed Time: {time.time() - tik: .1f}s")

    # multi-processes training
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29502'

    # train(0, nprocs, arguments, g, data)
    mp.set_start_method('fork')
    processes = []
    for i in range(nprocs):
        p = dmp.Process(target=train, args=(i, nprocs, arguments, g, data))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    # input_files = [TRACE_NAME.format(i) for i in range(nprocs)]
    # merge_trace_files(input_files, OUTPUT_TRACE_NAME)
    # for i in range(nprocs):
    #     os.remove(TRACE_NAME.format(i))

    print("Program finished")
