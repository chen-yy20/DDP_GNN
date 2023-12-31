import psutil
import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp


class ResourceManager:

    def __init__(self, args, is_cpu, threshold_1=3e5, threshold_2=0.2):
        self.args = args
        self.is_cpu = is_cpu
        self.threshold_1 = threshold_1
        self.threshold_2 = threshold_2

        self.time_sample = []  # μs
        self.time_compute = []  # μs
        self.time_cpu = []  # s
        self.time_gpu = []  # s
        self.num_sample_cores = 4 // args.cpu_process
        self.cpu_gpu_ratio = args.cpu_gpu_ratio

    def reset(self):
        self.time_sample = []
        self.time_compute = []
        self.time_cpu = []
        self.time_gpu = []
        self.num_sample_cores = 4 // self.args.cpu_process
        self.cpu_gpu_ratio = self.args.cpu_gpu_ratio

    def config(self):
        load_core, comp_core = self.assign_cores()
        return {
            'load_core': load_core,
            'comp_core': comp_core,
            'cpu_gpu_ratio': self.cpu_gpu_ratio,
        }

    def update(self, profile):
        prof, cpu_runtime, gpu_runtime = profile
        time_sample = time_compute = 0
        if self.is_cpu:
            event_list = prof.key_averages()
            for event in event_list:
                if 'DataLoaderIter.__next__' in event.key:
                    time_sample = event.cpu_time_total
                if 'DistributedDataParallel.forward' in event.key:
                    time_compute = event.cpu_time_total
            assert time_sample != 0 and time_compute != 0

        runtime = torch.tensor([time_sample, time_compute, cpu_runtime, gpu_runtime], dtype=torch.float32)
        dist.all_reduce(runtime, op=ReduceOp.SUM)
        runtime = runtime.tolist()
        self.time_sample.append(runtime[0] / self.args.cpu_process)
        self.time_compute.append(runtime[1] / self.args.cpu_process)
        self.time_cpu.append(runtime[2] / self.args.cpu_process)
        # self.time_gpu.append(runtime[3] / self.args.gpu_process)

        # update self.num_sample_cores & self.cpu_gpu_ratio
        time_sample, time_compute, time_cpu = self.estimate_time()
        if abs(time_sample - time_compute) > self.threshold_1:
            if time_sample > time_compute:
                self.num_sample_cores = min(8, self.num_sample_cores * 2)
            else:
                self.num_sample_cores = max(2, self.num_sample_cores // 2)
        else:
            self.cpu_gpu_ratio = 1

    def estimate_time(self):
        time_sample = sum(self.time_sample) / len(self.time_sample)
        time_compute = sum(self.time_compute) / len(self.time_compute)
        time_cpu = sum(self.time_cpu) / len(self.time_cpu)
        # time_gpu = sum(self.time_gpu) / len(self.time_gpu)
        return time_sample, time_compute, time_cpu

    def assign_cores(self):
        num_cpu_proc = self.args.cpu_process
        num_sample_cores = self.num_sample_cores
        n = psutil.cpu_count(logical=False)
        rank = dist.get_rank()
        load_core, comp_core = [], []

        load_core = list(range(n//num_cpu_proc*rank,n//num_cpu_proc*rank+num_sample_cores))
        comp_core = list(range(n//num_cpu_proc*rank+num_sample_cores,n//num_cpu_proc*(rank+1)))

        return load_core, comp_core
