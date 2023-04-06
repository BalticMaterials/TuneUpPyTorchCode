################################################################## 
#  DataParallel
#  Version: 0.1
#  Code-Source: https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html
# 
# 
# 
# 
##################################################################

import torch
import torch.nn as nn


class DataParallelModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.block1 = nn.Linear(10, 20)

        # wrap block2 in DataParallel
        self.block2 = nn.Linear(20, 20)
        self.block2 = nn.DataParallel(self.block2)

        self.block3 = nn.Linear(20, 20)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

### Part of the model on CPU and part on the GPU

class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        return getattr(self.module, name)

def data_parallel(module, input, device_ids, output_device=None):
    if not device_ids:
        return module(input)

    if output_device is None:
        output_device = device_ids[0]

    replicas = nn.parallel.replicate(module, device_ids)
    inputs = nn.parallel.scatter(input, device_ids)
    replicas = replicas[:len(inputs)]
    outputs = nn.parallel.parallel_apply(replicas, inputs)
    return nn.parallel.gather(outputs, output_device)


### Part of the model on CPU and part on the GPU

device = torch.device("cuda:0")

class DistributedModel(nn.Module):

    def __init__(self):
        super().__init__(
            embedding=nn.Embedding(1000, 10),
            rnn=nn.Linear(10, 10).to(device),
        )

    def forward(self, x):
        # Compute embedding on CPU
        x = self.embedding(x)

        # Transfer to GPU
        x = x.to(device)

        # Compute RNN on GPU
        x = self.rnn(x)
        return x