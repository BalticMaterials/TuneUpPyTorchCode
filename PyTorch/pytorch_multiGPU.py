##############################################################################################################################################
#
#    Multi GPU - DDP 
#    Setting up multi GPU processing in PyTorch
#    https://medium.com/exemplifyml-ai/multi-gpu-training-in-pytorch-ab1a9500377e
#
#
#
#
#
#
#
#
#
##############################################################################################################################################
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

# number of GPUs equal to number of processes
world_size = torch.cuda.device_count()
print(world_size)
#mp.spawn(<selfcontainedmethodforeachproc>, nprocs=world_size, args=(args,))

# GPU Process Assignment:


def train(self, rank, args):
    current_gpu_index = rank
    torch.cuda.set_device(current_gpu_index)

    dist.init_process_group(
        backend='nccl', world_size=args.world_size, 
        rank=current_gpu_index,
        init_method='env://'
    )
    
# Setup Image-Loader
'''Info'''
<basedir>/testset/<categoryname>/<listofimages>
<basedir>/valset/<categoryname>/<listofimages>
<basedir>/trainset/<categoryname>/<listofimages>

# Setup Data-Set
'''Info'''
from torchvision.datasets import ImageFolder
train_dataset = ImageFolder(root=os.path.join(<basedir>, "trainset"), transform=train_transform)


# Setup Distributed-Sampler
'''Info'''
from torch.utils.data import DistributedSampler
dist_train_samples = DistributedSampler(dataset=train_dataset, num_replicas =4, rank=rank, seed=17)

#Setup Data-Loader
'''Info'''
from torch.utils.data import DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=self.BATCH_SIZE,
    num_workers=4,
    sampler=dist_train_samples,
    pin_memory=True,
)

# Multi Process Model Initialization:

from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import models as models
model = models.resnet34(pretrained=True)
loss_fn = nn.CrossEntropyLoss()
model.cuda(current_gpu_index)
model = DDP(model)

loss_fn.cuda(current_gpu_index)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7)


















