from torch._C import FloatStorageBase
import torch.utils.data
from torchvision.datasets import cifar
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import cifar

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])


noise = True

if noise:
    noise = np.loadtxt("data/cifar100_pair_0.3.csv")
    def target_transform(index):
        return int(noise[index][1])
else:
    target_transform=None


train_loader = torch.utils.data.DataLoader(
                cifar.CIFAR100(root='./data', train=True, 
                                transform=transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32, 4),
                                        transforms.ToTensor(),
                                        normalize,]), 
                                target_transform=target_transform,
                                download=True),
                                batch_size=128, shuffle=True,
                                num_workers=4, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    cifar.CIFAR100(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]),
    target_transform=None),
    batch_size=128, shuffle=False,
    num_workers=4, pin_memory=True)