import torch

mean = torch.tensor([0.0])
std = torch.tensor([1.0])
normal_dist =torch.distributions.Normal(mean, std)  # 创建一个均值为0，标准差为1的正态分布
print(normal_dist)
sample = normal_dist.sample()
print(sample)