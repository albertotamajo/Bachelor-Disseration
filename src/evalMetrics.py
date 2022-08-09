import torch
from pytorch3d.loss import chamfer_distance

def coverage_cd(gen_samples:torch.Tensor, real_samples:torch.Tensor):
    # gen_samples size = (N,I,D) where N=batch, I=points, D=dimension
    # real_samples size = (N,I,D) where N=batch, I=points, D=dimension
    d = {}
    for i in range(len(real_samples)):
        d[i] = 0
    for gen_sample in gen_samples:
        cd = float("inf")
        sim = None
        for i in range(len(real_samples)):
            cd_, _ = chamfer_distance(gen_sample.unsqueeze(0), real_samples[i].unsqueeze(0))
            if cd_ < cd:
                cd = cd_
                sim = i
        d[sim] = d[sim] + 1
    return len([d[i] for i in range(len(real_samples)) if d[i] != 0]) / len(real_samples)


def mmd_cd(gen_samples:torch.Tensor, real_samples:torch.Tensor):
    # gen_samples size = (N,I,D) where N=batch, I=points, D=dimension
    # real_samples size = (N,I,D) where N=batch, I=points, D=dimension
    cds = []
    for real_sample in real_samples:
        cd = float("inf")
        for gen_sample in gen_samples:
            cd_, _ = chamfer_distance(gen_sample.unsqueeze(0), real_sample.unsqueeze(0))
            if cd_ < cd:
                cd = cd_
        cds.append(cd)
    return sum(cds) / len(cds)