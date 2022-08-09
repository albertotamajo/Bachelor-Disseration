import torch.nn as nn
from stackGraphConvPool3DPnet import ShrinkingLayerStack
from discriminator import Discriminator

# Hyper-parameters
sample_points = 32  # number of points in the structure point cloud
out_points_decay = 5  # the decay of the points in the point cloud as it goes through the ShrinkingLayerStacks
n_classes = 10  # number of classes in the dataset


class MLP2LayersTanH(nn.Module):
    def __init__(self, in_feature1, out_feature1, out_feature2, out_feature3, out_feature4, out_feature5, out_feature6):
        super().__init__()
        lin1 = nn.Linear(in_feature1, out_feature1)
        lin2 = nn.Linear(out_feature1, out_feature2)
        lin3 = nn.Linear(out_feature2, out_feature3)
        lin4 = nn.Linear(out_feature3, out_feature4)
        lin5 = nn.Linear(out_feature4, out_feature5)
        lin6 = nn.Linear(out_feature5, out_feature6)
        nn.init.xavier_uniform_(lin1.weight)
        nn.init.xavier_uniform_(lin2.weight)
        nn.init.xavier_uniform_(lin3.weight)
        nn.init.xavier_uniform_(lin4.weight)
        nn.init.xavier_uniform_(lin5.weight)
        nn.init.xavier_uniform_(lin6.weight)
        self.neuralNet = nn.Sequential(
            lin1,
            nn.ReLU(),
            lin2,
            nn.ReLU(),
            lin3,
            nn.Tanh(),
            lin4,
            nn.ReLU(),
            lin5,
            nn.ReLU(),
            lin6,
            nn.Tanh()
        )

    def forward(self, X):
        return self.neuralNet(X)


# MLP classifier
class MLPClassifer(nn.Module):
    def __init__(self, in_feature: int):
        super().__init__()
        lin1 = nn.Linear(in_feature, in_feature * 2)
        lin2 = nn.Linear(in_feature * 2, in_feature * 2 + 10)
        lin3 = nn.Linear(in_feature * 2 + 10, in_feature * 2 + 20)
        lin4 = nn.Linear(in_feature * 2 + 20, in_feature * 2 + 10)
        lin5 = nn.Linear(in_feature * 2 + 10, 1)
        nn.init.xavier_uniform_(lin1.weight)
        nn.init.xavier_uniform_(lin2.weight)
        nn.init.xavier_uniform_(lin3.weight)
        nn.init.xavier_uniform_(lin4.weight)
        nn.init.xavier_uniform_(lin5.weight)
        self.main = nn.Sequential(
            lin1,
            nn.ReLU(),
            lin2,
            nn.ReLU(),
            lin3,
            nn.ReLU(),
            lin4,
            nn.ReLU(),
            lin5
        )

    def forward(self, feature_matrix_batch):
        output = self.main(feature_matrix_batch.squeeze())
        return output


# Shrinking Layer 1
input_feature = 3
out_points = sample_points // out_points_decay
input_stack = 1
stack_fork = 2
n_init = 1
C = input_feature
P = 27
out_feature = C + P
mlp = MLP2LayersTanH(C, C + 10, C + 15, C + 20, C+15, C + 10, C)
sigma = nn.ReLU()
F = MLP2LayersTanH(C, C + 10, C + 15, C + 20, C+15, C + 10, C * (C + P))
W = MLP2LayersTanH(C, C + 10, C + 15, C + 20, C+15, C + 10, C * (C + P))
M = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, 1)
B = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp1 = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp2 = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
input_feature = out_feature
shrinkingLayer1 = ShrinkingLayerStack(input_stack, stack_fork, mlp, 1.0, out_points, "k-means++", n_init, sigma, F, W, M, B, C, P, mlp1, mlp2)

# Shrinking Layer 2
out_points = 1
n_init = 1
input_stack = input_stack * stack_fork
stack_fork = 1
C = input_feature
P = 30
out_feature = C + P
mlp = MLP2LayersTanH(C, C + 10, C + 15, C + 20, C+15, C + 10, C)
sigma = nn.ReLU()
F = MLP2LayersTanH(C, C + 10, C + 15, C + 20, C+15, C + 10, C * (C + P))
W = MLP2LayersTanH(C, C + 10, C + 15, C + 20, C+15, C + 10, C * (C + P))
M = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, 1)
B = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp1 = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp2 = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
input_feature = out_feature
shrinkingLayer2 = ShrinkingLayerStack(input_stack, stack_fork, mlp, 1.0, out_points, "k-means++", n_init, sigma, F, W, M, B, C, P, mlp1, mlp2)

shrinkingLayers = [shrinkingLayer1, shrinkingLayer2]
mlp = MLPClassifer(input_feature + n_classes)

struct_discriminator = Discriminator(shrinkingLayers, mlp)


