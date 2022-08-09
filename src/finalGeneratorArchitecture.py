import torch.nn as nn
from generator import ExpandingLayer, FinalGenerator
import structureGeneratorArchitecture as structGenArchi


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


# Expanding Layer Struct 1
input_feature = 3
k = 8
branching = 10
n_init = 1
C = input_feature
P = 0
mlp = MLP2LayersTanH(C, C + 10, C + 15, C + 20, C+15, C + 10, C)
sigma = nn.ReLU()
F = MLP2LayersTanH(C, C + 10, C + 15, C + 20, C+15, C + 10, C * (C + P))
W = MLP2LayersTanH(C, C + 10, C + 15, C + 20, C+15, C + 10, C * (C + P))
M = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, 1)
B = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp1 = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp2 = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp3 = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, (C + P)*branching)
expandingLayer1Struct = ExpandingLayer(mlp, 1.0, k, "k-means++", n_init, sigma, F, W, M, B, C, P, mlp1, mlp2, branching, mlp3)

# Expanding Layer Struct 2
k = 16
branching = 30
n_init = 1
C = input_feature
P = 0
mlp = MLP2LayersTanH(C, C + 10, C + 15, C + 20, C+15, C + 10, C)
sigma = nn.ReLU()
F = MLP2LayersTanH(C, C + 10, C + 15, C + 20, C+15, C + 10, C * (C + P))
W = MLP2LayersTanH(C, C + 10, C + 15, C + 20, C+15, C + 10, C * (C + P))
M = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, 1)
B = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp1 = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp2 = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp3 = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, (C + P)*branching)
expandingLayer2Struct = ExpandingLayer(mlp, 1.0, k, "k-means++", n_init, sigma, F, W, M, B, C, P, mlp1, mlp2, branching, mlp3)

# Expanding Layer Struct 3
k = 16
branching = 13
n_init = 1
C = input_feature
P = 0
mlp = MLP2LayersTanH(C, C + 10, C + 15, C + 20, C+15, C + 10, C)
sigma = nn.ReLU()
F = MLP2LayersTanH(C, C + 10, C + 15, C + 20, C+15, C + 10, C * (C + P))
W = MLP2LayersTanH(C, C + 10, C + 15, C + 20, C+15, C + 10, C * (C + P))
M = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, 1)
B = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp1 = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp2 = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp3 = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, (C + P)*branching)
expandingLayer3Struct = ExpandingLayer(mlp, 1.0, k, "k-means++", n_init, sigma, F, W, M, B, C, P, mlp1, mlp2, branching, mlp3)


expandingLayersStruct = [expandingLayer1Struct, expandingLayer2Struct, expandingLayer3Struct]


# Expanding Layer Latent 1
input_feature = 3
k = 4
branching = 28
n_init = 1
C = input_feature
P = 0
mlp = MLP2LayersTanH(C, C + 10, C + 15, C + 20, C+15, C + 10, C)
sigma = nn.ReLU()
F = MLP2LayersTanH(C, C + 10, C + 15, C + 20, C+15, C + 10, C * (C + P))
W = MLP2LayersTanH(C, C + 10, C + 15, C + 20, C+15, C + 10, C * (C + P))
M = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, 1)
B = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp1 = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp2 = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp3 = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, (C + P)*branching)
expandingLayer1Latent = ExpandingLayer(mlp, 1.0, k, "k-means++", n_init, sigma, F, W, M, B, C, P, mlp1, mlp2, branching, mlp3)

# Expanding Layer Latent 2
k = 16
branching = 30
n_init = 1
C = input_feature
P = 0
mlp = MLP2LayersTanH(C, C + 10, C + 15, C + 20, C+15, C + 10, C)
sigma = nn.ReLU()
F = MLP2LayersTanH(C, C + 10, C + 15, C + 20, C+15, C + 10, C * (C + P))
W = MLP2LayersTanH(C, C + 10, C + 15, C + 20, C+15, C + 10, C * (C + P))
M = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, 1)
B = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp1 = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp2 = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp3 = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, (C + P)*branching)
expandingLayer2Latent = ExpandingLayer(mlp, 1.0, k, "k-means++", n_init, sigma, F, W, M, B, C, P, mlp1, mlp2, branching, mlp3)

# Expanding Layer Latent 3
k = 18
branching = 11
n_init = 1
C = input_feature
P = 0
mlp = MLP2LayersTanH(C, C + 10, C + 15, C + 20, C+15, C + 10, C)
sigma = nn.ReLU()
F = MLP2LayersTanH(C, C + 10, C + 15, C + 20, C+15, C + 10, C * (C + P))
W = MLP2LayersTanH(C, C + 10, C + 15, C + 20, C+15, C + 10, C * (C + P))
M = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, 1)
B = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp1 = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp2 = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp3 = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, (C + P)*branching)
expandingLayer3Latent = ExpandingLayer(mlp, 1.0, k, "k-means++", n_init, sigma, F, W, M, B, C, P, mlp1, mlp2, branching, mlp3)


expandingLayersLatent = [expandingLayer1Latent, expandingLayer2Latent, expandingLayer3Latent]
mlp = structGenArchi.mlp_latent

final_generator = FinalGenerator(structGenArchi.i, structGenArchi.d, mlp, expandingLayersLatent, expandingLayersStruct)
