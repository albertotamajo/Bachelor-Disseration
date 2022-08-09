import torch.nn as nn
from stackGraphConvPool3DPnet import ShrinkingLayerStack
from graphConvPool3DPnet import ShrinkingLayer
from vaegan import Encoder

# Hyper-parameters
sample_points = 800  # number of points in the initial point cloud
out_points_decay = 5  # the decay of the points in the point cloud as it goes through the ShrinkingLayerStacks
latent_vector_dim = 30  # dimension of the latent vector output by the encoder


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


class MLP2LayersLinear(nn.Module):
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
        )

    def forward(self, X):
        return self.neuralNet(X)


# Shrinking Layer 1
input_feature = 3
out_points = sample_points // out_points_decay
n_init = 1
C = input_feature
P = 10
out_feature = C + P
mlp = MLP2LayersTanH(C, C + 10, C + 15, C + 20, C + 15, C + 10, C)
sigma = nn.ReLU()
F = MLP2LayersTanH(C, C + 10, C + 15, C + 20, C + 15, C + 10, C * (C + P))
W = MLP2LayersTanH(C, C + 10, C + 15, C + 20, C + 15, C + 10, C * (C + P))
M = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, 1)
B = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp1 = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp2 = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
input_feature = out_feature
shrinkingLayer1 = ShrinkingLayer(mlp, 1.0, out_points, "k-means++", n_init, sigma, F, W,
                                      M, B, C, P, mlp1, mlp2)
                                      
# Shrinking Layer 2
out_points = sample_points // out_points_decay
n_init = 1
C = input_feature
P = 17
out_feature = C + P
mlp = MLP2LayersTanH(C, C + 10, C + 15, C + 20, C + 15, C + 10, C)
sigma = nn.ReLU()
F = MLP2LayersTanH(C, C + 10, C + 15, C + 20, C + 15, C + 10, C * (C + P))
W = MLP2LayersTanH(C, C + 10, C + 15, C + 20, C + 15, C + 10, C * (C + P))
M = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, 1)
B = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp1 = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp2 = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
input_feature = out_feature
shrinkingLayer2 = ShrinkingLayer(mlp, 1.0, out_points, "k-means++", n_init, sigma, F, W,
                                      M, B, C, P, mlp1, mlp2)                                      

# Shrinking Layer 3
out_points = 1
n_init = 1
C = input_feature
P = 30
out_feature = C + P
mlp = MLP2LayersTanH(C, C + 10, C + 15, C + 20, C + 15, C + 10, C)
sigma = nn.ReLU()
F = MLP2LayersTanH(C, C + 10, C + 15, C + 20, C + 15, C + 10, C * (C + P))
W = MLP2LayersTanH(C, C + 10, C + 15, C + 20, C + 15, C + 10, C * (C + P))
M = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, 1)
B = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp1 = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp2 = MLP2LayersTanH(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
input_feature = out_feature
shrinkingLayer3 = ShrinkingLayer(mlp, 1.0, out_points, "k-means++", n_init, sigma, F, W,
                                      M, B, C, P, mlp1, mlp2)

shrinkingLayers = [shrinkingLayer1, shrinkingLayer2, shrinkingLayer3]

mlp1 = MLP2LayersLinear(input_feature, input_feature + 10, input_feature + 15, input_feature + 20, input_feature + 15,
                      input_feature + 10, latent_vector_dim)
mlp2 = MLP2LayersLinear(input_feature, input_feature + 10, input_feature + 15, input_feature + 20, input_feature + 15,
                      input_feature + 10, latent_vector_dim)

encoder = Encoder(shrinkingLayers, mlp1, mlp2)
