from typing import Union
from typing import List
from graphConvPool3DPnet import ShrinkingLayer
from stackGraphConvPool3DPnet import ShrinkingLayerStack
import torch
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, shrinkingLayers: Union[List[ShrinkingLayer], List[ShrinkingLayerStack]], mlp: nn.Module):
        """

        :param shrinkingLayers: a list of either shrinking layers or shrinking layer stacks
        :param mlp: mlp: R^(C+I) -> R where C=dimensionality of a point cloud's encoding and I=size of the one hot encoding which depends on the number of classes
        """
        super().__init__()
        self.neuralnet = nn.Sequential(*shrinkingLayers)
        self.mlp = mlp
        self.isStack = isinstance(shrinkingLayers[0], ShrinkingLayerStack)

    def forward(self, feature_matrix_batch: torch.Tensor, one_hot_encoding_batch: torch.Tensor):
        if self.isStack:
            # feature_matrix_batch size = (N,I,D) where N=batch number, I=members, D=dimensionality
            # one_hot_encoding_batch size = (N,E) where N=batch number, E=classes
            feature_matrix_batch = feature_matrix_batch.unsqueeze(0)
            # feature_matrix_batch size = (1,N,I,D) where N=batch number, I=members, D=member dimensionality
            encoding_batch = self.neuralnet(feature_matrix_batch)
            # encoding_batch size = (S,N,1,D') where S= stack size, N=batch number, D'=member dimensionality
            encoding_batch = torch.mean(encoding_batch, dim=0)
            # encoding_batch size = (N,1,D') where N=batch number, D'=member dimensionality
            encoding_batch = encoding_batch.squeeze()
            # encoding_batch size = (N,D') where N=batch number, D'=member dimensionality
            input_batch = torch.cat((encoding_batch, one_hot_encoding_batch), dim=1)
            # input_batch size = (N,L) where N=batch number, L=D'+E
            output = self.mlp(input_batch)
            # output size = (N)
            return output

        else:
            # feature_matrix_batch size = (N,I,D) where N=batch number, I=members, D=dimensionality
            # one_hot_encoding_batch size = (N,E) where N=batch number, E=classes
            encoding_batch = self.neuralnet(feature_matrix_batch)
            # encoding_batch size = (N,1,D') where N=batch number, D'=dimensionality
            encoding_batch = encoding_batch.squeeze()
            # encoding_batch size = (N,D') where N=batch number, D'=member dimensionality
            input_batch = torch.cat((encoding_batch, one_hot_encoding_batch), dim=1)
            # input_batch size = (N,L) where N=batch number, L=D'+E
            output = self.mlp(input_batch)
            # output size = (N)
            return output




