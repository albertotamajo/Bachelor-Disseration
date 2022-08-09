from typing import Union, List
import torch
import torch.nn as nn
from graphConvPool3DPnet import ShrinkingLayer
from stackGraphConvPool3DPnet import ShrinkingLayerStack
from generator import StructureGenerator
from discriminator import Discriminator
from graphConvPool3DPnet import getDevice


class Encoder(nn.Module):

    def __init__(self, shrinkingLayers: Union[List[ShrinkingLayer], List[ShrinkingLayerStack]], mlp1: nn.Module,
                 mlp2: nn.Module):
        """

        :param shrinkingLayers: a list of either shrinking layers or shrinking layer stacks
        :param mlp1: mlp1: R^C -> R^D maps the output of the shrinking layers to a vector of means
        :param mlp2: mlp2: R^C -> R^D maps the output of the shrinking layers to a vector of standard deviations
        """
        super().__init__()
        self.neuralnet = nn.Sequential(*shrinkingLayers)
        self.mlp1 = mlp1
        self.mlp2 = mlp2
        self.isStack = isinstance(shrinkingLayers[0], ShrinkingLayerStack)

    def reparameterise(self, means_batch: torch.Tensor, log_stds_batch: torch.Tensor):
        # Trick to sample from the appropriate normal distribution using the standard normal distribution
        # so that to make backpropagation possible
        rand = torch.empty(means_batch.shape, device=getDevice(means_batch)).normal_()
        stds_batch = torch.exp(0.5*log_stds_batch)
        return means_batch + stds_batch * rand

    def forward(self, feature_matrix_batch: torch.Tensor):
        if self.isStack:
            # feature_matrix_batch size = (N,I,D) where N=batch number, I=members, D=dimensionality
            feature_matrix_batch = feature_matrix_batch.unsqueeze(0)
            # feature_matrix_batch size = (1,N,I,D) where N=batch number, I=members, D=member dimensionality
            encoding_batch = self.neuralnet(feature_matrix_batch)
            # encoding_batch size = (S,N,1,D') where S= stack size, N=batch number, D'=member dimensionality
            encoding_batch = torch.mean(encoding_batch, dim=0)
            # encoding_batch size = (N,1,D') where N=batch number, D'=member dimensionality
            encoding_batch = encoding_batch.squeeze()
            # encoding_batch size = (N,D') where N=batch number, D'=member dimensionality
            means_batch = self.mlp1(encoding_batch)
            # means_batch size = (N,C) where N=batch number, C=dimensionality
            log_stds_batch = self.mlp2(encoding_batch)
            # log_stds_batch size = (N,C) where N=batch number, C=dimensionality
            latent_vector_batch = self.reparameterise(means_batch, log_stds_batch)
            # latent_vector_batch size = (N,C) where N=batch number, C=dimensionality
            return latent_vector_batch, means_batch, log_stds_batch

        else:
            # feature_matrix_batch size = (N,I,D) where N=batch number, I=members, D=dimensionality
            encoding_batch = self.neuralnet(feature_matrix_batch)
            # encoding_batch size = (N,1,C) where N=batch number, C=dimensionality
            encoding_batch = encoding_batch.squeeze()
            # encoding_batch size = (N,C) where N=batch number, C=dimensionality
            means_batch = self.mlp1(encoding_batch)
            # means_batch size = (N,D') where N=batch number, D'=dimensionality
            log_stds_batch = self.mlp2(encoding_batch)
            # log_stds_batch size = (N,D') where N=batch number, D'=dimensionality
            latent_vector_batch = self.reparameterise(means_batch, log_stds_batch)
            # latent_vector_batch size = (N,D') where N=batch number, D'=dimensionality
            return latent_vector_batch, means_batch, log_stds_batch


class Struct_VAE_GAN(nn.Module):

    def __init__(self, encoder: Encoder, struct_generator: StructureGenerator, struct_discriminator: Discriminator):
        super().__init__()
        self.encoder = encoder
        self.struct_generator = struct_generator
        self.struct_discriminator = struct_discriminator
        self.is_discriminating = True

    def forward(self, feature_matrix_batch: torch.Tensor, one_hot_encoding_batch: torch.Tensor):
        if self.is_discriminating:  # discriminating mode
            # one_hot_encoding_batch size = (N,E) where N=batch number, E=classes
            struct_feature_matrix_batch = feature_matrix_batch
            # struct_feature_matrix_batch size = (N, I', D) where N=batch number, I'=members, D=dimensionality
            return self.struct_discriminator(struct_feature_matrix_batch, one_hot_encoding_batch)

        else:  # generating-discriminating mode
            # feature_matrix_batch size = (N,I,D) where N=batch number, I=members, D=dimensionality
            # one_hot_encoding_batch size = (N,E) where N=batch number, E=classes
            latent_vector_batch, means_batch, log_stds_batch = self.encoder(feature_matrix_batch)
            # latent_vector_batch size = (N,D') where N=batch number, D'=dimensionality
            # means_batch size = (N,D') where N=batch number, D'=dimensionality
            # log_stds_batch size = (N,D') where N=batch number, D'=dimensionality
            struct_feature_matrix_batch = self.struct_generator(latent_vector_batch)
            # feature_matrix_batch size = (N,I',D'') where N=batch number, I'=members, D''=dimensionality
            output = self.struct_discriminator(struct_feature_matrix_batch, one_hot_encoding_batch)
            # output size = (N)
            return output, struct_feature_matrix_batch, means_batch, log_stds_batch

