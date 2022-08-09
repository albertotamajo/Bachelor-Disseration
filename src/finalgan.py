import torch
import torch.nn as nn
from generator import StructureGenerator, FinalGenerator
from discriminator import Discriminator


class FinalGAN(nn.Module):

    def __init__(self, struct_generator: StructureGenerator, final_generator: FinalGenerator, final_discriminator: Discriminator):
        super().__init__()
        self.struct_generator = struct_generator
        self.final_generator = final_generator
        self.final_discriminator = final_discriminator
        self.is_discriminating = True

    def forward(self, feature_matrix_batch: torch.Tensor, one_hot_encoding_batch: torch.Tensor):
        if self.is_discriminating:  # Discriminating mode
            # feature_matrix_batch size = (N,I,D) where N=batch number, I=members, D=dimensionality
            return self.final_discriminator(feature_matrix_batch, one_hot_encoding_batch)

        else:  # Generating-discriminating mode
            with torch.no_grad():  # struct generator operations do not have to require gradient computation
                latent_vector_batch = feature_matrix_batch
                # latent_vector_batch size = (N,D) where N=batch number, D=dimensionality
                # one_hot_encoding_batch size = (N,E) where N=batch number, E=classes
                struct_feature_matrix_batch = self.struct_generator(latent_vector_batch)
                # struct_feature_matrix_batch size = (N,I,D') where N=batch number, I=members, D'=member dimensionality

            feature_matrix_batch = self.final_generator(latent_vector_batch, struct_feature_matrix_batch)
            # feature_matrix_batch size = (N,I',D'') where N=batch number, I=members, D'=member dimensionality
            output = self.final_discriminator(feature_matrix_batch, one_hot_encoding_batch)
            # output size = (N)
            return output, struct_feature_matrix_batch, feature_matrix_batch




