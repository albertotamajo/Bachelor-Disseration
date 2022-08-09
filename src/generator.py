from graphConvPool3DPnet import SelfCorrelation, KMeansConv, LocalAdaptiveFeatureAggregation, GraphMaxPool
import torch
import torch.nn as nn


class ExpandingLayer(nn.Module):

    def __init__(self, mlp: nn.Module, learning_rate: int, k: int, kmeansInit, n_init, sigma: nn.Module,
                 F: nn.Module, W: nn.Module, M: nn.Module, B: nn.Module, C, P, mlp1: nn.Module, mlp2: nn.Module,
                 branching: int, mlp3: nn.Module):
        """
        An expanding layer is a stacked sequence of modules:
            -Self-correlation
            -K-Means Convolution
            -Local adaptive Feature Aggregation
            -Graph Max Pool

        :param mlp: mlp: R^C -> R^C
        :param learning_rate: learning rate for the self-correlation module
        :param k: number of clusters for each point cloud
        :param kmeansInit: initializer for the kmeans algorithm
        :param n_init: number of restarts for the kmeans algorithm
        :param sigma: sigma: R^(C+P) -> R^(C+P)
        :param F: F: R^C -> R^(C x (C+P))
        :param W: W: R^C -> R^(C x (C+P))
        :param M: M: R^(C+P) -> R
        :param B: B: R^(C+P) -> R^(C+P)
        :param C: dimensionality of each point
        :param P: augmentation
        :param mlp1: mlp1: R^(C+P) -> R^(C+P)
        :param mlp2: mlp2: R^(C+P) -> R^(C+P)
        :param branching: branching factor indicates how many children nodes each cluster produces
        :param mlp3: mlp3: R^(C+P) -> R^D where D = (C+P) * branching
        """

        super().__init__()
        self.selfCorr = SelfCorrelation(mlp, learning_rate)
        self.kmeansConv = KMeansConv(k, kmeansInit, n_init, sigma, F, W, M, B, C, P)
        self.localAdaptFeaAggre = LocalAdaptiveFeatureAggregation(mlp1, mlp2)
        self.graphMaxPool = GraphMaxPool(k)
        self.upsampling = UpSampling(branching, mlp3)

    def forward(self, feature_matrix_batch):
        # feature_matrix_batch size = (N,I,D) where N=batch number, I=members, D=member dimensionality
        feature_matrix_batch = self.selfCorr(feature_matrix_batch)
        # feature_matrix_batch size = (N,I,D) where N=batch number, I=members, D=member dimensionality
        feature_matrix_batch, conv_feature_matrix_batch, cluster_index = self.kmeansConv(feature_matrix_batch)
        feature_matrix_batch = self.localAdaptFeaAggre(feature_matrix_batch, conv_feature_matrix_batch)
        cluster_encoding_batch = self.graphMaxPool(feature_matrix_batch, cluster_index)
        # cluster_encoding_batch size = (N,K,D) where N=batch number, K=members, D=member dimensionality
        upsamples_batch = self.upsampling(cluster_encoding_batch)
        # upsamples_batch size = (N,I',D) where N=batch number, I'=members, D=member dimensionality)
        output = torch.cat((feature_matrix_batch, upsamples_batch), dim=1)
        # output size = (N,I'',D)  where N=batch number, I''=members, D=member dimensionality
        return output


class UpSampling(nn.Module):

    def __init__(self, branching: int, mlp: nn.Module):
        """
        This module up-samples the number of nodes in a point cloud given its clusters' encodings
        :param branching: branching factor indicates how many children nodes each cluster produces
        :param mlp: mlp: R^D -> R^C where C = D * branching
        """
        super().__init__()
        self.branching = branching
        self.mlp = mlp

    def forward(self, cluster_encoding_batch):
        # cluster_encoding_batch size = (N,I,D) where N=batch number, I=members, D=member dimensionality
        N, I, D = cluster_encoding_batch.size()
        cluster_encoding_batch = cluster_encoding_batch.view(-1, D)
        # cluster_encoding_batch size = (L,D) where L=N*I, D=member dimensionality
        output = self.mlp(cluster_encoding_batch)
        # output size = (L,C) where L=N*I, C=member dimensionality
        output = output.view(N, -1, D)
        # output size = (N,I',D) where N=batch number, I'=members, D=member dimensionality
        return output


class StructureGenerator(nn.Module):

    def __init__(self, i: int, d: int, mlp: nn.Module, expandingLayers: [ExpandingLayer]):
        """
        This module represents a Structural Generator
        :param i: initial number of points
        :param d: initial dimensionality of the points
        :param mlp: mlp: R^D -> R^(i x d) where D= latent vector dimensionality
        :param expandingLayers: a list of expanding layers
        """
        super().__init__()
        self.i = i
        self.d = d
        self.mlp = mlp
        self.neuralnet = nn.Sequential(*expandingLayers)

    def forward(self, latent_vector_batch: torch.Tensor):
        # latent_vector_batch size = (N,D) where N=batch number, D=dimensionality
        N,D = latent_vector_batch.size()
        feature_matrix_batch = self.mlp(latent_vector_batch).view(N, self.i, self.d)
        # feature_matrix_batch size = (N, I, D') where N=batch number, I=members, D'=member dimensionality
        output = self.neuralnet(feature_matrix_batch)
        # output size = (N,I',D'') where N=batch number, I'=members, D''=member dimensionality
        return output


class FinalGenerator(nn.Module):

    def __init__(self, i: int, d: int, mlp: nn.Module, expandingLayers1: [ExpandingLayer], expandingLayers2: [ExpandingLayer]):
        """
        This module represents a Final Generator
        :param i: initial number of points
        :param d: initial dimensionality of the points
        :param mlp: mlp: R^D -> R^(i x d) where D= latent vector dimensionality
        :param expandingLayers1: a list of expanding layers which are given as input the transformation of the latent vector
        :param expandingLayers2: a list of expanding layers which are given as input the structural points
        """
        super().__init__()
        self.i = i
        self.d = d
        self.mlp = mlp
        self.neuralnet1 = nn.Sequential(*expandingLayers1)
        self.neuralnet2 = nn.Sequential(*expandingLayers2)

    def forward(self, latent_vector_batch: torch.Tensor, structural_points_batch: torch.Tensor):
        # latent_vector_batch size = (N,D) where N=batch number, D=dimensionality
        # structural_points_batch size = (N,L,C) where N=batch number, L= members, C=dimensionality
        N, D = latent_vector_batch.size()
        with torch.no_grad():
            feature_matrix_batch = self.mlp(latent_vector_batch).view(N, self.i, self.d)
        # feature_matrix_batch size = (N, I, D') where N=batch number, I=members, D'=member dimensionality
        output1 = self.neuralnet1(feature_matrix_batch)
        # output1 size = (N,I',D'') where N=batch number, I'=members, D''=member dimensionality
        output2 = self.neuralnet2(structural_points_batch)
        # output2 size = (N,I',D'') where N=batch number, I'=members, D''=member dimensionality
        output = output1 + output2
        # output size = (N,I',D'') where N=batch number, I'=members, D''=member dimensionality
        return output