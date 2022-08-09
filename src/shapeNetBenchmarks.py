import os.path
from pytorch3d.loss import chamfer_distance
from sklearn.cluster import KMeans
from evalMetrics import mmd_cd, coverage_cd
import generatorArchi
import torch
from torch_geometric.transforms import FixedPoints, NormalizeScale
from torch_geometric.datasets import ShapeNet
import torch_geometric.data as data

generator = generatorArchi.generator

def load_parameters(load_path, generator):
    checkpoint = torch.load(load_path, map_location=torch.device("cpu"))
    generator.load_state_dict(checkpoint['model_state_dict'])

def cluster(feature_matrix_batch, n_clusters, init, n_init):
    feature_matrix_batch = torch.clone(feature_matrix_batch).detach().cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters, init=init, n_init=n_init)
    centroids_batch = []
    for feature_matrix in feature_matrix_batch:
        kmeans.fit(feature_matrix)
        centroids_batch.append(kmeans.cluster_centers_)
    return torch.tensor(centroids_batch)

def dataset():
    test_data = ShapeNet("ShapeNet_test_data", categories="Airplane", split="test", transform=lambda x: (NormalizeScale()(FixedPoints(872)(x))))
    test_dataloader = data.DataLoader(dataset=test_data, batch_size=len(test_data), shuffle=False)
    dataset = None
    for batch in test_dataloader:
        dataset = batch.pos.view(len(test_data), 872, -1)
    return dataset


def generate_statistics(data, generator, n_clusters, init, n_init):
    feature_matrix_batch = data
    struct_feature_matrix_batch = cluster(feature_matrix_batch, n_clusters, init, n_init)
    gen_feature_matrix_batch = generator(struct_feature_matrix_batch)
    struct_gen_feature_matrix_batch = cluster(gen_feature_matrix_batch, n_clusters, init, n_init)
    metric1 = mmd_cd(gen_feature_matrix_batch, feature_matrix_batch)
    metric2 = coverage_cd(gen_feature_matrix_batch, feature_matrix_batch)
    metric3, _ = chamfer_distance(struct_gen_feature_matrix_batch, struct_feature_matrix_batch)
    print((metric1, metric2, metric3))
    torch.save((metric1, metric2, metric3), "shapenetbenchmarks2.pth")


load_parameters('train_2_epoch199.pth', generator)
generate_statistics(dataset(), generator, 200, "k-means++", 1)
