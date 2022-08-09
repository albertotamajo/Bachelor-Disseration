import os.path
from pytorch3d.loss import chamfer_distance
from sklearn.cluster import KMeans
from evalMetrics import mmd_cd, coverage_cd
import generatorArchi
import torch
from torch_geometric.io import read_off
from torch_geometric.transforms import SamplePoints, NormalizeScale

generator = generatorArchi.generator


def load_parameters(load_path, generator):
    checkpoint = torch.load(load_path)
    generator.load_state_dict(checkpoint['model_state_dict'])

def cluster(feature_matrix_batch, n_clusters, init, n_init):
    feature_matrix_batch = torch.clone(feature_matrix_batch).detach().cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters, init=init, n_init=n_init)
    centroids_batch = []
    for feature_matrix in feature_matrix_batch:
        kmeans.fit(feature_matrix)
        centroids_batch.append(kmeans.cluster_centers_)
    return torch.tensor(centroids_batch)

def dataset(category, traintest, sample_points):
    path = os.path.join("ModelNet10_train_data", "raw", category, traintest)
    dataset = torch.stack([NormalizeScale()(SamplePoints(sample_points)(read_off(os.path.join(path, f)))).pos for f in os.listdir(path) if f.endswith('.off')])
    return dataset

def generate_statistics(data, generator, n_clusters, init, n_init):
    feature_matrix_batch = data
    struct_feature_matrix_batch = cluster(feature_matrix_batch, n_clusters, init, n_init)
    gen_feature_matrix_batch = generator(struct_feature_matrix_batch)
    struct_gen_feature_matrix_batch = cluster(gen_feature_matrix_batch, n_clusters, init, n_init)
    metric1 = mmd_cd(gen_feature_matrix_batch, feature_matrix_batch)
    metric2 = coverage_cd(gen_feature_matrix_batch, feature_matrix_batch)
    metric3, _ = chamfer_distance(struct_gen_feature_matrix_batch, struct_feature_matrix_batch)
    return metric1, metric2, metric3

def generate_statistics_dataset(generator, sample_points, n_clusters, init, n_init):
    path = os.path.join("ModelNet10_train_data", "raw")
    categories = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    metrics = []
    for category in categories:
        metrics.append(generate_statistics(dataset(category, "test", sample_points), generator, n_clusters, init, n_init))
    torch.save(metrics, "modelNet10Benchmarks.pth")

load_parameters('epoch110.pth', generator)
generate_statistics_dataset(generator, 872, 200, "k-means++", 1)
