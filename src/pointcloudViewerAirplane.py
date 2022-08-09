import os.path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import generatorArchi
import torch
from torch_geometric.datasets import ShapeNet
import torch_geometric.data as data
import open3d as o3d
from torch_geometric.io import read_off
from torch_geometric.transforms import FixedPoints, NormalizeScale

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


def visualise(generator, sample_points, n_clusters, init, n_init):
    training_data = ShapeNet("ShapeNet_train_data", categories="Airplane", split="train",
                             transform=lambda x: (NormalizeScale()(FixedPoints(sample_points)(x))))

    training_dataloader = data.DataLoader(training_data, 10, False)
    for batch in training_dataloader:
        train_feature_matrix_batch = batch
        train_feature_matrix_batch = train_feature_matrix_batch.pos.view(10, sample_points, -1)

        struct_train_feature_matrix_batch = cluster(train_feature_matrix_batch, n_clusters, init, n_init)
        # struct_test_feature_matrix_batch = cluster(test_feature_matrix_batch, n_clusters, init, n_init)
        gen_train_feature_matrix_batch = generator(struct_train_feature_matrix_batch).detach().numpy()
        train_feature_matrix_batch = train_feature_matrix_batch.detach().numpy()
        dir = os.path.join("ShapeNetTrainings", "Airplane")
        for i in range(10):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(struct_train_feature_matrix_batch[i])
            o3d.io.write_point_cloud(os.path.join(dir, f"struct_point{i}.ply"), pcd)
            #o3d.visualization.draw_geometries([pcd])

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(gen_train_feature_matrix_batch[i])
            o3d.io.write_point_cloud(os.path.join(dir, f"gen_point{i}.ply"), pcd)
            #o3d.visualization.draw_geometries([pcd])

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(train_feature_matrix_batch[i])
            o3d.io.write_point_cloud(os.path.join(dir, f"real_point{i}.ply"), pcd)
            #o3d.visualization.draw_geometries([pcd])


load_parameters('train_2_epoch199.pth', generator)
checkpoint = torch.load('train_2_epoch199.pth')
losses = checkpoint['train_losses']
visualise(generator, 872, 200, 'k-means++', 1)
