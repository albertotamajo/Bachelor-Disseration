import os.path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

import generatorArchi
import torch
from torch_geometric.datasets import ModelNet
import torch_geometric.data as data
import open3d as o3d
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


def visualise(generator, sample_points, n_clusters, init, n_init):
    data1 = NormalizeScale()(SamplePoints(sample_points)(read_off("ModelNet10_train_data\\raw\\bathtub\\train\\bathtub_0001.off"))).pos
    data2 = NormalizeScale()(SamplePoints(sample_points)(read_off("ModelNet10_train_data\\raw\\bed\\train\\bed_0001.off"))).pos
    data3 = NormalizeScale()(SamplePoints(sample_points)(read_off("ModelNet10_train_data\\raw\\chair\\train\\chair_0001.off"))).pos
    data4 = NormalizeScale()(SamplePoints(sample_points)(read_off("ModelNet10_train_data\\raw\\desk\\train\\desk_0001.off"))).pos
    data5 = NormalizeScale()(SamplePoints(sample_points)(read_off("ModelNet10_train_data\\raw\\dresser\\train\\dresser_0001.off"))).pos
    data6 = NormalizeScale()(SamplePoints(sample_points)(read_off("ModelNet10_train_data\\raw\\monitor\\train\\monitor_0001.off"))).pos
    data7 = NormalizeScale()(SamplePoints(sample_points)(read_off("ModelNet10_train_data\\raw\\night_stand\\train\\night_stand_0001.off"))).pos
    data8 = NormalizeScale()(SamplePoints(sample_points)(read_off("ModelNet10_train_data\\raw\\sofa\\train\\sofa_0001.off"))).pos
    data9 = NormalizeScale()(SamplePoints(sample_points)(read_off("ModelNet10_train_data\\raw\\table\\train\\table_0001.off"))).pos
    data10 = NormalizeScale()(SamplePoints(sample_points)(read_off("ModelNet10_train_data\\raw\\toilet\\train\\toilet_0001.off"))).pos
    dat = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10]
    dir = os.path.join("K-MeansExperiments")
    training_dataloader = data.DataLoader(dat, 10, False)
    for batch in training_dataloader:
        train_feature_matrix_batch = batch
        struct_train_feature_matrix_batch = cluster(train_feature_matrix_batch, n_clusters, init, n_init)
        # struct_test_feature_matrix_batch = cluster(test_feature_matrix_batch, n_clusters, init, n_init)
        gen_train_feature_matrix_batch = generator(struct_train_feature_matrix_batch).detach().numpy()
        train_feature_matrix_batch = train_feature_matrix_batch.detach().numpy()

        for i in range(10):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(struct_train_feature_matrix_batch[i])
            o3d.io.write_point_cloud(os.path.join(dir, f"struct_point{i}.ply"), pcd)
            #o3d.visualization.draw_geometries([pcd])

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(gen_train_feature_matrix_batch[i])
            # o3d.io.write_point_cloud(os.path.join(dir, f"gen_point{i}.ply"), pcd)
            # #o3d.visualization.draw_geometries([pcd])
            #
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(train_feature_matrix_batch[i])
            o3d.io.write_point_cloud(os.path.join(dir, f"real_point{i}.ply"), pcd)
            #o3d.visualization.draw_geometries([pcd])






load_parameters('epoch110.pth', generator)
pytorch_total_params = sum(p.numel() for p in generator.parameters())
visualise(generator, 800, 64, 'k-means++', 1)


checkpoint = torch.load("epoch110.pth")
plt.plot(range(1, 112), checkpoint["train_losses"], label = "Training loss")
plt.plot(range(1, 112), checkpoint["test_losses"], label = "Test loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
ax = plt.gca()
ax.set_ylim([0.0, 0.02])
plt.legend()
plt.show()
#plt.savefig("extendinglayerVerification.png")