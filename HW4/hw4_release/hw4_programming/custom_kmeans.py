#  =========================================================
#  HW 4: Unsupervised Learning, K-Means Clustering
#  CS 4824 / ECE 4484, Spring '21
#  Written by Matt Harrington, Haider Ali
#  Name: Guanang Su
#  VT email id: guanang
#  =========================================================

# Standard imports
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class CustomKMeans():

    # Initialize all attributes
    def __init__(self, k):
        self.k_ = k  # Number of clusters
        self.labels_ = 0  # Each sample's cluster label
        self.inertia_ = 0  # Sum of all samples' distances from their centroids

    # Find K cluster centers & label all samples
    def fit(self, data, plot_steps=False):
        # Fit the PCA module & Transform our data for later graphing
        self.pca = PCA(2).fit(data)
        self.data = pd.DataFrame(data)
        self.data_pca = pd.DataFrame(self.pca.transform(data))
        self.data_pca.columns = ['PC1', 'PC2']

        # Initialize variables
        self.iteration = 1
        n = data.shape[0]

        # Initialize centroids to random datapoints
        self.centroids = data.iloc[np.random.choice(range(n), self.k_, replace=False)].copy()
        self.centroids.index = np.arange(self.k_)

        #  =====================================================================
        #  ====================== TODO: IMPLEMENT KMEANS ======================= 

        #  while (not converged): # psuedocode - up to you to implement stopping criterion
        clusters = np.array(np.zeros((n, 2)))
        cluster_update = True

        while cluster_update == True:
            cluster_update = False

            for i in range(n):
                distance = float("inf")
                index = 0

                for j in range(self.k_):
                    dist = np.sqrt(np.sum((self.centroids.values[j, :] - data.values[i, :]) ** 2))

                    if dist < distance:
                        distance = dist
                        clusters[i, 1] = distance
                        index = j

                if clusters[i, 0] != index:
                    clusters[i, 0] = index
                    cluster_update = True

            for a in range(self.k_):
                cluster_index = np.nonzero(clusters[:, 0] == a)
                datapoint = data.values[cluster_index]
                self.centroids.values[j, :] = np.mean(datapoint, axis=0)

        self.labels_ = [clusters[i, 0] for i in range(n)]  # Update
        self.inertia_ = np.sum(clusters[:, 1])  # Update


        # show data & centroids at each iteration when testing performance
        if plot_steps:
            self.plot_state()
            self.iteration += 1
        #  =====================================================================

        return self

#     closest_centroid(self.centriods, data)
#
#     # show data & centroids at each iteration when testing performance
#     if plot_steps:
#         self.plot_state()
#         self.iteration += 1
#     #  =====================================================================
#
#     return self
#
#
# def closest_centroid(ic, X):
#     assigned_centroid = []
#     for i in X:
#         distance = []
#         for j in ic:
#             dist = sum((j, j) ** 2) * 0.5
#             distance.append(dist)
#         assigned_centroid.append(np.argmin(distance))

    # Plot projection of data and centroids in 2D
    def plot_state(self):
        # Project the centroids along the principal components
        centroid_pca = self.pca.transform(self.centroids)

        # Draw the plot
        plt.figure(figsize=(8, 8))
        plt.scatter(self.data_pca['PC1'], self.data_pca['PC2'], c=self.labels_)
        plt.scatter(centroid_pca[0], centroid_pca[1], marker='*', s=1000)
        plt.title("Clusters and Centroids After step {}".format(self.iteration))
        plt.show()
