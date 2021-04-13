#  =========================================================
#  HW 4: Unsupervised Learning, K-Means Clustering
#  CS 4824 / ECE 4484, Spring '21
#  Written by Matt Harrington, Haider Ali
#  =========================================================

# Standard imports
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def euclDistance(a, b):
    return np.sqrt(sum((b - a) ** 2))


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
        clusterAssignment = np.array(np.zeros((n, 2)))
        # the first col determine the sample's cluster,
        # second col determine the error or distance between the sample and centroids
        clusterChanged = True

        while clusterChanged:
            clusterChanged = False
            # for each sample
            for i in range(n):
                minDis = 100000.0
                minIndex = 0
                # for each centroid
                # find the centroid which is closest
                for j in range(self.k_):
                    distance = euclDistance(self.centroids.values[j, :], data.values[i, :])
                    if distance < minDis:
                        minDis = distance
                        clusterAssignment[i, 1] = minDis
                        minIndex = j

                # update the cluster
                if clusterAssignment[i, 0] != minIndex:
                    clusterChanged = True
                    clusterAssignment[i, 0] = minIndex

            # update the centroids
            for j in range(self.k_):
                cluster_index = np.nonzero(clusterAssignment[:, 0] == j)
                point_in_cluster = data.values[cluster_index]
                self.centroids.values[j, :] = np.mean(point_in_cluster, axis=0)

        #  ====================== TODO: IMPLEMENT KMEANS ======================= 

        #  while (not converged): # psuedocode - up to you to implement stopping criterion
        self.labels_ = [clusterAssignment[i, 0] for i in range(n)]  # update the cluster to labels
        self.inertia_ = np.sum(clusterAssignment[:, 1]) # Update the inertia

        # show data & centroids at each iteration when testing performance
        if plot_steps:
            self.plot_state()
            self.iteration += 1
        #  =====================================================================

        return self

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
