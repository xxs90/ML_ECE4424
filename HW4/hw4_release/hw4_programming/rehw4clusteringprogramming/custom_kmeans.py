#  =========================================================
#  HW 4: Unsupervised Learning, K-Means Clustering
#  CS 4824 / ECE 4484, Spring '21
#  Written by Matt Harrington, Haider Ali
#  =========================================================


import numpy as np
import pandas as pd

class CustomKMeans():

    def __init__(self, k):
        self.inertia_ = 0
        self.labels_ = 0
        self.k_ = k

    def fit(self, data):
        self.data = data
        self.n = data.shape[0]
        self.labels_ = [random.randint(0,self.k_-1) for i in range(self.n)]
        self.inertia_ = np.sum(np.arange(self.n))

        #  ===============================================
        #  TODO: IMPLEMENT KMEANS
        #  ===============================================

        return self
