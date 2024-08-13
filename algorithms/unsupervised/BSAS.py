import numpy as np
import math
from statistics import mode


class BSAS:
    def __init__(self):
        self.centers = None

    def fit(self, data, thresh=1, max_clusters=math.inf):
        m = 0  # initialize cluster counter
        clusters = [[data[0]]]  # initialize cluster array
        self.centers = [data[0]]
        membership = [0]

        for i in range(1, data.shape[0]):
            # pick the closest cluster
            distance, k = self.find_cluster(data, i)

            # determine if a new cluster needs to be created
            if distance > thresh and m < (max_clusters - 1):
                m += 1
                clusters.append([data[i]])
                self.centers.append(data[i])
                membership.append(m)
            else:
                clusters[k].append(data[i])
                # recompute center
                size = len(clusters[k])
                self.centers[k] = ((size - 1) * self.centers[k] + data[i]) / size
                membership.append(k)

        return membership

    def find_cluster(self, data, index):
        # get distance of current data point from each cluster center
        distances = []
        for k in range(len(self.centers)):
            distances.append(np.linalg.norm(data[index] - self.centers[k]))

        # pick the closest cluster
        distance = np.min(distances)
        k = np.argmin(distances)

        return distance, k

    def predict(self, data):
        membership = []
        for i in range(data.shape[0]):
            distances = []
            for k in range(len(self.centers)):
                distances.append(np.linalg.norm(data[i] - self.centers[k]))

            membership.append(np.argmin(distances))
        return membership

    def optimize(self, data, c=0.1, s=20):
        similarities = []
        for i in range(data.shape[0]):
            for j in range(data.shape[0]):
                distance = np.linalg.norm(data[i] - data[j])
                if distance != 0:
                    similarities.append(distance)
        a = min(similarities)
        b = max(similarities)

        thresholds, cluster_counts = [], []

        theta = a
        while theta <= b:
            num_clusters = []
            for i in range(s):
                temp = data.copy()
                np.random.shuffle(temp)
                alg = BSAS()
                alg.fit(temp, theta)
                num_clusters.append(len(alg.centers))

            k_count = mode(num_clusters)
            if k_count <= 1:
                return thresholds, cluster_counts

            thresholds.append(theta)
            cluster_counts.append(k_count)
            theta += c

        return thresholds, cluster_counts

    def pick_hyperparams(self, data, c=0.1, s=20):
        thresholds, cluster_counts = self.optimize(data, c, s)

        k = mode(cluster_counts)
        avg, count = 0, 0
        for i in range(len(thresholds) - 1):
            if cluster_counts[i] == k:
                avg += thresholds[i]
                count += 1

        return (avg/count), k

