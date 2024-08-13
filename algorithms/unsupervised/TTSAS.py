import numpy as np
import math
from statistics import mode
from Unsupervised.BSAS import BSAS

class TTSAS(BSAS):
    def __init__(self):
        BSAS.__init__(self)

    def fit(self, data, thresh1=1, thresh2=1.09):
        clas, unclustered = np.zeros(data.shape[0]), data.shape[0]
        m, prev, curr, exists = -1, 0, 0, 0

        clusters = []  # initialize cluster array
        self.centers = []
        membership = np.zeros(data.shape[0])

        while unclustered > 0:
            print(unclustered)
            temp_count = 0
            for i in range(data.shape[0]):
                # this if statement is provided in the book
                if clas[i] == 0 and temp_count == 0 and exists == 0:
                    m += 1
                    clusters.append([data[i]])
                    self.centers.append(data[i])
                    membership[i] = m
                    clas[i] = 1
                    curr += 1
                    unclustered -= 1
                elif clas[i] == 0:
                    distance, k = self.find_cluster(data, i)
                    if distance < thresh1:
                        clusters[k].append(data[i])
                        size = len(clusters[k])
                        self.centers[k] = ((size - 1) * self.centers[k] + data[i]) / size
                        membership[i] = k
                        clas[i] = 1
                        curr += 1
                        unclustered -= 1
                    elif distance > thresh2:
                        m += 1
                        clusters.append([data[i]])
                        self.centers.append(data[i])
                        membership[i] = m
                        clas[i] = 1
                        curr += 1
                        unclustered -= 1
                elif clas[i] == 1:
                    curr += 1

                temp_count += 1
            exists = np.abs(curr - prev)
            prev = curr
            curr = 0

        return membership

    def create_cluster(self, data, index, m, clusters, membership, clas, curr, unclustered):
        m += 1
        clusters.append([data[index]])
        self.centers.append(data[index])
        membership[index] = m
        clas[index] = 1
        curr += 1
        unclustered -= 1
