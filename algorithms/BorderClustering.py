import numpy as np
from FCM import FCM
from PCM import PCM


class BorderClustering:
    def __init__(self, data, num_clusters, m1=2, m2=3, thresh=0.1, centers=None):
        self.data = data
        self.n_clusters = num_clusters
        self.m1 = m1
        self.m2 = m2
        self.thresh = thresh
        self.centers = centers

    def fit(self):
        if self.centers is None:
            model = FCM(self.data, self.n_clusters, m=self.m1)
        else:
            model = FCM(self.data, self.n_clusters, m=self.m1, centers=self.centers)
        _, membership, _ = model.fit()

        border_points, bp_indexs = self.find_border_points(model.membership.copy(), self.data)

        return self.cluster_borders(border_points, bp_indexs, model.centers, membership), model.centers

    def find_border_points(self, memb, data):
        border_points, bp_indexs = [], []
        for i in range(len(memb)):
            curr = memb[i]
            two_largest = np.sort(curr)[-2:]
            diff = two_largest[1] - two_largest[0]
            if diff <= self.thresh:
                border_points.append(data[i])
                bp_indexs.append(i)

        border_points = np.array(border_points)

        return border_points, bp_indexs

    def cluster_borders(self, bps, bp_is, centers, memb):
        pcm = PCM(bps, self.n_clusters, m=self.m2, centers=centers)
        pcm_preds = pcm.predict(bps)

        final_labels, count = [], 0
        for i in range(len(memb)):
            if count < len(bp_is):
                if i == bp_is[count]:
                    final_labels.append(pcm_preds[count])
                    count += 1
                else:
                    final_labels.append(memb[i])
            else:
                final_labels.append(memb[i])

        return final_labels
