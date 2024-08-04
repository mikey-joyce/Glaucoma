import numpy as np
from FCM import FCM


# A possibilistic approach to clustering; possibilistic c-means (PCM)
class PCM:
    def __init__(self, data, num_clusters, m=2, max_epochs=1000, tol=1e-2, centers=None):
        self.data = data
        self.num_clusters = num_clusters
        self.m = m
        self.max_epochs = max_epochs
        self.tol = tol
        self.membership = None

        try:
            if not centers:
                fcm = FCM(data, num_clusters, m=m, max_epochs=1)
                centers, membership, fpc = fcm.fit()
                self.centers = centers
        except:
            self.centers = centers

    def harden_membership(self, membership):
        hardened = []

        for x in range(membership.shape[0]):
            hardened.append(membership[x].argmax())

        return np.array(hardened)

    def update_membership(self):
        membership = np.zeros((self.data.shape[0], self.num_clusters))

        for i in range(self.data.shape[0]):
            for j in range(self.num_clusters):
                numerator = np.linalg.norm(self.data[i] - self.centers[j])
                denominator = self.tol

                for k in range(self.num_clusters):
                    dist = np.linalg.norm(self.data[i] - self.centers[k])
                    dist = np.maximum(dist, 1e-8)  # Replace zeros with 1e-8
                    denominator += (numerator / dist) ** (1 / (self.m - 1))

                membership[i][j] = 1 / denominator

        return membership

    def update_center(self):
        return np.sum((self.membership[:, :, np.newaxis] ** self.m) * self.data[:, np.newaxis, :], axis=0) / np.sum(self.membership[:, :, np.newaxis] ** self.m, axis=0)

    def fit(self):
        for _ in range(self.max_epochs):
            old_centers = self.centers.copy()

            self.membership = self.update_membership()

            self.centers = self.update_center()

            # convergence ?
            if np.linalg.norm(self.centers - old_centers) < self.tol:
                break

        return self.centers, self.harden_membership(self.membership)

    def predict(self, data, harden=True):
        self.data = data
        preds = self.update_membership()

        if harden:
            preds = self.harden_membership(preds)

        return preds

