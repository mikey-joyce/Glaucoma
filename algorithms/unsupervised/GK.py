import numpy as np

class GK:
    '''
    Implementation of the Gustafson-Kessel (GK) algorithm
    '''
    def __init__(self, data, num_clusters, m=2, max_epochs=1000, tol=1e-2, centers=None):
        self.data = data
        self.num_clusters = num_clusters
        self.m = m
        self.max_epochs = max_epochs
        self.tol = tol
        self.membership = None
        self.covariance = None
        self.distances = None

        try:
            if not centers:
                self.centers = self.data[np.random.choice(self.data.shape[0], self.num_clusters, replace=False)]
        except:
            self.centers = centers

    def membership_init(self):
        """
        Initialize the fuzzy membership matrix.
        """
        membership = np.random.random((self.data.shape[0], self.num_clusters))
        membership /= np.sum(membership, axis=1, keepdims=True)
        return membership

    def update_center(self):
        """
        Calculate cluster centers.
        """
        n_clusters = self.membership.shape[1]
        centers = np.zeros((n_clusters, self.data.shape[1]))
        for i in range(n_clusters):
            num = np.sum((self.membership[:, i, np.newaxis] ** self.m) * self.data, axis=0)
            den = np.sum(self.membership[:, i] ** self.m)
            centers[i, :] = num / den
        return centers

    def update_membership(self):
        """
        Update the membership matrix.
        """
        n_samples = self.data.shape[0]
        n_clusters = self.centers.shape[0]
        membership = np.zeros((n_samples, n_clusters))

        for i in range(n_samples):
            for j in range(n_clusters):
                summation = 0
                for k in range(n_clusters):
                    num = np.linalg.norm(self.data[i, :] - self.centers[j, :])
                    den = np.linalg.norm(self.data[i, :] - self.centers[k, :])
                    summation += (num / den) ** (2 / (self.m - 1))
                membership[i, j] = 1 / summation

        return membership

    def update_covariance(self):
        """
        Compute covariance matrix for each cluster.
        """
        n_clusters = self.centers.shape[0]
        covariance = [np.zeros((self.data.shape[1], self.data.shape[1])) for _ in range(n_clusters)]

        for j in range(n_clusters):
            num = np.zeros((self.data.shape[1], self.data.shape[1]))
            den = np.sum(self.membership[:, j] ** self.m)

            for i in range(self.data.shape[0]):
                diff = (self.data[i, :] - self.centers[j, :])[np.newaxis]
                num += (self.membership[i, j] ** self.m) * np.dot(diff.T, diff)

            covariance[j] = num / den

        return covariance

    def get_distance(self):
        """
        Calculate the adaptive distance.
        """
        n_samples = self.data.shape[0]
        n_clusters = self.centers.shape[0]
        distances = np.zeros((n_samples, n_clusters))

        for i in range(n_samples):
            for j in range(n_clusters):
                diff = (self.data[i, :] - self.centers[j, :])[np.newaxis]
                inv_cov = np.linalg.inv(self.covariance[j])
                distances[i, j] = np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T))

        return distances

    def harden_membership(self, membership):
        hardened = []

        for x in range(membership.shape[0]):
            hardened.append(membership[x].argmax())

        return np.array(hardened)

    def fit(self):
        """
        Gustafson-Kessel algorithm.
        """
        self.membership = self.membership_init()

        for iteration in range(self.max_epochs):
            self.centers = self.update_center()
            self.covariance = self.update_covariance()
            self.distances = self.get_distance()
            self.membership = self.update_membership()

        return self.centers, self.harden_membership(self.membership)

