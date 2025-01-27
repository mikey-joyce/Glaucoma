import numpy as np

class KMeans:
    def __init__(self, data, num_clusters, max_epochs=1000, tol=1e-2):
        self.data = data

        self.num_clusters = num_clusters
        self.max_epochs = max_epochs
        self.tol = tol

        # parameters & labels
        self.centers = self.data[np.random.choice(self.data.shape[0], self.num_clusters, replace=False)]
        self.membership = None

    def __get_distance(self, point):
        return np.linalg.norm(point - self.centers, axis=1)

    def __expectation(self, data):
        distances = np.array([self.__get_distance(d) for d in data])
        return np.argmin(distances, axis=1)

    def __maximization(self):
        self.centers = [np.mean(self.data[self.membership == i], axis=0) for i in range(self.num_clusters)]

    def fit(self):
        for _ in range(self.max_epochs):
            old_centers = np.copy(self.centers)

            self.membership = self.__expectation(self.data)
            self.__maximization()

            # convergence ?
            if np.linalg.norm(self.centers - old_centers) < self.tol:
                break

    def predict(self, data):
        return self.__expectation(data)
