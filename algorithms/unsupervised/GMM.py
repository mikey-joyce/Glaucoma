import numpy as np
from scipy.stats import multivariate_normal  # for the gaussian pdf :)

class GMM:
    def __init__(self, data, num_clusters, centers=None, max_epochs=1000, tol=1e-17):
        self.data = data
        self.num_nodes, self.num_features = self.data.shape

        self.num_clusters = num_clusters
        self.max_epochs = max_epochs
        self.tol = tol

        # parameters and labels
        if centers is not None:
            self.centers = centers
        else:
            self.centers = self.data[np.random.choice(self.data.shape[0], self.num_clusters, replace=False)]

        self.covariances = [np.cov(data, rowvar=False)] * self.num_clusters
        self.weights = np.ones(self.num_clusters) / self.num_clusters
        self.responsibilities = None
        self.predict_responsiblities = None

        self.old_llh = 100000

    def __expectation(self, data):  # calculate soft labels
        distribution = np.array([multivariate_normal.pdf(data, mean=self.centers[c], cov=self.covariances[c]) for c in range(self.num_clusters)]).T * self.weights
        return distribution / np.sum(distribution, axis=1, keepdims=True)

    def __maximization(self): # update params
        num_resp = np.sum(self.responsibilities, axis=0)

        self.weights = num_resp / self.num_nodes
        self.centers = np.dot(self.responsibilities.T, self.data) / num_resp[:, np.newaxis]
        self.covariances = [np.dot((self.responsibilities[:, c][:, np.newaxis] * (self.data - self.centers[c])).T, (self.data - self.centers[c])) / num_resp[c] for c in range(self.num_clusters)]

    def fit(self):
        for i in range(self.max_epochs):
            self.responsibilities = self.__expectation(self.data)
            self.__maximization()

            # convergence ?
            llh = np.sum(np.log(np.sum(self.responsibilities, axis=1)))
            if np.abs(llh - self.old_llh) < self.tol:
                break

            self.old_llh = llh

        self.old_llh = None

    def predict(self, data):
        self.predict_responsiblities = self.__expectation(data)
        return np.argmax(self.predict_responsiblities, axis=1)
