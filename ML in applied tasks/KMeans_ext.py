import numpy as np
from math import sqrt


class Point:
    def __init__(self, features: np.ndarray, cluster: int = 0):
        self.features = features.copy()
        self.cluster = cluster

    def set_cluster(self, new_cluster: int):
        self.cluster = new_cluster


def count_dist(p1: Point, p2: Point) -> float:
    distance = 0.0
    for x1, x2 in zip(p1.features, p2.features):
        distance += (x1 - x2) ** 2
    return sqrt(distance)


class KMeans(object):
    def __init__(self, K, init):
        self.K = K
        self.centers = np.array(init)
        self.points = [[] * len(self.centers)]

    def __def_new_centers(self) -> np.ndarray:
        new_points = []
        for _ in range(len(self.centers)):
            new_points.append([])
        for i, cluster in enumerate(self.points):
            for j, point in enumerate(cluster):
                distances = np.zeros(shape=(len(self.centers),), dtype=float)
                for ind, center in enumerate(self.centers):
                    distances[ind] = count_dist(Point(center), point)
                self.points[i][j].set_cluster(distances.argmin(axis=0))
                new_points[self.points[i][j].cluster].append(Point(self.points[i][j].features, self.points[i][j].cluster))
        self.points = new_points
        new_centers = np.zeros(shape=self.centers.shape, dtype=float)
        for i, cluster in enumerate(self.points):
            for point in cluster:
                new_centers[i, :] += point.features
            new_centers[i] /= (len(cluster) + 0.001)
        return new_centers

    def __def_cluster(self, obj: np.ndarray) -> int:
        distances = np.zeros(shape=(len(self.centers),), dtype=float)
        for ind, center in enumerate(self.centers):
            distances[ind] = np.sqrt(np.sum((center - obj) ** 2))
        return distances.argmin()

    def fit(self, X):
        # TODO: write your code here
        for row in X:
            self.points[0].append(Point(row))
        new_centers = self.__def_new_centers()
        while max(np.sqrt(np.sum((self.centers - new_centers) ** 2, axis=1))) > 0.001:
            self.centers = new_centers
            new_centers = self.__def_new_centers()

    def predict(self, X):
        # TODO: write your code here
        labels = np.zeros(shape=(len(X),), dtype=int)
        for ind, row in enumerate(X):
            labels[ind] = self.__def_cluster(row)
        return labels
