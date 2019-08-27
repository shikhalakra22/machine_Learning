import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
#from sklearn.cluster import KMeans

class K_Means:
    def _init_(self, k = 2, tot=0.001, max_iter=300):
        self.k = k
        self.tot = tot
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for features in X :
                dist = [np.linalg.norm(features-self.centroids[centroid]) for centroid in self.centroids]
                classification = dist.index(min(dist))
                self.classifications[classification].append(features)

            prev_centroids = dict(self_centroids)

            for classification in  self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)
            optimized=True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0)> self.tot:
                   print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                   optimized=False
                   
                                      
            if optimized:
                break

    def predict(self, data):
         dist = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
         classification = dist.index(min(dist))
         return classification
X = np.array([[1, 2],
             [1.5, 1.8],
             [5, 8],
             [8, 8],
             [1, 0.6],
             [9, 11]])

plt.scatter(X[:,0], X[:,1], s=150)
plt.show()



colors = 10*["g","r","c","b","k"]

clf = K_Means()
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker="o", color="K", s=150, linewidths=5)

for classification in clf.classifications:
    color = colors[classification]
    for features in clf.classifications[classification]:
        plt.scatter(features[0], features[1], marker="o", color=color, s=150, linewidths=5)

uk = np.array([[1,3],
               [0,3],
               [8,9],
               [5,4],
               [6,4],])
for u in uk:
    classification = clf.predict(u)
    plt.scatter(u[0], u[1], marker='*',color=colors[classification], s=150, linewidths=5)
    



plt.show()        
