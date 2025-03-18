import numpy as np
from collections import Counter

def euclidian(x1, x2):
    return np.sqrt(np.sum(x1-x2)**2)

class KNN:
    def __init__(self, k=3):
        self.k = k
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions
        
    def _predict(self, x):
        
        #compute distances
        distance =  [euclidian(x, x_train) for x_train in self.X_train]



        #get closest neighbor
        k_indeces = np.argsort(distance)[:self.k]
        k_lables  = [self.y_train[i] for i in k_indeces]
        most_common = Counter(k_lables).most_common()
        return most_common[0][0]