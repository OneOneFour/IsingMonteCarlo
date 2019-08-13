'''
Custom Implementation of the Support Vector Machine Algorithm
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import metrics
from ising import IsingData

data_dict = {
    -1: np.array([[1, 7],
                  [2, 8],
                  [3, 8]]),
    1: np.array([[5, 1],
                 [6, -1],
                 [7, 3]])
}

# class SupportVectorMachine:
#     # Find x.w + b, optimise w,b for test data
#     # minimising ||w|| corresponds to maximising seperation
#     def __init__(self, visualization=False):
#         self.vis = visualization
#         self.colour = {1: 'r', -1: 'b'}
#         if self.vis:
#             self.fig = plt.figure()
#             self.ax = self.fig.add_subplot(111)
#
#     def plot(self, data):
#         for k, v in data.items():
#             x, y = v.T
#             self.ax.scatter(x, y, c=self.colour[k])
#         plt.show()
#
#     def fit(self, data):
#         self.data = data
#         # Minimise langrangian L =  sum(lambda_i) - 1/2 sum_i sum_j lamda_i y_i x_i DOT lambda_j y_j x_j
#
#         opt_dict = {}  # Opt dict is of the form {||w|| : [w,b]} once optimised select w,b with minimal key val
#         # transforms = [
#         #     [1, 1], [-1, 1], [-1, -1], [1, -1]
#         # ] We can do better than this
#
#     # Features in this case will be 1/-1 values of the lattice sites
#     def predict(self, features):
#         # Get the sign of (x.w +b)
#         classification = np.sign(np.dot(np.array(features), self.w) + self.b)
#         return classification

if __name__ == '__main__':
    ttf = IsingData(size=50)
    ttf.load("../dumps/dump_testT00-1(old).json")
    (train_data, train_labels),(test_data, test_labels) = ttf.get_data_flattened()
    svm = SVC(kernel="linear")

    svm.fit(train_data, train_labels)

    predict_labels = svm.predict(test_data)
    print(f"Accuracy:{metrics.accuracy_score(test_labels, predict_labels)}")
