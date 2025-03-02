import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

from linear_regression import LinearRegressor


def load_data():
    X = np.loadtxt("../data/Q1/linearX.csv", delimiter=',', dtype=float)
    if X.ndim == 1:
        X = X[:, np.newaxis]
    y = np.loadtxt("../data/Q1/linearY.csv", dtype=float)

    return X, y

X, y = load_data()


model = LinearRegressor()
params_list = model.fit(X, y, learning_rate=0.001)
theta_history = np.array(params_list)

theta = model.theta


theta0_vals = np.linspace(-2, theta[0] + 2, 50) 
theta1_vals = np.linspace(-2, theta[1] + 10, 50)
theta0_mesh, theta1_mesh = np.meshgrid(theta0_vals, theta1_vals)


J_vals = np.zeros_like(theta0_mesh)
m = len(y)

for i in range(theta0_mesh.shape[0]):
    for j in range(theta0_mesh.shape[1]):
        theta_test = np.array([theta0_mesh[i, j], theta1_mesh[i, j]])
        predictions = np.c_[np.ones(m), X] @ theta_test
        J_vals[i, j] = np.sum((predictions - y) ** 2) / (2 * m)


plt.figure(figsize=(8, 6))
contour = plt.contourf(theta0_mesh, theta1_mesh, J_vals, levels=np.logspace(-2, 3, 20), cmap="YlGnBu")  
plt.colorbar(contour)  


theta0_path, theta1_path = theta_history[:, 0], theta_history[:, 1]


plt.xlabel("Theta 0")
plt.ylabel("Theta 1")
plt.title("Gradient Descent Path on Cost Function Contours")
plt.legend()


plt.axis('equal')

first_pause = True

for i in range(len(theta_history)):
    plt.scatter(theta_history[i, 0], theta_history[i, 1], c='red', marker='o', s=5)
    plt.draw() 
    if first_pause:
        first_pause = False
        plt.pause(5)
    plt.pause(0.2) 
print("done")
plt.show()