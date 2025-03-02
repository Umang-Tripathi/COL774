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


theta0_vals = np.linspace(-2,  theta[0] + 2, 50)  
theta1_vals = np.linspace(-2,  theta[1] + 10, 50)
theta0_mesh, theta1_mesh = np.meshgrid(theta0_vals, theta1_vals)


J_vals = np.zeros_like(theta0_mesh)
m = len(y)

for i in range(theta0_mesh.shape[0]):
    for j in range(theta0_mesh.shape[1]):
        theta_test = np.array([theta0_mesh[i, j], theta1_mesh[i, j]])
        predictions = np.c_[np.ones(m), X] @ theta_test
        J_vals[i, j] = np.sum((predictions - y) ** 2) / (2 * m)


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')


ax.plot_surface(theta0_mesh, theta1_mesh, J_vals, cmap='viridis', alpha=0.7)


theta0_path, theta1_path = theta_history[:, 0], theta_history[:, 1]
J_path = [
    np.sum((np.c_[np.ones(m), X] @ theta - y) ** 2) / (2 * m)
    for theta in theta_history
]
#ax.scatter(theta0_path, theta1_path, J_path, c='red', marker='o', label="GD Path")

ax.plot(theta0_path, theta1_path, J_path, color='blue', label="GD Path Line")


ax.set_xlabel("Theta 0")
ax.set_ylabel("Theta 1")
ax.set_zlabel("Cost J_theta")
ax.set_title("Gradient Descent Path on Cost Function Surface")
ax.view_init(elev=20, azim=135)
ax.legend()

first_pause = True


for i in range(len(theta_history)):
    ax.scatter(theta_history[i, 0], theta_history[i, 1], J_path[i], c='red', marker='o',s=5)

    plt.draw()  
    if first_pause:
        first_pause = False
        plt.pause(5)
    plt.pause(0.2) 

plt.show()