import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Load saved data
loss_grids = np.load("schoolwork\\sc\\loss_grids.npy")  # Shape: (T, H, W)
optimizer_path = np.load("schoolwork\\sc\\optimizer_path.npy")  # Shape: (T, 2)
T, H, W = loss_grids.shape

# Set up coordinate grid
space = np.linspace(-1, 1, H)
X, Y = np.meshgrid(space, space)

# Create 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(0, 8)
ax.set_xlabel("Alpha (d1)")
ax.set_ylabel("Beta (d2)")
ax.set_zlabel("Loss")

# Initial plot
surf = [ax.plot_surface(X, Y, loss_grids[0], cmap='viridis', edgecolor='none')]
path_dot = ax.plot([optimizer_path[0, 0]], [optimizer_path[0, 1]], 
                   [loss_grids[0][H // 2, W // 2]], 'ro')[0]

def update(frame):
    ax.collections.clear()  # Clear old surface
    surf[0] = ax.plot_surface(X, Y, loss_grids[frame], cmap='viridis', edgecolor='none')
    
    # Path dot (we project the alpha-beta path onto current loss surface midpoint for simplicity)
    alpha, beta = optimizer_path[frame]
    path_dot.set_data([alpha], [beta])
    z = loss_grids[frame][H // 2, W // 2]
    path_dot.set_3d_properties([z])
    return surf + [path_dot]

ani = FuncAnimation(fig, update, frames=T, interval=200, blit=False)

plt.show()
