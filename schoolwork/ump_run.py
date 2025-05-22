import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Load data
loss_grids   = np.load("schoolwork/loss_grids.npy")     # shape: (N, H, W)
trajectories = np.load("schoolwork/trajectories.npy")   # shape: (N, 3) → (x, y, loss)

# Use the most recent loss grid
Z = loss_grids[-1]
H, W = Z.shape
print(trajectories.shape)
x = np.linspace(-1, 1, W)
y = np.linspace(-1, 1, H)
X, Y = np.meshgrid(x, y)
Z_scaled = Z * 2

# Extract trajectory components
tx = trajectories[:, 0]
ty = trajectories[:, 1]
tz = trajectories[:, 2] * 2  # Scale loss to match surface

# Set up the figure and 3D plot
fig = plt.figure(figsize=(12, 8))
ax  = fig.add_subplot(111, projection='3d')

# Plot the static loss surface
ax.plot_surface(
    X, Y, Z_scaled,
    cmap='viridis', alpha=0.7,
    rstride=1, cstride=1,
    linewidth=0
)

# Trajectory line (red) and current point (black)
trajectory_line, = ax.plot([], [], [], color='red', linewidth=2)
current_dot = ax.scatter([], [], [], color='black', s=50)

# Labels and limits
ax.set_xlabel("Direction 1")
ax.set_ylabel("Direction 2")
ax.set_zlabel("Scaled Loss (×2)")
ax.set_title("Animated Loss Landscape with Optimizer Trajectory")
ax.set_zlim(Z_scaled.min(), Z_scaled.max())

# Init function
def init():
    trajectory_line.set_data([], [])
    trajectory_line.set_3d_properties([])
    current_dot._offsets3d = ([], [], [])
    return trajectory_line, current_dot

# Update function per frame
def update(frame):
    trajectory_line.set_data(tx[:frame], ty[:frame])
    trajectory_line.set_3d_properties(tz[:frame])
    current_dot._offsets3d = ([tx[frame-1]], [ty[frame-1]], [tz[frame-1]])
    return trajectory_line, current_dot

# Animate and save
frames = len(tx)
anim = FuncAnimation(fig, update, frames=frames, init_func=init, interval=50, blit=False)
plt.show()
