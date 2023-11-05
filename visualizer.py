import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def create_camera_pyramid(R, t, ax, camera_id, scale=0.5):
    # Define the camera in camera coordinates with swapped width and height
    # and scaled by the 'scale' factor
    camera = np.array([[0, 0, 0],                           # Camera position (apex of the pyramid)
                       [-scale, scale, -scale],             # Top-left of the camera's view field
                       [scale, scale, -scale],              # Top-right of the camera's view field
                       [scale, -scale, -scale],             # Bottom-right of the camera's view field
                       [-scale, -scale, -scale]])           # Bottom-left of the camera's view field

    # Convert camera coordinates to world coordinates
    camera_rotated = R.T @ camera.T
    camera_transformed = camera_rotated + t.reshape(-1, 1)

    # Define the edges of the pyramid
    edges = [[camera_transformed[:,0], camera_transformed[:,1]],
             [camera_transformed[:,0], camera_transformed[:,2]],
             [camera_transformed[:,0], camera_transformed[:,3]],
             [camera_transformed[:,0], camera_transformed[:,4]],
             [camera_transformed[:,1], camera_transformed[:,2]],
             [camera_transformed[:,2], camera_transformed[:,3]],
             [camera_transformed[:,3], camera_transformed[:,4]],
             [camera_transformed[:,4], camera_transformed[:,1]]]

    # Create a collection of the edges for plotting
    for edge in edges:
        ax.plot(*zip(*edge), color="k")

    # Create the base of the pyramid for plotting
    base = Poly3DCollection([camera_transformed[:,1:].T], alpha=0.5, facecolors='skyblue')
    ax.add_collection3d(base)

    # Set labels and titles
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Camera {camera_id} Pose')

    # Set limits
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)


def visualize_camera_poses(R1, t1, R2, t2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Assuming you have R1, t1 for the first camera and R2, t2 for the second camera
    create_camera_pyramid(R1, t1, ax, camera_id='1')

    create_camera_pyramid(R2, t2, ax, camera_id='2')

    plt.show()
