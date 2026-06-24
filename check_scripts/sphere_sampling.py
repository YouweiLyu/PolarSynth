import os
from numpy import pi, cos, sin, arccos, arange
# import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt



def sphere_spiral(num_pts, save_dir):
    indices = arange(0, num_pts, dtype=float) + 0.5

    phi = arccos(1 - 2*indices/num_pts)
    theta = pi * (1 + 5**0.5) * indices

    x, y, z = cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi)
    plt.figure().add_subplot(111, projection='3d').scatter(x, y, z)
    plt.show()
    save_path = os.path.join(save_dir, f'sphere_spiral_sampling_{num_pts}pts.png')
    plt.savefig(save_path)

def hemisphere_spiral(num_pts, save_dir):
    indices = arange(0, num_pts, dtype=float) + 0.5

    phi = arccos(1 - indices/num_pts)
    theta = pi * (1 + 5**0.5) * indices

    x, y, z = cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi)
    plt.figure().add_subplot(111, projection='3d').scatter(x, y, z)
    plt.show()
    save_path = os.path.join(save_dir, f'sphere_spiral_sampling_{num_pts}pts.png')
    plt.savefig(save_path)

if __name__ == "__main__":
    num_spp = 1000
    save_dir = 'figures'
    os.makedirs(save_dir, exist_ok=True)
    hemisphere_spiral(num_spp, save_dir)
