import sys, os, glob
import random
import numpy as np
from common.data_io import read_binvox

DEVELOPMENT = os.getenv("ENV") == "DEVELOPMENT"
if not DEVELOPMENT:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use("ggplot")

def plot_voxels(in_path, title, out_path):
    files = glob.glob(in_path + '*.binvox')
    
    # plot
    for i, path in enumerate(files):
        print(path)
        voxel = read_binvox(path)

        plot_voxel(voxel.astype(np.bool),
                   '{}{}.png'.format(out_path, i))

def plot_voxel(voxel, title, save_file = None):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxel, edgecolor='k')
    # plt.title(title)
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file)

def main():
    # check arguments
    if len(sys.argv) == 1:
        print("usage: python visualize.py <file or directory> (<save dir(if you want to save as png)>)")
        sys.exit()
    
    # .binvox paths
    path = sys.argv[1]
    if path.split('.')[-1] == 'binvox':
        files = [path]
    else:
        if path[-1] != '/':
            path += '/'
        print(path)
        files = glob.glob(path + '*.binvox')

    if len(files) == 0:
        print("invalid path (.binvox file not found)")
        sys.exit()
    else:
        print("{} files found".format(len(files)))
    
    # save directory
    save_dir = None
    if len(sys.argv) > 2:
        save_dir = sys.argv[2]
        if save_dir[-1] != '/':
            save_dir += '/'
    os.makedirs(save_dir, exist_ok=True)
    
    # plot
    for i, path in enumerate(files):
        # print('plotting {}/{}'.format(i+1, len(files)))
        voxel = read_binvox(path, fix_coords=False)
        filename = path.split('/')[-1]

        if save_dir is None:
            plot_voxel(voxel.astype(np.bool), filename)
        else:
            plot_voxel(voxel.astype(np.bool), filename,
                       '{}{}.png'.format(save_dir, i))

if __name__=="__main__":
    main()
