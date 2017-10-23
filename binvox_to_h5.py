#
# Convert dataset(.binvox) to numpy array in advance
# to load dataset more efficient.
#
import sys, os, glob
from common.data_io import read_binvox, read_h5
import numpy as np
import h5py

def convert_h5(in_path, out_path):
    data = read_binvox(in_path)
    f = h5py.File(out_path, 'w')
    f.create_dataset('data', data = data)
    f.flush()
    f.close()

    return out_path

def check_open_h5(save_dir):
    if save_dir[-1] != '/':
        save_dir += '/'

    files = [f for f in glob.glob(save_dir + "**/*.h5", recursive=True)]
    for path in files:
        try:
            read_h5(path)
        except KeyError:
            print(path)

def extract_name(path):
    return path.split('/')[-1].split('.')[0]

def main():
    if len(sys.argv) < 3:
        print("usage: python dump.py dataset_path(.binvox) save_dir")
        sys.exit()
    
    dataset_path = sys.argv[1]
    save_dir = sys.argv[2]
    
    if dataset_path[-1] != '/':
        dataset_path += '/'
    if save_dir[-1] != '/':
        save_dir += '/'

    os.makedirs(save_dir, exist_ok=True)
    files = [f for f in glob.glob(dataset_path + "**/*.binvox", recursive=True)]
    print("{} files found.".format(len(files)))
    
    ext = sys.argv[1]
    for in_path in files:
        out_path = save_dir + extract_name(in_path) + '.h5'
        result = convert_h5(in_path, out_path)

if __name__=="__main__":
    # main()
    check_open_h5(sys.argv[1])
