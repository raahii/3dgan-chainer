import sys, os

import numpy as np
import chainer.cuda
from chainer import serializers
from chainer import Variable
from model.net import Generator
from common.data_io import write_binvox
from visualize import plot_voxel

def main():
    if len(sys.argv) < 4:
        print("usage: python generate_samples.py <model_file> <save_dir> <num smaples>")
        sys.exit()
    
    model_file = sys.argv[1]
    save_dir   = sys.argv[2]
    num        = int(sys.argv[3])
    
    # load model
    gen = Generator()
    serializers.load_npz(model_file, gen)

    try:
        xp = chainer.cuda.cupy
    except AttributeError:
        xp = np
    
    # generate samples
    # z = Variable(xp.asarray(gen.make_hidden(num, train=False)))
    z = Variable(xp.asarray(gen.make_hidden(num)))
    x = gen(z)
    
    try:
        x = chainer.cuda.to_cpu(x.data)
    except:
        x = x.data

    if save_dir[-1] != '/':
        save_dir += '/'

    # save
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir + 'binvox', exist_ok=True)
    os.makedirs(save_dir + 'png', exist_ok=True)
    for i in range(num):
        write_binvox(x[i,0], save_dir+"binvox/{:03d}.binvox".format(i+1))
        data = np.rint(x[i,0]).astype(np.bool).transpose(0, 2, 1)
        plot_voxel(data, save_file = save_dir+"png/{:03d}.png".format(i+1))
        print("*", end='')
        if (i+1) % 5 == 0:
            print("")

if __name__ == "__main__":
    main()
