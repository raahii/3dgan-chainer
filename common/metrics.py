import sys, os, glob
import numpy as np
import chainer.cuda
import chainer.training
from common.data_io import write_binvox
from chainer import Variable
import subprocess
from visualize import plot_voxels
# from notify_slack import post_images, post_text

def save_sample_voxels(gen, dst, num=3):
    """
    save generated voxels as .binvox files
    """
    @chainer.training.make_extension()
    def make_voxels(trainer):
        try:
            xp = chainer.cuda.cupy
        except AttributeError:
            xp = np

        z = Variable(xp.asarray(gen.make_hidden(num)))
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = gen(z)
        
        try:
            x = chainer.cuda.to_cpu(x.data)
        except:
            x = x.data
        
        epoch = trainer.updater.epoch
        save_voxel_dir = dst + str(epoch) + '/'
        if not os.path.exists(save_voxel_dir):
            os.mkdir(save_voxel_dir)
        
        # write as .binvox
        for i in range(x.shape[0]):
            write_binvox(x[i,0], save_voxel_dir + "{:02d}.binvox".format(i))

    return make_voxels

def save_sample_images(gen, dst_voxel, dst_img):
    """
    save generated voxels as .png files
    """
    @chainer.training.make_extension()
    def make_images(trainer):
        try:
            xp = chainer.cuda.cupy
        except AttributeError:
            xp = np
        
        epoch = trainer.updater.epoch
        save_img_dir = dst_img + str(epoch) + '/'
        if not os.path.exists(save_img_dir):
            os.mkdir(save_img_dir)
        
        save_voxel_dir = dst_voxel + str(epoch) + '/'
        
        # plot save_voxel and save 2D image asynchronously with other process
        p = subprocess.Popen('python visualize.py {} {}'.format(save_voxel_dir, save_img_dir),
                             shell=True, stdin=None, stdout=None, close_fds=True)
        # plot_voxels(save_voxel_dir, str(epoch)+'epoch', save_img_dir)

    return make_images

# def post_samples_to_slack(gen, dst):
#     """
#     post saved voxel images to slack channel
#     """
#     @chainer.training.make_extension()
#     def post_slack(trainer):
#         epoch = trainer.updater.epoch - 1
#         save_img_dir = dst + str(epoch) + '/'
#
#         if os.path.exists(save_img_dir):
#             post_images(save_img_dir, str(epoch)+' epoch')
#         else:
#             print('[slack]: image directory is not found: ' + save_img_dir)
#
#     return post_slack
