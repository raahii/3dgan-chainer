import sys, os, glob
from datetime import datetime

import chainer
from chainer import training
from chainer.training import extension
from chainer.training import extensions

from model.updater import Updater
from model.net import Generator, Discriminator
from common import binvox_rw
from common.metrics import save_sample_voxels, save_sample_images
#, post_samples_to_slack

DEVELOPMENT = os.getenv("ENV") == "DEVELOPMENT"
BATCH_SIZE = DEVELOPMENT and 4 or 30
NUM_EPOCH = 1000
DATASET_PATH = 'data/chair_h5/' # I used ShapeNet v2
DATASET_EXT = 'h5' # you can also use 'binvox'
g_lr = 2.5e-3
d_lr = 1e-5
beta = 0.5

def main():
    ###
    # print current environment ( DEVELOPMENT or PRODUCTION )
    print("### {} ###".format(DEVELOPMENT and "DEVELOPMENT" or "PRODUCTION"))

    ###
    # Setup save directories
    now = datetime.now().strftime("%Y_%m%d_%H%M")
    save_dir = 'result/' + now + '/'
    os.makedirs(save_dir, exist_ok=True)
    for name in ['params','voxels', 'images', 'tmpparams']:
        os.makedirs(save_dir + name, exist_ok=True)

    #####
    # Setup training data iterator
    print("preparing dataset...")
    dataset_files = [f for f in glob.glob(DATASET_PATH+"**/*."+DATASET_EXT, recursive=True)]
    if DEVELOPMENT:
        dataset_files = dataset_files[:10]
    dataset_iterator = chainer.iterators.SerialIterator(dataset_files, BATCH_SIZE)
    print("datasetsize: {}".format(len(dataset_files)))
    print("batchsize: {}".format(BATCH_SIZE))
    print("num_batches: {}".format(int(len(dataset_files) / BATCH_SIZE)))

    updater_args = {}
    updater_args["extension"] = DATASET_EXT
    updater_args["iterator"] =  {'main': dataset_iterator}

    #####
    # Setup models and optimizers
    generator     = Generator()
    discriminator = Discriminator()
    models = [generator, discriminator]

    opt_gen = chainer.optimizers.Adam(alpha=g_lr, beta1=beta)
    opt_gen.setup(generator)
    opt_dis = chainer.optimizers.Adam(alpha=d_lr, beta1=beta)
    opt_dis.setup(discriminator)

    opts = {}
    opts["opt_gen"] = opt_gen
    opts["opt_dis"] = opt_dis

    updater_args["optimizer"] = opts
    updater_args["models"] = models

    if DEVELOPMENT:
        updater_args["device"] = -1
    else:
        updater_args["device"] = 0
        chainer.cuda.get_device_from_id(0).use()
        print("using gpu 0")
        for m in models:
            m.to_gpu()

    #####
    # Set up updater and trainer
    updater = Updater(**updater_args)
    trainer = training.Trainer(updater, (NUM_EPOCH, 'epoch'), out=save_dir)

    #####
    # Set up logging
    report_keys = ["epoch", "iteration", "loss_gen", "loss_dis",]

    ## save models
    for m in models:
        trainer.extend(extensions.snapshot_object(m, 'params/'+ m.__class__.__name__ + '_{.updater.epoch}.npz'),
                       trigger=(5, 'epoch'))
    ## save log to file
    trainer.extend(extensions.LogReport(keys=report_keys,
                                        log_name="train.log",
                                        trigger=(1, 'iteration')))
    ## print log
    trainer.extend(extensions.PrintReport(report_keys), trigger=(1, 'iteration'))

    ## generate sample voxels
    trainer.extend(save_sample_voxels(generator, save_dir+'voxels/'),
                   trigger=(1, 'epoch'),
                   priority=extension.PRIORITY_WRITER)

    ## save generated voxels as 2D images
    trainer.extend(save_sample_images(generator, save_dir+'voxels/', save_dir+'images/'), 
                   trigger=(1, 'epoch'),
                   priority=extension.PRIORITY_WRITER)

    ## post slack
    # trainer.extend(post_samples_to_slack(generator, save_dir+'images/'),
    #                trigger=(1, 'epoch'),
    #                priority=extension.PRIORITY_WRITER)
    #####
    ## Run the training :)
    trainer.run()

if __name__ == '__main__':
    main()
