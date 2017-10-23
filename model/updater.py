import numpy as np

import chainer
import chainer.functions as F
from chainer import Variable
from common.data_io import read_h5, read_binvox

from chainer import cuda

class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self.ext = kwargs.pop('extension')

        if self.ext == 'h5':
            self.read_func = read_h5
        elif self.ext == 'binvox':
            self.read_func = read_binvox
        else:
            raise NotImplementedError

        super(Updater, self).__init__(*args, **kwargs)

    def update_core(self):
        gen_optimizer = self.get_optimizer('opt_gen')
        dis_optimizer = self.get_optimizer('opt_dis')

        try:
            xp = cuda.cupy
        except AttributeError:
            xp = np
        
        # batch
        batch_files = self.get_iterator('main').next()
        batchsize = len(batch_files)

        x = np.empty((batchsize, 1, 64, 64, 64), dtype=np.float32)
        for i in range(batchsize):
            x[i,0] = self.read_func(batch_files[i])

        x_real = Variable(xp.asarray(x))

        y_real = self.dis(x_real)

        z = Variable(xp.asarray(self.gen.make_hidden(batchsize)))
        x_fake = self.gen(z)
        y_fake = self.dis(x_fake)

        loss_dis = F.sum(F.softplus(-y_real)) / batchsize
        loss_dis += F.sum(F.softplus(y_fake)) / batchsize

        loss_gen = F.sum(F.softplus(-y_fake)) / batchsize

        self.gen.cleargrads()
        loss_gen.backward()
        gen_optimizer.update()
        x_fake.unchain_backward()

        self.dis.cleargrads()
        loss_dis.backward()
        dis_optimizer.update()

        chainer.reporter.report({'loss_gen': loss_gen})
        chainer.reporter.report({'loss_dis': loss_dis})
