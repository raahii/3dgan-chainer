import chainer
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import Variable
from chainer.utils import type_check
from chainer import function
from chainer.initializers import GlorotNormal

import chainer.functions as F
import chainer.links as L
import numpy as np

class Generator(chainer.Chain):
    def __init__(self):
        w = GlorotNormal()
        super(Generator, self).__init__(
            dc1 = L.DeconvolutionND(3, 200, 512, 4, stride=1, initialW=w),
            dc2 = L.DeconvolutionND(3, 512, 256, 4, stride=2, pad=1, initialW=w),
            dc3 = L.DeconvolutionND(3, 256, 128, 4, stride=2, pad=1, initialW=w),
            dc4 = L.DeconvolutionND(3, 128,  64, 4, stride=2, pad=1, initialW=w),
            dc5 = L.DeconvolutionND(3,  64,   1, 4, stride=2, pad=1, initialW=w),

            bn1 = L.BatchNormalization(512),
            bn2 = L.BatchNormalization(256),
            bn3 = L.BatchNormalization(128),
            bn4 = L.BatchNormalization(64),
            bn5 = L.BatchNormalization(1),
        )

    def __call__(self, z):
        y = F.relu(self.bn1(self.dc1(z)))
        y = F.relu(self.bn2(self.dc2(y)))
        y = F.relu(self.bn3(self.dc3(y)))
        y = F.relu(self.bn4(self.dc4(y)))
        y = F.sigmoid(self.bn5(self.dc5(y)))

        return y
    
    def make_hidden(self, batchsize, train=True):
        if train:
            return np.random.normal(0, 0.33, size=[batchsize, 200, 1, 1, 1]).astype(np.float32)
        else:
            return np.random.uniform(-1, 1, (batchsize, 200, 1, 1, 1)).astype(np.float32)

class Discriminator(chainer.Chain):
    def __init__(self):
        w = GlorotNormal()
        super(Discriminator, self).__init__(
            dc1 = L.ConvolutionND(3,   1,  64, 4, stride=2, pad=1, initialW=w),
            dc2 = L.ConvolutionND(3,  64, 128, 4, stride=2, pad=1, initialW=w),
            dc3 = L.ConvolutionND(3, 128, 256, 4, stride=2, pad=1, initialW=w),
            dc4 = L.ConvolutionND(3, 256, 512, 4, stride=2, pad=1, initialW=w),
            dc5 = L.ConvolutionND(3, 512,   1, 4, stride=1, pad=1, initialW=w),

            bn1 = L.BatchNormalization(64),
            bn2 = L.BatchNormalization(128),
            bn3 = L.BatchNormalization(256),
            bn4 = L.BatchNormalization(512),
            bn5 = L.BatchNormalization(1),
        )

    def __call__(self, x):
        y = F.leaky_relu(self.bn1(self.dc1(x)), slope=0.2)
        y = F.leaky_relu(self.bn2(self.dc2(y)), slope=0.2)
        y = F.leaky_relu(self.bn3(self.dc3(y)), slope=0.2)
        y = F.leaky_relu(self.bn4(self.dc4(y)), slope=0.2)
        y = F.sigmoid(self.bn5(self.dc5(y)))

        return y

def count_model_params(m):
    return sum(p.data.size for p in m.params())

def main():
    m = Generator()
    print(count_model_params(m))
    m = Discriminator()
    print(count_model_params(m))

if __name__=="__main__":
    main()
