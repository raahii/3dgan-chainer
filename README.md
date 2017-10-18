# 3dgan-chainer

[![arXiv Tag](https://img.shields.io/badge/arXiv-1610.07584-brightgreen.svg)](https://arxiv.org/abs/1610.07584)


## Chainer implementation of 3D Generative Adversarial Network.

### Requirements

- chainer 2.0.1
- scipy
- scikit-imag

```
pip install scipy scikit-image
```

#### Optional

- If you want to use `h5` extension files as dataset, `h5py` is needed.

  - h5py

  ```
  pip install h5py
  ```

- If you want to plot voxel, [latest matplotlib](https://github.com/matplotlib/matplotlib) is needed. 

  The [3D voxel / volumetric plot](https://matplotlib.org/devdocs/gallery/mplot3d/voxels.html) function is included only in unreleased version of matplotlib as of Oct 19, 2017)

  - matplotlib 2.1.0+323.ge6448bafc

  ```
  pip install git+https://github.com/matplotlib/matplotlib
  ```

### Dataset

I used **ShapeNet-v2** dataset. 

Training script only supporting `.binbox` or `.h5` extension.

Describe your dataset path to `DATASET_PATH` in `train.py`.

#### .binvox

Just use `.binvox` files in ShapeNet-v2. 

#### .h5

Assuming that `.h5` has ``{ 'data': <np.array, shape (64, 64, 64)> }``. If you want to convert `.binvox` into `.h5`, use `binvox_to_h5.py` script.

### Usage

``python train.py``

