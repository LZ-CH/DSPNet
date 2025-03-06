## Installation

This code is based on [EmbodiedScan](https://github.com/OpenRobotLab/EmbodiedScan) and [mmdetection3d](https://github.com/open-mmlab/mmdetection3d).

We test our codes under the following environment:

Ubuntu 20.04
NVIDIA Driver: 550.54.15
CUDA 11.8
Python 3.10.14
PyTorch 2.2.0+cu118
PyTorch3D 0.7.7

- Create an environment.
    ```shell
    conda create -n dspnet python=3.10 -y
    conda activate dspnet
    ```
- Install PyTorch:
    ```shell
    conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

- Install embodiedqa:
    ```shell
    python install.py all 
    ```
Note: The automatic installation script make each step a subprocess and the related messages are only printed when the subprocess is finished or killed. Therefore, it is normal to seemingly hang when installing heavier packages, such as PyTorch3D.

BTW, from our experience, it is easier to encounter problems when installing these package. Feel free to post your questions or suggestions during the installation procedure.