# Cuda configuration
## Windows
[Cuda 11.6 - cuda_11.6.0_511.23_windows](https://developer.nvidia.com/cuda-11-6-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local)

[//]: # (::[Pytorch previous versions]&#40;https://pytorch.org/get-started/previous-versions/#v1121&#41;)
Configure environment variables
![EnvironmentVariables](../fig/cuda_env.jpg?raw=true "UI")



## Python environment
``` bash
conda create -n lambda2 python=3.10 jupyter cudatoolkit pytorch torchvision torchaudio cpuonly -c pytorch  -c conda-forge
:: conda create -n lambda2 python=3.10 jupyter cudatoolkit -c conda-forge
:: conda activate lambda2
:: pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

