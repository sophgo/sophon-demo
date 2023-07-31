**第一步**：从[官网](https://docs.conda.io/en/latest/miniconda.html)下载并安装 Miniconda

**第二步**：创建并激活 conda 环境

```bash
conda create --name segformer python=3.8 -y
conda activate segformer
```

```bash
conda install pytorch torchvision==0.8.2 cpuonly -c pytorch
```

**第三步**：安装依赖

```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
pip install -U openmim
mim install 'mmcv-full>=1.1.4,<=1.3.0'
pip install ipython
pip install timm
pip install onnx
pip install -e . --user
```

**第四步**: 模型下载
Download `trained weights`. ( [google drive](https://drive.google.com/drive/folders/1GAku0G0iR9DsBxCbfENWMJ27c5lYUeQA?usp=sharing) | [onedrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/Ept_oetyUGFCsZTKiL_90kUBy5jmPV65O5rJInsnRCDWJQ?e=CvGohw) )


**第五步：** 模型转换示例

```bash
python SegFormer/tools/pytorch2onnx.py --config SegFormer/local_configs/segformer/B0/segformer.b0.512x1024.city.160k.py --checkpoint SegFormer/pth/segformer.b0.512x1024.city.160k.pth --show --verify --output-file SegFormer/onnx/segformer.b0.512x1024.city.160k.onnx
```

在SegFormer/onnx/文件夹下生成：segformer.b0.512x1024.city.160k.onnx