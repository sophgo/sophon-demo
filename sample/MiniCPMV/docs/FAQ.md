# FAQ
## 手动编译decord
soc环境下无法直接安装decord，需要人工编译安装，可以通过以下编译方法进行安装

```bash
sudo apt update
sudo apt install -y libavcodec-dev libavformat-dev libavutil-dev libavdevice-dev libavfilter-dev libswresample-dev libswscale-dev ffmpeg
cd ~
git clone --recursive https://github.com/dmlc/decord.git
cd decord
mkdir build
cd build
cmake .. -DUSE_CUDA=OFF -DUSE_OPENCL=OFF
make -j$(nproc)
cd ../python
python3.10 setup.py install --user
```

我们也有编译好的decord，您也可以直接下载，然后直接安装
```
cd ~
python3 -m dfss --url=open@sophgo.com:sophon-demo/MiniCPMV/decord.zip
unzip decord.zip -d .
rm decord.zip
cd decord
rm -rf build
mkdir build
cd build
cmake .. -DUSE_CUDA=OFF -DUSE_OPENCL=OFF
make -j$(nproc)
cd ../python
python3.10 setup.py install --user
```


## SE7安装python3.10
之前的方法无法实现了，必须手动编译
```bash
sudo apt install -y build-essential checkinstall zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev curl
wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz
tar -xvf Python-3.10.0.tgz
cd Python-3.10.0
./configure --enable-optimizations
sudo make altinstall

cd /data
# 创建名为myenv的虚拟环境
python3.10 -m venv myenv

# 进入虚拟环境
source myenv/bin/activate

```