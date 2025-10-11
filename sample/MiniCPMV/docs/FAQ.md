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
sudo apt-get purge python3.8 python3.8-*
sudo apt install -y build-essential checkinstall zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev curl
wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz
tar -xvf Python-3.10.0.tgz
cd Python-3.10.0
./configure --enable-optimizations
sudo make altinstall
python3
python
python3.10
python3.10 --version
ls -l /usr/bin/python3
which python3.10
sudo update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 2
sudo update-alternatives --config python3
# 1️⃣ 注册 python3 指向 python3.10
sudo update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 2
# 2️⃣ 注册 python 指向 python3.10
sudo update-alternatives --install /usr/bin/python python /usr/local/bin/python3.10 2
# 3️⃣ 选择默认版本（出现列表后选 Python 3.10）
sudo update-alternatives --config python3
sudo update-alternatives --config python
python3
sudo ln -sf /usr/local/bin/pip3.10 /usr/bin/pip3
sudo ln -sf /usr/local/bin/pip3.10 /usr/bin/pip
pip3 --version
pip --version

```