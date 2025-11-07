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
python3 setup.py install --user
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
python3 setup.py install --user
```