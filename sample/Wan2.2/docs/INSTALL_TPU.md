# Wan2.2测试环境配置

## 目录

* [1. 初始化测试环境](#1-初始化测试环境)
* [2. 下载测试用镜像以及torch_tpu](#2-下载测试用镜像以及torch_tpu)
* [3. 部署镜像并启动容器](#3-部署镜像并启动容器)
* [4. 进入容器并配置容器内测试环境](#4-进入容器并配置容器内测试环境)


## 1. 初始化测试环境

如环境已完成配置，则跳过该步骤

```bash
# 开始进行前确认当前环境为测试环境
# 安装dfss下载工具
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/SC11-FP300/v1.11_RC2/RC2/BSP/Runtime/Release_20251111_201831/PCIe/x86_64/tpuv7-driver_1.7.6_amd64.deb
python3 -m dfss --url=open@sophgo.com:/SC11-FP300/v1.11_RC2/RC2/BSP/Runtime/Release_20251111_201831/PCIe/x86_64/tpuv7-runtime_1.7.6_amd64.deb
python3 -m dfss --url=open@sophgo.com:/SC11-FP300/v1.11_RC2/RC2/BSP/Runtime/Release_20251111_201831/PCIe/x86_64/tpuv7-runtime-dev_1.7.6_amd64.deb

# 下载完成后
sudo dpkg -i tpuv7-*.deb
```

## 2. 下载测试用镜像，Torch_tpu以及部分资源数据

```bash
# 下载测试用镜像，Torch_tpu以及部分数据
cd ./scripts/
./download.sh
```

执行完成后会在测试用镜像与Torch_tpu包会放置在./packages路径下。
部分资源数据会解压至./python路径下。

## 3. 部署镜像并启动容器

```bash
# 加载镜像
bunzip -c docker-soph_vllm-0.7.3-20251114-2e82aebe-350d5894.tar.bz2 | docker load

# 启动容器，需要注意将demo路径与模型路径进行映射，这里使用-v ~/:/workspace以及-v /data:/data作为示例
docker run --privileged -itd --name test --shm-size 1g -v ~/:/workspace -v /dev/:/dev/ -v /data:/data -v /opt/tpuv7:/opt/tpuv7 --entrypoint /bin/bash  soph_vllm:0.7.3
```

## 4. 进入容器并配置容器内测试环境

进入容器

```bash
docker exec -it test bash
```

在容器内配置测试环境

```bash
cd /<path_to_Wan2.2 demo>/
pip3 install -r requirements_tpu.txt
cd /<path_to_torch_tpu>/
tar -zxvf torch-tpu_20251114_350d5894.tar.gz -C torch_tpu
cd torch_tpu/dist
pip3 install torch_tpu-2.1.0.post1-cp311-cp311-linux_x86_64.whl --force-reinstall
```
