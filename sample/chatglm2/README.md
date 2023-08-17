# ChatGLM2

## 目录
* [1. 简介](#1-简介)

* [2. 准备模型](#2-准备模型)

* [3. 程序编译和运行](#3-程序编译和运行)

  ​    [3.1-cpp](#3.1-cpp)

  ​    [3.2-python](#3.2-python)

  ​    [3.3-python_web](#3.3-python_web)


## 1. 简介
ChatGLM2-6B 是开源中英双语对话模型 ChatGLM-6B 的第二代版本,相比于初代模型，具有更强大的性能，更长的上下文，更高的推理性能和更开放的协议，ChatGLM2-6B 权重对学术研究完全开放。

该例程可以在下x86上正常运行，如果要在盒子上运行程序，除了编译外还需要修改内存，也可以使用我么们提供的刷机包，里面内置chatglm2_soc版本的程序，刷机包地址如下：
```
http://219.142.246.77:65000/sharing/Gf5Nvrv0D
```

## 2. 准备模型
该模型目前只支持在1684X上运行，由于原始模型巨大，此示例暂不提供模型编译脚本，直接提供编译好的bmodel。如果需要进行模型编译，参考[sophgo
/ChatGLM2-TPU](https://github.com/sophgo/ChatGLM2-TPU)


​本例程在`scripts`目录下提供了相关模型载脚本`download.sh`

```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

执行程序后，当前目录下的文件如下：

```
.
├── cpp                                  #cpp版本
│   ├── chatglm2.hpp                     #chatglm2推理base 
│   ├── CMakeLists.txt
│   ├── lib_pcie                         #pcie依赖的libsentencepiece.a
│   ├── lib_soc                          #soc依赖的libsentencepiece.a
│   ├── main.cpp                         #主程序
│   └── sentencepiece                    #sentencepiece头文件
├── models
│   └── BM1684X                          #bmodel、token
├── python
│   ├── chatglm2.py                      #主程序
│   ├── CMakeLists.txt
│   └── pybind.cpp                       #绑定chatglm2推理base 
├── python_web
│   ├── CMakeLists.txt
│   ├── pybind.cpp                       #绑定chatglm2推理base 
│   └── web_chatglm2.py                  #主程序
├── README.md                            #使用说明
└── script
    └── download.sh                      #模型下载脚本
```

## 2. 程序编译和运行

本例程一共分为三个版本，分别是cpp、python以及web版本，具体的编译和运行方法如下。为了提高运行效率，python和web版本都是通过调用cpp接口实现的。因此，每个版本都需要编译。这里包含三种编译方式，分别是x86下编译pcie运行版本，x86下交叉编译soc运行版本，以及在se7上编译运行版本。
在x86上进行交叉编译，需要先安装如下库：
```
apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

## 2.1 cpp
#### 2.1.1 x86下编译pcie运行版本
```
cd cpp
mkdir build
cd build
cmake ..
make
cd ..
```
执行程序：
```
./chatglm2.pcie
```

#### 2.1.2 x86下编译soc运行版本
/path/sdk-soc表示实际的sdk-soc完整路径

```
cd cpp
mkdir build
cd build
cmake -DTARGET_ARCH=soc -DSDK=/path/sdk-soc ..
make
cd ..
```
执行程序：
```
./chatglm2.soc
```

#### 2.1.3 se7下编译运行版本

```
cd cpp
mkdir build
cd build
cmake -DTARGET_ARCH=soc_base ..
make
cd ..
```
执行程序：
```
./chatglm2.soc
```

### 2.2 python
python版本需要pybind11，需要安装pybind11，并检查一下CMakelist的pybind11路径是否正确，
```
安装pybind11
sudo pip3 install pybind11
查找当前pybind11路径，并添加到CMakelist
sudo find / -name pybind11
```

#### 2.2.1 x86下编译pcie运行版本
```
cd python
mkdir build
cd build
cmake ..
make
cd ../..
```

#### 2.2.2 x86下编译soc运行版本

```
cd python
mkdir build
cd build
cmake -DTARGET_ARCH=soc -DSDK=/path/sdk-soc ..
make
cd ../..
```

#### 2.2.3 se7下编译运行版本

```
cd python
mkdir build
cd build
cmake -DTARGET_ARCH=soc_base ..
make
cd ../..
```
#### 2.2.4 运行程序
```
python3 python/chatglm2.py
```


### 2.3 python_web

web版本需要安装一些依赖：
```
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ gradio --upgrade
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ numpy==1.20.3
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ mdtex2html
```

#### 2.3.1 x86下编译pcie运行版本
```
cd python_web
mkdir build
cd build
cmake ..
make
cd ../..
```

#### 2.3.2 x86下编译soc运行版本

```
cd python_web
mkdir build
cd build
cmake -DTARGET_ARCH=soc -DSDK=/path/sdk-soc ..
make
cd ../..
```

#### 2.3.3 se7下编译运行版本

```
cd python_web
mkdir build
cd build
cmake -DTARGET_ARCH=soc_base ..
make
cd ../..
```
#### 2.3.4 运行程序
```
python3 python_web/web_chatglm2.py
```
模型运行后会自动生成一个链接，复制链接会进入到一个web对话框，即可开始对话。
