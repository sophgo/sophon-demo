# Environment Guide
Sophon Demo所依赖的环境主要包括用于编译和量化模型的tpu-nntc环境，用于编译C++程序的开发环境以及用于部署程序的运行环境。

## 1 tpu-nntc环境搭建
编译和量化模型需要在tpu-nntc环境中完成。通常需要在x86主机上安装tpu-nntc环境，x86主机已安装Ubuntu16.04/18.04/20.04系统，并且运行内存在12GB以上。tpu-nntc环境安装步骤主要包括：

1. 安装Docker

   若已安装docker，请跳过本节。
    ```bash
    # 安装docker
    sudo apt-get install docker.io
    # docker命令免root权限执行
    # 创建docker用户组，若已有docker组会报错，没关系可忽略
    sudo groupadd docker
    # 将当前用户加入docker组
    sudo gpasswd -a ${USER} docker
    # 切换当前会话到新group或重新登录重启X会话
    newgrp docker​ 
    ```
    > **提示**：需要logout系统然后重新登录，再使用docker就不需要sudo了。

2. 下载并解压tpu-nntc

    从[算能官网](https://developer.sophgo.com/site/index/material/all/all.html)上下载tpu-nntc的压缩包，命名如tpu-nntc_vx.y.z-hash-date.tar.gz，x.y.z表示版本号，并进行解压。
    ```bash
    mkdir tpu-nntc
    # 将压缩包解压到tpu-nntc
    tar zxvf tpu-nntc_vx.y.z-<hash>-<date>.tar.gz --strip-components=1 -C tpu-nntc
    ```

3. 创建并进入docker

    tpu-nntc使用的docker是sophgo/tpuc_dev:latest, docker镜像和tpu-nntc有绑定关系，少数情况下有可能更新了tpu-nntc，需要新的镜像。
    ```bash
    cd tpu-nntc
    # 进入docker，如果当前系统没有对应镜像，会自动从docker hub上下载
    # 将tpu-nntc的上一级目录映射到docker内的/workspace目录
    # 这里用了8001到8001端口映射，之后在使用ufw可视化工具会用到
    # 如果端口已经占用，请更换其他未占用端口，后面根据需要更换进行调整
    docker run -v $PWD/..:/workspace -p 8001:8001 -it sophgo/tpuc_dev:latest
    # 此时已经进入docker，并在/workspace目录下
    # 下面初始化软件环境
    cd /workspace/tpu-nntc
    source scripts/envsetup.sh
    ```
此镜像仅用于编译和量化模型，程序编译和运行请在开发和运行环境中进行。更多tpu-nntc信息请参考《TPU-NNTC快速入门指南.pdf》和《TPU-NNTC开发参考手册.pdf》。

## 2 x86 PCIe平台的开发和运行环境搭建
如果您在x86平台安装了PCIe加速卡，开发环境与运行环境可以是统一的，您可以直接在宿主机上搭建开发和运行环境。

### 2.1 安装libsophon
从[算能官网](https://developer.sophgo.com/site/index/material/all/all.html)上下载libsophon安装包，包括:
* sophon-driver_x.y.z_amd64.deb
* sophon-libsophon_x.y.z_amd64.deb
* sophon-libsophon-dev_x.y.z_amd64.deb

其中：x.y.z表示版本号；sophon-driver包含了PCIe加速卡驱动；sophon-libsophon包含了运行时环境（库文件、工具等）；sophon-libsophon-dev包含了开发环境（头文件等）。如果只是在部署环境上安装，则不需要安装 sophon-libsophon-dev。
```bash
# 安装依赖库，只需要执行一次
sudo apt install dkms libncurses5
# 安装libsophon
sudo dpkg -i sophon-*amd64.deb
# 在终端执行如下命令，或者登出再登入当前用户后即可使用bm-smi等命令：
source /etc/profile
```

更多libsophon信息请参考《LIBSOPHON使用手册.pdf》。

### 2.2 安装sophon-ffmpeg和sophon-opencv
从[算能官网](https://developer.sophgo.com/site/index/material/all/all.html)上下载sophon-mw安装包，包括:
* sophon-mw-sophon-ffmpeg_x.y.z_amd64.deb
* sophon-mw-sophon-ffmpeg-dev_x.y.z_amd64.deb
* sophon-mw-sophon-opencv_x.y.z_amd64.deb
* sophon-mw-sophon-opencv-dev_x.y.z_amd64.deb

其中：x.y.z表示版本号；sophon-ffmpeg/sophon-opencv包含了ffmpeg/opencv运行时环境（库文件、工具等）；sophon-ffmpeg-dev/sophon-opencv-dev包含了开发环境（头文件、pkgconfig、cmake等）。如果只是在部署环境上安装，则不需要安装 sophon-ffmpeg-dev/sophon-opencv-dev。

sophon-mw-sophon-ffmpeg依赖sophon-libsophon包，而sophon-mw-sophon-opencv依赖sophon-mw-sophon-ffmpeg，因此在安装次序上必须
先安装libsophon, 然后sophon-mw-sophon-ffmpeg, 最后安装sophon-mw-sophon-opencv。

如果运行环境中使用的libstdc++库使用GCC5.1之前的旧版本ABI接口（典型的有CENTOS系统），请使用sophon-mw-sophon-opencv-abi0相关安装包。

```bash
# 安装sophon-ffmpeg
sudo dpkg -i sophon-mw-sophon-ffmpeg_*amd64.deb sophon-mw-sophon-ffmpeg-dev_*amd64.deb
# 安装sophon-opencv
sudo dpkg -i sophon-mw-sophon-opencv_*amd64.deb sophon-mw-sophon-opencv-dev_*amd64.deb
# 在终端执行如下命令，或者logout再login当前用户后即可使用安装的工具
source /etc/profile
```

更多sophon-mw信息请参考《MULTIMEDIA使用手册.pdf》、《MULTIMEDIA开发参考手册.pdf》。

### 2.3 编译安装sophon-sail
如果例程依赖sophon-sail则需要编译和安装sophon-sail，否则可跳过本章节。需从[算能官网](https://developer.sophgo.com/site/index/material/all/all.html)上下载sophon-sail的压缩包，命名如sophon-sail_x.y.z.tar.gz，x.y.z表示版本号。
1. 解压并进入目录

    ```bash
    tar zxvf sophon-sail_x.y.z.tar.gz
    cd sophon-sail
    ```              

2. 创建编译文件夹build,并进入build文件夹

    ```bash
    mkdir build && cd build 
    ```                  

3. 执行编译命令

    ```bash
    cmake ..                                   
    make                                      
    ```

4. 安装SAIL动态库及头文件,编译结果将安装在`/opt/sophon`下面

    ```bash
    sudo make install                               
    ```


5. 打包生成python whell,生成的whell包的路径为`python/pcie/dist`

    ```bash
    cd ../python/pcie 
    chmod +x sophon_pcie_whl.sh
    ./sophon_pcie_whl.sh  
    ```

6. 安装python whell  

    ```bash
    # 需根据实际生成的whell包修改其文件名
    pip3 install ./dist/sophon-master-py3-none-any.whl --force-reinstall 
    ```

## 3 SoC平台的开发和运行环境搭建
对于SoC平台，内部已经集成了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包，位于`/opt/sophon/`下，可直接用于运行环境。通常在x86主机上交叉编译程序，使之能够在SoC平台运行。

### 3.1 交叉编译环境搭建
需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中。
1. 安装交叉编译工具链
    ```bash
    sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
    ```

2. 打包libsophon

    从[算能官网](https://developer.sophgo.com/site/index/material/all/all.html)上下载sophon-img安装包，其中包括libsophon_soc_x.y.z_aarch64.tar.gz，x.y.z表示版本号，并进行解压。

    ```bash
    # 创建依赖文件的根目录
    mkdir -p soc-sdk
    # 解压libsophon_soc_x.y.z_aarch64.tar.gz
    tar -zxf libsophon_soc_${x.y.z}_aarch64.tar.gz
    # 将相关的库目录和头文件目录拷贝到依赖文件根目录下
    cp -rf libsophon_soc_${x.y.z}_aarch64/opt/sophon/libsophon-${x.y.z}/lib ${soc-sdk}
    cp -rf libsophon_soc_${x.y.z}_aarch64/opt/sophon/libsophon-${x.y.z}/include ${soc-sdk}
    ```

3. 打包sophon-ffmpeg和sophon-opencv

    从[算能官网](https://developer.sophgo.com/site/index/material/all/all.html)上下载sophon-mw安装包，其中包括sophon-mw-soc_x.y.z_aarch64.tar.gz，x.y.z表示版本号，并进行解压。
    ```bash
    # 解压sophon-mw-soc_x.y.z_aarch64.tar.gz
    tar -zxf sophon-mw-soc_${x.y.z}_aarch64.tar.gz
    # 将ffmpeg和opencv的库目录和头文件目录拷贝到soc-sdk目录下
    cp -rf sophon-mw-soc_${x.y.z}_aarch64/opt/sophon/sophon-ffmpeg_${x.y.z}/lib ${soc-sdk}
    cp -rf sophon-mw-soc_${x.y.z}_aarch64/opt/sophon/sophon-ffmpeg_${x.y.z}/include ${soc-sdk}
    cp -rf sophon-mw-soc_${x.y.z}_aarch64/opt/sophon/sophon-opencv_${x.y.z}/lib ${soc-sdk}
    cp -rf sophon-mw-soc_${x.y.z}_aarch64/opt/sophon/sophon-opencv_${x.y.z}/include ${soc-sdk}
    ```

这里，交叉编译环境已经搭建完成，接下来可以使用打包好的soc-sdk编译需要在SoC平台上运行的程序。更多交叉编译信息请参考《LIBSOPHON使用手册.pdf》。

### 3.2 交叉编译安装sophon-sail
如果例程依赖sophon-sail则需要编译和安装sophon-sail，否则可跳过本章节。需要在x86主机上交叉编译sophon-sail，并在SoC平台上安装。

需从[算能官网](https://developer.sophgo.com/site/index/material/all/all.html)上下载sophon-sail的压缩包，命名如sophon-sail_x.y.z.tar.gz，x.y.z表示版本号。
1. 解压并进入目录

    ```bash
    tar zxvf sophon-sail_x.y.z.tar.gz
    cd sophon-sail
    ```              

2. 创建编译文件夹build,并进入build文件夹

    ```bash
    mkdir build && cd build 
    ```                  

3. 执行编译命令

    使用指定的python3,通过交叉编译的方式,编译出包含bmcv,sophon-ffmpeg,sophon-opencv的SAIL,python3的安装方式可通过python官方文档获取,也可以从[此链接](http://219.142.246.77:65000/sharing/8MlSKnV8x)下载已经编译好的python3,本示例使用的python3路径为`python_3.8.2/bin/python3`,python3的动态库目录`python_3.8.2/lib`。

    请参考[3.1 交叉编译环境搭建](#31-交叉编译环境搭建)获取libsophon、sophon-ffmpeg和sophon-opencv的交叉编译库包。

    ```bash
    # 请根据实际情况修改DPYTHON_EXECUTABLE、DCUSTOM_PY_LIBDIR、DLIBSOPHON_BASIC_PATH、DOPENCV_BASIC_PATH的路径
     cmake -DBUILD_TYPE=soc  \
        -DCMAKE_TOOLCHAIN_FILE=../cmake/BM168x_SOC/ToolChain_aarch64_linux.cmake \
        -DPYTHON_EXECUTABLE=python_3.8.2/bin/python3 \
        -DCUSTOM_PY_LIBDIR=python_3.8.2/lib \
        -DLIBSOPHON_BASIC_PATH=libsophon_soc_${x.y.z}_aarch64/opt/sophon/libsophon-${x.y.z} \
        -DFFMPEG_BASIC_PATH=sophon-mw-soc_${x.y.z}_aarch64/opt/sophon/sophon-ffmpeg_${x.y.z} \
        -DOPENCV_BASIC_PATH=sophon-mw-soc_${x.y.z}_aarch64/opt/sophon/sophon-opencv_${x.y.z} \
        ..                                
    make                                      
    ```
    编译参数：

    * BUILD_TYPE : 编译的类型,目前有pcie和soc两种模式,pcie是编译在x86主机上可用的SAIL包,soc表示使用交叉编译的方式,在x86主机上编译soc上可用的SAIL包。默认pcie。
    
    * ONLY_RUNTIME : 编译结果是否只包含运行时,而不包含bmcv,sophon-ffmpeg,sophon-opencv,如果此编译选项为`ON`,这SAIL的编解码及Bmcv接口不可用,只有推理接口可用。默认`OFF`。
    
    * INSTALL_PREFIX : 执行make install时的安装路径,pcie模式下默认`/opt/sophon`,与libsophon的安装路径一致,交叉编译模式下默认`build_soc`。
    
    * PYTHON_EXECUTABLE : 编译使用的python3的路径名称(路径+名称),默认使用当前系统中默认的python3。
    
    * CUSTOM_PY_LIBDIR : 编译使用的python3的动态库的路径(只包含路径),默认使用当前系统中默认python3的动态库目录。
    
    * LIBSOPHON_BASIC_PATH : 交叉编译模式下,libsophon的路径,如果配置不正确则会编译失败。pcie模式下面此编译选项不生效。
    
    * FFMPEG_BASIC_PATH : 交叉编译模式下,sophon-ffmpeg的路径,如果配置不正确,且ONLY_RUNTIME为`ON`时会编译失败。pcie模式下面此编译选项不生效。
    
    * OPENCV_BASIC_PATH : 交叉编译模式下,sophon-opencv的路径,如果配置不正确,且ONLY_RUNTIME为`ON`时会编译失败。pcie模式下面此编译选项不生效。


4. 安装SAIL动态库及头文件,编译结果将安装在`build_soc`下面

    ```bash
    sudo make install                               
    ```
    将`build_soc`文件夹拷贝至目标SOC上,即可直接引用其头文件调用动态库

5. 打包生成python whell,生成的whell包的路径为`python/soc/dist`

    ```bash
    cd ../python/soc 
    chmod +x sophon_soc_whl.sh
    ./sophon_soc_whl.sh  
    ```

5. 安装python whell  

    将生成的whell包拷贝到目标SOC上,然后执行如下安装命令
    ```bash
    # 需根据实际生成的whell包修改其文件名
    pip3 install sophon_arm-master_-py3-none-any.whl --force-reinstall 
    ```