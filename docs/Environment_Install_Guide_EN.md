[简体中文](./Environment_Install_Guide.md) | [English](./Environment_Install_Guide_EN.md)

# sophon-demo Environmental Installation Guide
## Contents
* [sophon-demo Environmental Installation Guide](#sophon-demo-environmental-installation-guide)
  * [Contents](#contents)
  * [1 TPU-MLIR Environmental Installation](#1-tpu-mlir-environmental-installation)
  * [2 TPU-NNTC Environmental Installation](#2-tpu-nntc-environmental-installation)
  * [3 x86 PCIe Platform Development and Runtime Environment Construction](#3-x86-pcie-platform-development-and-runtime-environment-construction)
    * [3.1 Installation of libsophon](#31-installation-of-libsophon)
    * [3.2 Installation of sophon-ffmpeg and sophon-opencv](#32-installation-of-sophon-ffmpeg-and-sophon-opencv)
    * [3.3 Compilation and Installation of sophon-sail](#33-compilation-and-installation-of-sophon-sail)
  * [4 SoC Platform Development and Runtime Environment Construction](#4-soc-platform-development-and-runtime-environment-construction)
    * [4.1 Cross-compiling Environment Construction](#41-cross-compiling-environment-construction)
    * [4.2 Cross-compiling and sophon-sail Installation](#42-cross-compiling-and-sophon-sail-installation)
  * [5 arm PCIe Platform Development and Runtime Environment Construction](#5-arm-pcie-platform-development-and-runtime-environment-construction)
    * [5.1 Installation of libsophon](#51-installation-of-libsophon)
    * [5.2 Installation of sophon-ffmpeg and sophon-opencv](#52-installation-of-sophon-ffmpeg-and-sophon-opencv)
    * [5.3 Compilation and Installation of sophon-sail](#53-compilation-and-installation-of-sophon-sail)

The environments Sophon Demo relies on include the TPU-NNTC and TPU-MLIR environments for compiling and quantifying the models, the development environment for compiling C++ programs, and the runtime environment for deploying the programs.

## 1 TPU-MLIR Environmental Installation

If you are using BM1684X, it is recommended to use TPU-MLIR to compile BModel. Usually, you need to install TPU-MLIR environment on an x86 host with Ubuntu 16.04/18.04/20.04 installed and running memory of 12GB or more. Here are the installation steps for TPU-MLIR environment:

1. Installation of Docker

   If you already have docker installed, skip this section.
    ```bash
    # Install docker
    sudo apt-get install docker.io
    # In docker, commands can be executed without root privileges
    # Create docker user group,if there is already a docker user group, it will raise an error,this error can be ignored
    sudo groupadd docker
    # Add the current user to the docker group
    sudo gpasswd -a ${USER} docker
    # Switch current session to new group or re-login to restart X session
    newgrp docker​ 
    ```
    > **Note**：You need to logout the system and then log back in, and then you can use docker without sudo.

2. Download and Unzip TPU-MLIR

    Download the [Compatible](../README_EN.md#environment-dependencies) TPU-MLIR package from [the official website of Sophgo](https://developer.sophgo.com/site/index/material/28/all.html) and unzip it, the package is named in the format tpu-mlir_vx.y.z-hash-date.tar.gz, where x.y.z indicates the version number.
    ```bash
    tar zxvf tpu-mlir_vx.y.z-<hash>-<date>.tar.gz
    ```

3. Create and Enter Docker

    The docker used by TPU-MLIR is sophgo/tpuc_dev:latest, the docker image and tpu-mlir have a binding relationship, in a few cases it is possible that tpu-mlir is updated and a new image is needed.
    ```bash
    # If the current system does not have the corresponding image, it will automatically download from docker hub
    # Here will map current directory to /workspace directory in docker, users need to map the demo directory to docker according to the actual situation
    # Myname is just an example of a name, please specify it as the name of the container you want
    docker run --name myname -v $PWD/..:/workspace -p 8001:8001 -it sophgo/tpuc_dev:v2.1
    # Now, you are already in docker, and in the /workspace directory
    # Then, initializing the software environment
    cd /workspace/tpu-mlir
    source scripts/envsetup.sh
    ```

This image is only for compiling and quantifying the model, please compile and run the program in the development and runtime environment. For more tutorials on TPU-MLIR, please refer to the "TPU-MLIR Quick Start" and the "TPU-MLIR Technical Reference Manual" on [the official website of Sophgo](https://developer.sophgo.com/site/index/material/31/all.html).

## 2 TPU-NNTC Environmental Installation

If you are using BM1684, it is recommended to use TPU-NNTC to compile BModel. Usually, you need to install TPU-NNTC environment on an x86 host with Ubuntu 16.04/18.04/20.04 installed and running memory of 12GB or more. Here are the installation steps for TPU-NNTC environment:

1. Installation of Docker

   If you already have docker installed, skip this section.
    ```bash
    # Install docker
    sudo apt-get install docker.io
    # In docker, commands can be executed without root privileges
    # Create docker user group,if there is already a docker user group, it will raise an error,this error can be ignored
    sudo groupadd docker
    # Add the current user to the docker group
    sudo gpasswd -a ${USER} docker
    # Switch current session to new group or re-login to restart X session
    newgrp docker​ 
    ```
    > **Note**：You need to logout the system and then log back in, and then you can use docker without sudo.

2. Download and Unzip TPU-NNTC

    Download the [Compatible](../README_EN.md#environment-dependencies) TPU-NNTC package from [the official website of Sophgo](https://developer.sophgo.com/site/index/material/28/all.html) and unzip it, the package is named in the format tpu-nntc_vx.y.z-hash-date.tar.gz, where x.y.z indicates the version number.
    ```bash
    mkdir tpu-nntc
    # Unzip the package to tpu-nntc
    tar zxvf tpu-nntc_vx.y.z-<hash>-<date>.tar.gz --strip-components=1 -C tpu-nntc
    ```

3. Create and Enter Docker

    The docker used by TPU-NNTC is sophgo/tpuc_dev:latest, the docker image and tpu-nntc have a binding relationship, in a few cases it is possible that tpu-nntc is updated and a new image is needed.
    ```bash
    cd tpu-nntc
    # Enter docker, if the current system does not have the corresponding image, it will automatically download from docker hub
    # Here will map tpu-nntc's higher level directory to /workspace directory in docker, users need to map the demo directory to docker according to the actual situation
    # 8001 to 8001 port mapping is used here, which will be used later in the ufw visualization tool
    # If the port is already occupied, please replace it with another unoccupied port, and adjust it later as needed
    docker run --name myname -v $PWD/..:/workspace -p 8001:8001 -it sophgo/tpuc_dev:v2.1
    # Now, you are already in docker, and in the /workspace directory
    # Then, initializing the software environment
    cd /workspace/tpu-nntc
    source scripts/envsetup.sh
    ```

This image is only for compiling and quantifying the model, please compile and run the program in the development and runtime environment. For more tutorials on TPU-NNTC, please refer to the "TPU-NNTC Quick Start Guide" and the "TPU-NNTC Development Reference Manual" on [the official website of Sophgo](https://developer.sophgo.com/site/index/material/31/all.html).

## 3 x86 PCIe Platform Development and Runtime Environment Construction
If you have installed a PCIe accelerator card on an x86 platform, the development environment and runtime environment can be the same, and you can build the development and runtime environment directly on the host computer.

### 3.1 Installation of libsophon
Download the [Compatible](../README_EN.md#environment-dependencies) libsophon installation package from [the official website of Sophgo](https://developer.sophgo.com/site/index/material/28/all.html), which includes:
* sophon-driver_x.y.z_amd64.deb
* sophon-libsophon_x.y.z_amd64.deb
* sophon-libsophon-dev_x.y.z_amd64.deb

Regarding the above three installation packages: x.y.z indicates the version number; sophon-driver contains the PCIe accelerator card driver; sophon-libsophon contains the runtime environment (library files, tools, etc.); and sophon-libsophon-dev contains the development environment (header files, etc.). If you are only installing on a deployment environment, you do not need to install sophon-libsophon-dev.
```bash
# Install the dependency library, you only need to execute once
sudo apt install dkms libncurses5
# Install libsophon
sudo dpkg -i sophon-*amd64.deb
# You can use a command such as bm-smi after execute the following command in the terminal, or log out and log in to the current user
source /etc/profile
```

For more information on libsophon, please refer to "LIBSOPHON Manual".

### 3.2 Installation of sophon-ffmpeg and sophon-opencv

Download the [Compatible](../README_EN.md#environment-dependencies) sophon-mw installation package from [the official website of Sophgo](https://developer.sophgo.com/site/index/material/28/all.html), which includes:
* sophon-mw-sophon-ffmpeg_x.y.z_amd64.deb
* sophon-mw-sophon-ffmpeg-dev_x.y.z_amd64.deb
* sophon-mw-sophon-opencv_x.y.z_amd64.deb
* sophon-mw-sophon-opencv-dev_x.y.z_amd64.deb

Regarding the above packages: x.y.z indicates the version number; sophon-ffmpeg/sophon-opencv contains the runtime environment of ffmpeg/opencv(library files, tools, etc.); and sophon-ffmpeg-dev/sophon-opencv-dev contains the development environment (header files, pkgconfig, cmake, etc.). If you are only installing on a deployment environment, you do not need to install sophon-ffmpeg-dev/sophon-opencv-dev.

sophon-mw-sophon-ffmpeg depends on the sophon-libsophon package, while sophon-mw-sophon-opencv depends on sophon-mw-sophon-ffmpeg, so you must install libsophon first, then sophon-mw-sophon-ffmpeg, and finally install sophon-mw-sophon-opencv.

If the libstdc++ library used in the runtime environment uses an older version of the ABI interface prior to GCC 5.1 (typical with CENTOS systems), please use the sophon-mw-sophon-opencv-abi0 related installation package.

```bash
# Install sophon-ffmpeg
sudo dpkg -i sophon-mw-sophon-ffmpeg_*amd64.deb sophon-mw-sophon-ffmpeg-dev_*amd64.deb
# Install sophon-opencv
sudo dpkg -i sophon-mw-sophon-opencv_*amd64.deb sophon-mw-sophon-opencv-dev_*amd64.deb
# Execute the following command in the terminal, or log out and log in to the current user
source /etc/profile
```

For more information on libsophon, please refer to "MULTIMEDIA User Manual", "Multimedia Development Reference Manual".

### 3.3 Compilation and Installation of sophon-sail

If the demo depends on sophon-sail, you need to compile and install sophon-sail, otherwise you can skip this section. You need to download the [Compatible](../README_EN.md#environment-dependencies) package of sophon-sail from [the official website of Sophgo](https://developer.sophgo.com/site/index/material/28/all.html), the package is named in the format sophon-sail_x.y.z.tar.gz, where x.y.z indicates the version number.
1. Unzip and enter the directory

    ```bash
    tar zxvf sophon-sail_x.y.z.tar.gz
    cd sophon-sail
    ```              

2. Create the folder "build", and go to the "build" folder

    ```bash
    mkdir build && cd build 
    ```                  

3. Execute the compile command

    ```bash
    cmake ..                                   
    make                                      
    ```

4. Install SAIL dynamic library and header files, the compiled result will be installed under `/opt/sophon`.

    ```bash
    sudo make install                               
    ```


5. Generate python wheel, the path of the generated wheel package is `python/pcie/dist`.

    ```bash
    cd ../python/pcie 
    chmod +x sophon_pcie_whl.sh
    ./sophon_pcie_whl.sh  
    ```

6. Install python wheel  

    ```bash
    # You need to change the file name according to the actual generated wheel package
    pip3 install ./dist/sophon-master-py3-none-any.whl --force-reinstall 
    ```


## 4 SoC Platform Development and Runtime Environment Construction
For SoC platforms, the SophonSDK (>=v22.09.02) has been installed with the corresponding libsophon, sophon-opencv and sophon-ffmpeg runtime library packages integrated inside, located under `/opt/sophon/`, which can be used directly for the runtime environment. Programs are usually cross-compiled on x86 hosts to enable them to run on SoC platforms. SophonSDK firmware refresh methods can be found in the [FAQ document](./FAQ_EN.md#12-how-do-i-use-an-sd-card-to-update-firmware-in-soc-mode).

### 4.1 Cross-compiling Environment Construction
You need to build a cross-compilation environment on an x86 host using SOPHON SDK and package the header and library files that the program depends on into the soc-sdk directory.
1. Install the cross-compilation tool chain
    ```bash
    sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
    ```
    If you encount this error: `/lib/aarch64-linux-gnu/libc.so.6: version 'GLIBC_2.33' not found`。
    You can install gcc with lower version.
    ```bash
    sudo apt remove cpp-*-aarch64-linux-gnu
    sudo apt-get install gcc-7-aarch64-linux-gnu g++-7-aarch64-linux-gnu
    sudo ln -s /usr/bin/aarch64-linux-gnu-gcc-7 /usr/bin/aarch64-linux-gnu-gcc
    sudo ln -s /usr/bin/aarch64-linux-gnu-g++-7 /usr/bin/aarch64-linux-gnu-g++
    ```
2. Package libsophon

    Download and unzip the [Compatible](../README_EN.md#environment-dependencies) sophon-img installation package from [the official website of Sophgo](https://developer.sophgo.com/site/index/material/28/all.html), which includes libsophon_soc_x.y.z_aarch64.tar.gz, where x.y.z indicates the version number.


    ```bash
    # Create the root directory of the dependency files
    mkdir -p soc-sdk
    # Unzip libsophon_soc_x.y.z_aarch64.tar.gz
    tar -zxf libsophon_soc_${x.y.z}_aarch64.tar.gz
    # Copy the relevant library and header directories to the root of the dependency file
    cp -rf libsophon_soc_${x.y.z}_aarch64/opt/sophon/libsophon-${x.y.z}/lib ${soc-sdk}
    cp -rf libsophon_soc_${x.y.z}_aarch64/opt/sophon/libsophon-${x.y.z}/include ${soc-sdk}
    ```

3. Package sophon-ffmpeg and sophon-opencv

    Download and unzip the [Compatible](../README_EN.md#environment-dependencies) sophon-mw installation package from [the official website of Sophgo](https://developer.sophgo.com/site/index/material/28/all.html), which includes libsophon_soc_x.y.z_aarch64.tar.gz, where x.y.z indicates the version number.
    ```bash
    # unzip sophon-mw-soc_x.y.z_aarch64.tar.gz
    tar -zxf sophon-mw-soc_${x.y.z}_aarch64.tar.gz
    # Copy the ffmpeg and opencv library directory and header directory to the soc-sdk directory
    cp -rf sophon-mw-soc_${x.y.z}_aarch64/opt/sophon/sophon-ffmpeg_${x.y.z}/lib ${soc-sdk}
    cp -rf sophon-mw-soc_${x.y.z}_aarch64/opt/sophon/sophon-ffmpeg_${x.y.z}/include ${soc-sdk}
    cp -rf sophon-mw-soc_${x.y.z}_aarch64/opt/sophon/sophon-opencv_${x.y.z}/lib ${soc-sdk}
    cp -rf sophon-mw-soc_${x.y.z}_aarch64/opt/sophon/sophon-opencv_${x.y.z}/include ${soc-sdk}
    ```

Here, the cross-compilation environment has been built, and then you can use the packaged soc-sdk to compile the programs that need to run on the SoC platform. For more information on cross-compilation, please refer to the "LIBSOPHON Manual".

### 4.2 Cross-compiling and sophon-sail Installation

If the demo depends on sophon-sail, sophon-sail needs to be compiled and installed, otherwise this section can be skipped. You need to cross-compile sophon-sail on an x86 host and install it on the SoC platform.

Download the [Compatible](../README_EN.md#environment-dependencies) sophon-sail installation package from [the official website of Sophgo](https://developer.sophgo.com/site/index/material/28/all.html), the package is named in the format sophon-sail_x.y.z.tar.gz, where x.y.z indicates the version number.
1. Unzip and enter the directory

    ```bash
    tar zxvf sophon-sail_x.y.z.tar.gz
    cd sophon-sail
    ```              

2. Create the folder "build", and go to the "build" folder

    ```bash
    mkdir build && cd build 
    ```                  

3. Execute the compile command

    Use the specified python3 to compile SAIL with bmcv, sophon-ffmpeg, sophon-opencv by cross-compiling, the installation of python3 can be obtained from the official python documentation,or you can download the compiled python3 from [this link](http://219.142.246.77:65000/sharing/8MlSKnV8x) to download the compiled python3, the path of python3 used in this example is `python_3.8.2/bin/python3`, and the dynamic library directory of python3 is `python_3.8.2/lib`.

    Please refer to [4.1 Cross-compiling Environment Construction](#41-cross-compiling-environment-construction) for the cross-compilation library packages of libsophon, sophon-ffmpeg and sophon-opencv.

    ```bash
    # Please modify the paths of DPYTHON_EXECUTABLE、DCUSTOM_PY_LIBDIR、DLIBSOPHON_BASIC_PATH、DOPENCV_BASIC_PATH according to the actual situation
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
    Compile parameters:

    * BUILD_TYPE : The type of compilation. Currently, there are two modes: pcie and soc, pcie means to compile SAIL packages available on x86 host, soc means to use cross-compilation to compile SAIL packages which available on soc on x86 host. The default option is pcie.
    
    * ONLY_RUNTIME :  Whether the compilation result contains only runtime, but not bmcv, sophon-ffmpeg, sophon-opencv. if the option is `ON`, this SAIL codec and Bmcv interface is not available, only inference interface is available. The default option is `OFF`.
    
    * INSTALL_PREFIX : The installation path when make install is executed. The default option is `/opt/sophon` in pcie mode, same as libsophon's installation path, and the default option is `build_soc` in cross-compile mode.
    
    * PYTHON_EXECUTABLE : The path name of python3 used for compilation (path+name), the default option is python3 in current system.
    
    * CUSTOM_PY_LIBDIR : The path of the dynamic library of python3 used for compilation (path only), the default option is the default dynamic library directory of python3 on the current system.
    
    * LIBSOPHON_BASIC_PATH : The path of libsophon in cross-compile mode, if not configured correctly and, it will fail to compile. In pcie mode, this compile option does not work.
    
    * FFMPEG_BASIC_PATH : The path of sophon-ffmpeg in cross-compile mode, compile will fail if configured incorrectly and ONLY_RUNTIME is `ON`. In pcie mode, this compile option does not work.
    
    * OPENCV_BASIC_PATH : The path of sophon-opencv in cross-compile mode, compile will fail if configured incorrectly and ONLY_RUNTIME is `ON`. In pcie mode, this compile option does not work.


4. Install SAIL dynamic libraries and header files, the compiled results will be installed under `../build_soc`

    ```bash
    sudo make install                               
    ```
    Copy `sophon-sail` from the `build_soc` folder to the `/opt/sophon` directory of the target SOC machine, and do: 
    ```
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/sophon/sophon-sail/lib' >> ~/.bashrc
    source ~/.bashrc
    ```
    then return to the host for subsequent operations.

5. Generate python wheel, the path of the generated wheel package is `python/soc/dist`.

    ```bash
    cd ../python/soc 
    chmod +x sophon_soc_whl.sh
    ./sophon_soc_whl.sh  
    ```

5. Install python wheel  

    Copy the generated wheel package to the target SOC, and then execute the following installation command
    ```bash
    # You need to modify the file name according to the actual generated wheel package, x.y.z indicates the version number
    pip3 install sophon_arm-${x.y.z}-py3-none-any.whl --force-reinstall 
    ```

## 5 arm PCIe Platform Development and Runtime Environment Construction
If you have installed PCIe accelerator card in arm platform, the development environment and running environment can be unified, you can directly build the development and running environment on the host.
Here we provide the environment installation method for NeoKylin v10 machine, other types of machines please refer to the official development manual for details.
### 5.1 Installation of libsophon
Download the [Compatible](../README_EN.md#environment-dependencies) libsophon installation package from [the official website of Sophgo](https://developer.sophgo.com/site/index/material/28/all.html), the installation package consists of a file where "$arch" is the hardware architecture of the current machine, and the current server's arch can be obtained using the following command.
```
uname -m
```
Usually the corresponding hardware architecture is x86_64 for x86_64 machines and aarch64 for arm64 machines:
```
libsophon_x.y.z_$arch.tar.gz, x.y.z indicates the version number.
```
It can be installed by following the steps below:

**Note: If you have an old version, first refer to the uninstall steps below to uninstall the old version**
```
tar -xzvf libsophon_${x.y.z}_aarch64.tar.gz
sudo cp -r libsophon_${x.y.z}_aarch64/* /
sudo ln -s /opt/sophon/libsophon-${x.y.z} /opt/sophon/libsophon-current
```
Next, set up your driver compilation environment according to the requirements of your Linux distribution, and then do the following operation:
```
sudo ln -s /opt/sophon/driver-${x.y.z}/$bin /lib/firmware/bm1684x_firmware.bin
sudo ln -s /opt/sophon/driver-${x.y.z}/$bin /lib/firmware/bm1684_ddr_firmware.bin
sudo ln -s /opt/sophon/driver-${x.y.z}/$bin /lib/firmware/bm1684_tcm_firmware.bin
cd /opt/sophon/driver-${x.y.z}
```
Here "$bin" is the full name of the bin file with version number, for bm1684x boards, e.g. bm1684x.bin_v3.1.0-9734c1da-220802, for bm1684 boards, e.g. bm1684_ddr.bin_v3.1.1-63a8614d-220906 and bm1684_tcm.bin_v3.1.1-63a8614d-220906.

Then, you can compile the driver (no dependency on dkms here):.
```
sudo make SOC_MODE=0 PLATFORM=asic SYNC_API_INT_MODE=1 \
          TARGET_PROJECT=sg_pcie_device FW_SIMPLE=0 \
          PCIE_MODE_ENABLE_CPU=1
sudo cp ./bmsophon.ko /lib/modules/$(uname -r)/kernel/
sudo depmod
sudo modprobe bmsophon
```
Finally, there are some configuration works:

Add library and paths for executable files
```
sudo cp /opt/sophon/libsophon-current/data/libsophon.conf /etc/ld.so.conf.d/
sudo ldconfig
sudo cp /opt/sophon/libsophon-current/data/libsophon-bin-path.sh /etc/profile.d/
```
You can use a command such as bm-smi after execute the following command in the terminal, or log out and log in to the current user
```
source /etc/profile
```
Add the cmake config file：
```
sudo mkdir -p /usr/lib/cmake/libsophon
sudo cp /opt/sophon/libsophon-current/data/libsophon-config.cmake /usr/lib/cmake/libsophon/
```
For other platform machines, please refer to the[libsophon installation tutorial](https://doc.sophgo.com/sdk-docs/v22.12.01/docs_latest_release/docs/libsophon/guide/html/1_install.html)。
For more information on libsophon, please refer to "LIBSOPHON Manual"

### 5.2 Installation of sophon-ffmpeg and sophon-opencv
Download the [Compatible](../README_EN.md#environment-dependencies) installation package of sophon-mw from [the official website of sophongo](https://developer.sophgo.com/site/index/material/28/all.html),
The installation package consists of one file:
```
sophon-mw_x.y.z_aarch64.tar.gz，x.y.z indicates the version number
```
It can be installed by the following steps:

First, install the libsophon package according to the "LIBSOPHON Manual", and then:
```
tar -xzvf sophon-mw_${x.y.z}_aarch64.tar.gz
sudo cp -r sophon-mw_${x.y.z}_aarch64/* /
sudo ln -s /opt/sophon/sophon-ffmpeg_${x.y.z} /opt/sophon/sophon-ffmpeg-latest
sudo ln -s /opt/sophon/sophon-opencv_${x.y.z} /opt/sophon/sophon-opencv-latest
sudo ln -s /opt/sophon/sophon-sample_${x.y.z} /opt/sophon/sophon-sample-latest
sudo sed -i "s/usr\/local/opt\/sophon\/sophon-ffmpeg-latest/g" /opt/sophon/sophon-ffmpeg-latest/lib/pkgconfig/*.pc
sudo sed -i "s/^prefix=.*$/prefix=\/opt\/sophon\/sophon-opencv-latest/g" /opt/sophon/sophon-opencv-latest/lib/pkgconfig/opencv4.pc
```
Finally, **install the bz2 libc6 libgcc dependency library**（you need to select the corresponding installation package based on the operating system, which will not be detailed here）
Then there are some configuration tasks:

Add library and paths for executable files:
```
sudo cp /opt/sophon/sophon-ffmpeg-latest/data/01_sophon-ffmpeg.conf /etc/ld.so.conf.d/
sudo cp /opt/sophon/sophon-opencv-latest/data/02_sophon-opencv.conf /etc/ld.so.conf.d/
sudo ldconfig
sudo cp /opt/sophon/sophon-ffmpeg-latest/data/sophon-ffmpeg-autoconf.sh /etc/profile.d/
sudo cp /opt/sophon/sophon-opencv-latest/data/sophon-opencv-autoconf.sh /etc/profile.d/
sudo cp /opt/sophon/sophon-sample-latest/data/sophon-sample-autoconf.sh /etc/profile.d/
source /etc/profile
```
For other platform machines, please refer to the[libsophon installation tutorial](https://doc.sophgo.com/sdk-docs/v22.12.01/docs_latest_release/docs/sophon-mw/manual/html/1_install.html).
For more information on sophon-mw, please refer to "MULTIMEDIA User Manual", "Multimedia Development Reference Manual".

### 5.3 Compilation and Installation of sophon-sail
Same installation method as[3.3 Compilation and Installation of sophon-sail](#33-compilation-and-installation-of-sophon-sail).
