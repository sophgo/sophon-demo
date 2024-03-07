#!/bin/bash
# set -ex
shell_dir=$(dirname $(readlink -f "$0"))

DEMO_BASIC_PATH=$shell_dir/..
SOC_SDK_PATH=$DEMO_BASIC_PATH/soc-sdk-allin

function judge_ret() {
  if [[ $1 == 0 ]]; then
    echo -e "\033[32m Passed: $2 \033[0m"
  else
    echo -e "\033[31m Failed: $2 \033[0m"
    exit 1
  fi
  sleep 1
}

function apt_install()
{
    apt-get update -y
    apt-get install -y libboost-dev
    apt-get install -y libeigen3-dev
    apt-get install -y liblapack-dev
    apt-get install -y libblas-dev 
    apt-get install -y libopenblas-dev 
    apt-get install -y libarmadillo-dev 
    apt-get install -y libsndfile1-dev
    apt-get install -y libyaml-cpp-dev 
    apt-get install -y libyaml-cpp0.6
}

function download_soc_sdk_allin()
{
    sdk_version=$1
    pushd $DEMO_BASIC_PATH/
    python3 -m dfss --url=open@sophgo.com:/soc-sdk-allin/$sdk_version/soc-sdk-allin.tgz
    judge_ret $? "download $sdk_version/soc-sdk-allin.tgz"
    if [ -f $DEMO_BASIC_PATH/soc-sdk-allin ]; then
        rm -rf $DEMO_BASIC_PATH/soc-sdk-allin
    fi
    tar -xzvf soc-sdk-allin.tgz
    judge_ret $? "tar -xzvf soc-sdk-allin.tgz"
    popd
}

function build_soc_YOLOv5(){
 
    pushd $DEMO_BASIC_PATH/sample/YOLOv5/cpp/yolov5_sail
    if [ -d build ]; then
        rm -rf build
    fi
    mkdir build && cd build
    cmake .. -DTARGET_ARCH=soc -DSDK=$1 -DSAIL_PATH=$1/sophon-sail && make -j
    judge_ret $? "build soc yolov5_sail"
    cd ..
    rm -rf build
    popd

    pushd $DEMO_BASIC_PATH/sample/YOLOv5/cpp/yolov5_bmcv
    if [ -d build ]; then
        rm -rf build
    fi
    mkdir build && cd build
    cmake .. -DTARGET_ARCH=soc -DSDK=$1 -DSAIL_PATH=$1/sophon-sail && make -j
    judge_ret $? "build soc yolov5_bmcv"
    cd ..
    rm -rf build
    popd
}

function build_soc_BERT(){
 
    pushd $DEMO_BASIC_PATH/sample/BERT/cpp/bert_sail
    if [ -d build ]; then
        rm -rf build
    fi
    mkdir build && cd build
    cmake .. -DTARGET_ARCH=soc -DSDK=$1 -DSAIL_PATH=$1/sophon-sail && make -j
    judge_ret $? "build soc bert_sail"
    cd ..
    rm -rf build
    popd
}

function build_soc_C3D(){
 
    pushd $DEMO_BASIC_PATH/sample/C3D/cpp/c3d_bmcv
    if [ -d build ]; then
        rm -rf build
    fi
    mkdir build && cd build
    cmake .. -DTARGET_ARCH=soc -DSDK=$1 -DSAIL_PATH=$1/sophon-sail && make -j
    judge_ret $? "build soc c3d_bmcv"
    cd ..
    rm -rf build
    popd

    pushd $DEMO_BASIC_PATH/sample/C3D/cpp/c3d_opencv
    if [ -d build ]; then
        rm -rf build
    fi
    mkdir build && cd build
    cmake .. -DTARGET_ARCH=soc -DSDK=$1 -DSAIL_PATH=$1/sophon-sail && make -j
    judge_ret $? "build soc c3d_opencv"
    cd ..
    rm -rf build
    popd
}

function build_soc_DeepSORT(){
 
    pushd $DEMO_BASIC_PATH/sample/DeepSORT/cpp/deepsort_bmcv
    if [ -d build ]; then
        rm -rf build
    fi
    mkdir build && cd build
    cmake .. -DTARGET_ARCH=soc -DSDK=$1 -DSAIL_PATH=$1/sophon-sail && make -j
    judge_ret $? "build soc deepsort_bmcv"
    cd ..
    rm -rf build
    popd
}

function build_soc_LPRNet(){
 
    pushd $DEMO_BASIC_PATH/sample/LPRNet/cpp/lprnet_bmcv
    if [ -d build ]; then
        rm -rf build
    fi
    mkdir build && cd build
    cmake .. -DTARGET_ARCH=soc -DSDK=$1 -DSAIL_PATH=$1/sophon-sail && make -j
    judge_ret $? "build soc lprnet_bmcv"
    cd ..
    rm -rf build
    popd

    pushd $DEMO_BASIC_PATH/sample/LPRNet/cpp/lprnet_opencv
    if [ -d build ]; then
        rm -rf build
    fi
    mkdir build && cd build
    cmake .. -DTARGET_ARCH=soc -DSDK=$1 -DSAIL_PATH=$1/sophon-sail && make -j
    judge_ret $? "build soc lprnet_opencv"
    cd ..
    rm -rf build
    popd
}

function build_soc_P2PNet(){
 
    pushd $DEMO_BASIC_PATH/sample/P2PNet/cpp/p2pnet_bmcv
    if [ -d build ]; then
        rm -rf build
    fi
    mkdir build && cd build
    cmake .. -DTARGET_ARCH=soc -DSDK=$1 -DSAIL_PATH=$1/sophon-sail && make -j
    judge_ret $? "build soc p2pnet_bmcv"
    cd ..
    rm -rf build
    popd
}

function build_soc_CenterNet(){

    
    pushd $DEMO_BASIC_PATH/sample/CenterNet/cpp/centernet_sail
    if [ -d build ]; then
        rm -rf build
    fi
    mkdir build && cd build
    cmake .. -DTARGET_ARCH=soc -DSDK=$1 -DSAIL_PATH=$1/sophon-sail && make -j
    judge_ret $? "build soc centernet_sail"
    cd ..
    rm -rf build
    popd

    pushd $DEMO_BASIC_PATH/sample/CenterNet/cpp/centernet_bmcv
    if [ -d build ]; then
        rm -rf build
    fi
    mkdir build && cd build
    cmake .. -DTARGET_ARCH=soc -DSDK=$1 -DSAIL_PATH=$1/sophon-sail && make -j
    judge_ret $? "build soc centernet_bmcv"
    cd ..
    rm -rf build
    popd
}

function build_soc_OpenPose(){
 
    pushd $DEMO_BASIC_PATH/sample/OpenPose/cpp/openpose_bmcv
    if [ -d build ]; then
        rm -rf build
    fi
    mkdir build && cd build
    cmake .. -DTARGET_ARCH=soc -DSDK=$1 -DSAIL_PATH=$1/sophon-sail && make -j
    judge_ret $? "build soc openpose_bmcv"
    cd ..
    rm -rf build
    popd
}

function build_soc_PPOCR(){
 
    pushd $DEMO_BASIC_PATH/sample/PP-OCR/cpp/ppocr_bmcv
    if [ -d build ]; then
        rm -rf build
    fi
    mkdir build && cd build
    cmake .. -DTARGET_ARCH=soc -DSDK=$1 -DSAIL_PATH=$1/sophon-sail && make -j
    judge_ret $? "build soc ppocr_bmcv"
    cd ..
    rm -rf build
    popd
}

function build_soc_YOLOv5_opt(){
 
    pushd $DEMO_BASIC_PATH/sample/YOLOv5_opt/cpp/yolov5_sail
    if [ -d build ]; then
        rm -rf build
    fi
    mkdir build && cd build
    cmake .. -DTARGET_ARCH=soc -DSDK=$1 -DSAIL_PATH=$1/sophon-sail && make -j
    judge_ret $? "build soc yolov5_opt_sail"
    cd ..
    rm -rf build
    popd

    pushd $DEMO_BASIC_PATH/sample/YOLOv5_opt/cpp/yolov5_bmcv
    if [ -d build ]; then
        rm -rf build
    fi
    mkdir build && cd build
    cmake .. -DTARGET_ARCH=soc -DSDK=$1 -DSAIL_PATH=$1/sophon-sail && make -j
    judge_ret $? "build soc yolov5_opt_bmcv"
    cd ..
    rm -rf build
    popd
}

function build_soc_YOLOv8_det(){
 
    pushd $DEMO_BASIC_PATH/sample/YOLOv8_det/cpp/yolov8_bmcv
    if [ -d build ]; then
        rm -rf build
    fi
    mkdir build && cd build
    cmake .. -DTARGET_ARCH=soc -DSDK=$1 -DSAIL_PATH=$1/sophon-sail && make -j
    judge_ret $? "build soc yolov8_bmcv"
    cd ..
    rm -rf build
    popd
}

pip3_install_package() {
    local package_name="$1"
    
    for i in {1..5}; do
        echo "Attempt $i: Installing $package_name"
        
        if pip3 show $package_name > /dev/null 2>&1; then
            echo "$package_name is already installed."
            return
        fi

        pip3 install $package_name --upgrade

        if [ $? -eq 0 ]; then
            echo "$package_name installed successfully."
            return
        else
            echo "Failed to install $package_name. Retrying in 5 seconds..."
            sleep 5
        fi
    done
}

echo "-------------------------Start Install dfss --------------------------------------"
pip3_install_package dfss
echo "-------------------------Start apt install ---------------------------------------"
apt_install
echo "-------------------------Start Download soc-sdk-allin v23.10.01 ------------------"
download_soc_sdk_allin v23.10.01
echo "-------------------------Start build_soc_YOLOv5 ------------------"
build_soc_YOLOv5 $SOC_SDK_PATH
# echo "-------------------------Start build_soc_BERT ------------------"
# build_soc_BERT $SOC_SDK_PATH
echo "-------------------------Start build_soc_C3D ------------------"
build_soc_C3D $SOC_SDK_PATH
echo "-------------------------Start build_soc_DeepSORT ------------------"
build_soc_DeepSORT $SOC_SDK_PATH
echo "-------------------------Start build_soc_LPRNet ------------------"
build_soc_LPRNet $SOC_SDK_PATH
# echo "-------------------------Start build_soc_P2PNet ------------------"
# build_soc_P2PNet $SOC_SDK_PATH
echo "-------------------------Start build_soc_CenterNet ------------------"
build_soc_CenterNet $SOC_SDK_PATH
echo "-------------------------Start build_soc_OpenPose ------------------"
build_soc_OpenPose $SOC_SDK_PATH
echo "-------------------------Start build_soc_PPOCR ------------------"
build_soc_PPOCR $SOC_SDK_PATH
echo "-------------------------Start build_soc_YOLOv5_opt ------------------"
build_soc_YOLOv5_opt $SOC_SDK_PATH
echo "-------------------------Start build_soc_YOLOv8 ------------------"
build_soc_YOLOv8_det $SOC_SDK_PATH
