#!/bin/bash
# set -ex
shell_dir=$(dirname $(readlink -f "$0"))

DEMO_BASIC_PATH=$shell_dir/..
SOC_SDK_PATH=$DEMO_BASIC_PATH/soc-sdk-allin
sail_list=("YOLOv5" "CenterNet" "BERT" "ppYOLOv3" "YOLOv34" "YOLOX" "segformer" "ppYoloe" "YOLOv5_opt") #c++ sail demo

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

function test_sample(){
    pushd $DEMO_BASIC_PATH/
    echo "=============="
    echo "NOW: $1"
    echo "=============="
    chmod +x ./sample/$1/scripts/auto_test.sh
    current_time=$(date +%Y-%m-%d_%H:%M:%S)
    if [[ " ${sail_list[*]} " == *" $1 "* ]]; then
      ./sample/$1/scripts/auto_test.sh -m soc_build -s $SOC_SDK_PATH -a $SOC_SDK_PATH/sophon-sail
    else
      ./sample/$1/scripts/auto_test.sh -m soc_build -s $SOC_SDK_PATH
    fi
    rm -r `find -name build`
    echo "=============="
    echo "EXIT: $1"
    echo "=============="
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

#SE5
download_soc_sdk_allin v23.10.01
test_sample BERT
test_sample PP-OCR
test_sample YOLOv7
test_sample YOLOv34
test_sample ByteTrack
test_sample C3D
test_sample CenterNet
test_sample DeepSORT
test_sample LPRNet
test_sample OpenPose
test_sample ResNet
test_sample YOLOv5
test_sample YOLOv5_opt
test_sample YOLOv8_det
rm -r $SOC_SDK_PATH

# SE9
download_soc_sdk_allin v1.5.0
test_sample BERT
test_sample PP-OCR
test_sample YOLOv7
test_sample YOLOv34
test_sample ByteTrack
test_sample C3D
test_sample CenterNet
test_sample DeepSORT
test_sample LPRNet
test_sample OpenPose
test_sample ResNet
test_sample YOLOv5
test_sample YOLOv8_det
rm -r $SOC_SDK_PATH
