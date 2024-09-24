#!/bin/bash -x
shell_dir=$(dirname $(readlink -f "$0"))
sudo systemctl stop SophonHDMI
export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/qt5/plugins
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/aarch64-linux-gnu/qt5/lib
export QT_QPA_FB_DRM=1
export QT_QPA_PLATFORM=linuxfb:fb=/dev/dri/card0
export SOPHON_QT_FONT_SIZE=15
pushd $shell_dir
./yolov5_bmcv/yolov5_bmcv.soc --config=./yolov5_bmcv/config_se9-8.json
popd
