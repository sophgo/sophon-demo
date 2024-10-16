#!/bin/bash -x
shell_dir=$(dirname $(readlink -f "$0"))
export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/qt5/plugins
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/aarch64-linux-gnu/qt5/lib
export QT_QPA_FB_DRM=1
export QT_QPA_PLATFORM=linuxfb:fb=/dev/dri/card0
export SOPHON_QT_FONT_SIZE=15

crtc_state=$(cat /sys/class/drm/card0-HDMI-A-1/status)
echo "HDMI state: $crtc_state"
if [ "$crtc_state" == "disconnected" ]; then
    echo "Please connect hdmi if you want to run a gui case."
    echo "Please connect hdmi if you want to run a gui case."
    echo "Please connect hdmi if you want to run a gui case."
    echo "Please connect hdmi if you want to run a gui case."
    echo "Please connect hdmi if you want to run a gui case."
    echo "Please connect hdmi if you want to run a gui case."
    exit
fi
hdmiservice=$(systemctl list-units --type=service --state=running | grep SophonHDMI.service)
echo $hdmiservice
if [ "$hdmiservice" != "" ]; then
    sudo systemctl disable SophonHDMI.service
    sudo systemctl stop SophonHDMI.service
    echo "SophonHDMI.service has been disabled and stoped."
else
    echo "SophonHDMI.service does not exist."
fi

pushd $shell_dir
./yolov5_bmcv/yolov5_bmcv.soc --config=./yolov5_bmcv/config_se9-8.json
popd
