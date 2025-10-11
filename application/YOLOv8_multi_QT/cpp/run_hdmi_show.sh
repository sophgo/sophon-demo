#!/bin/bash -x

if [ -e "/dev/fl2000-0" ]; then
    echo "Detected 1684x"
    FB_DEVICE="/dev/fl2000-0"
    NEED_SUDO=1
    HDMI_STATUS_PATH=$(find /sys -name "hdmi_status" 2>/dev/null | head -n 1)
    HDMI_STATUS="$HDMI_STATUS_PATH/status"
else
    echo "Detected 1688/cv186"
    FB_DEVICE="/dev/dri/card0"
    NEED_SUDO=0
    HDMI_STATUS="/sys/class/drm/card0-HDMI-A-1/status"
fi

shell_dir=$(dirname $(readlink -f "$0"))
export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/qt5/plugins
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/aarch64-linux-gnu/qt5/lib
export QT_QPA_FB_DRM=1
export QT_QPA_PLATFORM="linuxfb:fb=$FB_DEVICE"
export SOPHON_QT_FONT_SIZE=15

if [ -n "$HDMI_STATUS" ]; then
    echo "HDMI state: $crtc_state"
    if [[ "$crtc_state" == "0" || "$crtc_state" == "disconnected" ]]; then
        echo "Please connect hdmi if you want to run a gui case."
        echo "Please connect hdmi if you want to run a gui case."
        echo "Please connect hdmi if you want to run a gui case."
        echo "Please connect hdmi if you want to run a gui case."
        echo "Please connect hdmi if you want to run a gui case."
        echo "Please connect hdmi if you want to run a gui case."
        exit
    fi
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
if [ $NEED_SUDO -eq 1 ]; then
    echo "Running with sudo for 1684x"
    sudo -E ./yolov8_bmcv/yolov8_bmcv.soc --config=./yolov8_bmcv/config_yolov8_mluti_qt.json
else
    ./yolov8_bmcv/yolov8_bmcv.soc --config=./yolov8_bmcv/config_yolov8_mluti_qt.json
fi
popd