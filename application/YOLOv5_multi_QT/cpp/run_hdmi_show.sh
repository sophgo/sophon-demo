#!/bin/sh -x
shell_dir=$(dirname $(readlink -f "$0"))
fl2000=$(lsmod | grep fl2000 | awk '{print $1}')

echo $fl2000
if [ "$fl2000" != "fl2000" ]; then
	echo "insmod fl2000"
else
	echo "fl2000 already insmod"
fi

export PATH=$PATH:/opt/bin:/bm_bin
export QTDIR=/usr/lib/aarch64-linux-gnu #qtsdk在系统上的路径
export QT_QPA_FONTDIR=$QTDIR/fonts 
export QT_QPA_PLATFORM_PLUGIN_PATH=$QTDIR/qt5/plugins/ 
export LD_LIBRARY_PATH=$shell_dir/../tools/lib:/opt/lib
export QT_QPA_PLATFORM=linuxfb:fb=/dev/fl2000-0 #framebuffer驱动
export QWS_MOUSE_PROTO=/dev/input/event3
export NO_FRAMEBUFFER=1
pushd $shell_dir
./build/yolov5_app --config=../config/yolov5_app.json
popd
