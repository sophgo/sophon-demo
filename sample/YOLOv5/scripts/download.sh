#!/bin/bash

root_dir=$(cd `dirname $BASH_SOURCE[0]`/../ && pwd)
mkdir -p $root_dir/build
pushd $root_dir/build

echo "root_dir=$root_dir"
export PYTHONPATH=$root_dir/../yolov5
img_size=${1:-640}
batch_size=${2:-1}
if [ "$img_size" == "1280" ]; then
  python3 $root_dir/script/export_full.py --batch-size $batch_size --weights yolov5s6.pt --img-size 1280
  mv yolov5s6.torchscript.pt $root_dir/data/models/yolov5s.torchscript.$img_size.$batch_size.pt
  echo "model is moved to $root_dir/data/models/yolov5s.torchscript.$img_size.$batch_size.pt"
else
  python3 $root_dir/script/export_full.py --batch-size $batch_size --weights yolov5s.pt --img-size 640
  mv yolov5s.torchscript.pt $root_dir/data/models/yolov5s.torchscript.$img_size.$batch_size.pt
  echo "model is moved to $root_dir/data/models/yolov5s.torchscript.$img_size.$batch_size.pt"
fi

popd
