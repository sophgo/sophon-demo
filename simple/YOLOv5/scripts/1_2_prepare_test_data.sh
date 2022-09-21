#!/bin/bash
pip3 install dfn
script_dir=$(dirname $(readlink -f "$0"))
pushd ${script_dir}
root_dir=$script_dir/..
data_dir=$root_dir/data
if [ ! -d "$data_dir" ]; then
  echo "create data dir: $data_dir"
  mkdir -p $data_dir
fi

python3 -m dfn --url  http://219.142.246.77:65000/sharing/Fol0BuTA0
python3 -m dfn --url  http://219.142.246.77:65000/sharing/mWdrrPbpn

python3 -m dfn --url  http://219.142.246.77:65000/sharing/A7C4AXUfY
popd


pushd $data_dir

image_dir=$data_dir/images
if [ ! -d "$image_dir" ]; then
  echo "create image dir: $image_dir"
  mkdir -p $image_dir
fi
pushd $image_dir
mv $script_dir/dog.jpg ./
mv $script_dir/zidane.jpg ./
popd

video_dir=$data_dir/videos
if [ ! -d "$video_dir" ]; then
  echo "create video dir: $video_dir"
  mkdir -p $video_dir
fi
pushd $video_dir
mv $script_dir/dance.mp4 ./
popd

popd
