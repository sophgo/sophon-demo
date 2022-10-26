#!/bin/bash
pip3 install dfn

script_dir=$(dirname $(readlink -f "$0"))
root_dir=$script_dir/..
data_dir=$root_dir/data
if [ ! -d "$data_dir" ]; then
  echo "create data dir: $data_dir"
  mkdir -p $data_dir
fi

pushd $script_dir

# images
python3 -m dfn --url http://219.142.246.77:65000/sharing/r1ENDQWz3
python3 -m dfn --url http://219.142.246.77:65000/sharing/X9kDZRcKJ
python3 -m dfn --url http://219.142.246.77:65000/sharing/0AZ6cglFe
python3 -m dfn --url http://219.142.246.77:65000/sharing/vv7sw4FL5
python3 -m dfn --url http://219.142.246.77:65000/sharing/2LqvHTw2I
python3 -m dfn --url http://219.142.246.77:65000/sharing/QCOAQGj3i
# videos
python3 -m dfn --url http://219.142.246.77:65000/sharing/iLmuKYi1F

image_dir=$data_dir/images
if [ ! -d "$image_dir" ]; then
  echo "create image dir: $image_dir"
  mkdir -p $image_dir
fi

mv $script_dir/000000162415.jpg $image_dir
mv $script_dir/000000250758.jpg $image_dir
mv $script_dir/000000404484.jpg $image_dir
mv $script_dir/000000404568.jpg $image_dir
mv $script_dir/n02412080_66.JPEG $image_dir
mv $script_dir/n07697537_55793.JPEG $image_dir

video_dir=$data_dir/videos
if [ ! -d "$video_dir" ]; then
  echo "create video dir: $video_dir"
  mkdir -p $video_dir
fi
mv $script_dir/road.mp4 $video_dir

popd





