#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))
# echo $scripts_dir
pip3 install dfn
pushd $scripts_dir
python3 -m dfn --url http://219.142.246.77:65000/sharing/wsCN1dBbO
python3 -m dfn --url http://219.142.246.77:65000/sharing/fVyXNs9Eb
tar -xvf data*.tar.gz -C ../
rm data*.tar.gz
mv video.mp4 ../data/
echo "All done!"
popd
