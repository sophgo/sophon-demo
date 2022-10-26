#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))
# echo $scripts_dir

pushd $scripts_dir
python3 -m dfn --url http://219.142.246.77:65000/sharing/SN8uvGhnF
tar -xvf data*.tar.gz -C ../
rm data*.tar.gz
echo "All done!"
popd
