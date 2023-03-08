#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))
# echo $scripts_dir
# sudo apt-get install zip
pip3 install dfn
pushd $scripts_dir
python3 -m dfn --url http://219.142.246.77:65000/sharing/WWGkMQaca
unzip data.zip -d ../
rm data.zip
echo "All done!"
popd
