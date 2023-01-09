#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))
# echo $scripts_dir
pip3 install dfn
pushd $scripts_dir
python3 -m dfn --url http://219.142.246.77:65000/sharing/7Y6g7eDnS  #c3d bmodels
python3 -m dfn --url http://219.142.246.77:65000/sharing/jTuyJMnHO  #ucf 101 test 01


if [ ! -d '../data' ]; then
    mkdir ../data
fi

mv c3d_models_*.tar.gz ../data/
mv UCF_test_01.tar.gz ../data/
cd ../data/
tar xvf c3d_models_*.tar.gz
rm c3d_models_*.tar.gz  

tar xvf UCF_test_01.tar.gz
rm UCF_test_01.tar.gz

echo "All done!"
popd
