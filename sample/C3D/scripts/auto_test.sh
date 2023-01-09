#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))
if [ ! $1 ]; then
    target="BM1684X"
else
    target=$1
fi

if [ ! $2 ]; then
    tpu_id=0
else
    tpu_id=$2
fi

pushd $scripts_dir
if [ ! -d '../data/UCF_test_01' ]; then
    ./download.sh
fi

cd ../cpp/c3d_opencv/
if [ ! -d 'build' ]; then
    mkdir build
fi
cd build
rm fp32_*b.log int8_*b.log
cmake .. && make
./c3d_opencv.pcie ../../../data/UCF_test_01 ../../../data/models/$target/c3d_fp32_1b.bmodel $tpu_id > fp32_1b.log 2>&1
./c3d_opencv.pcie ../../../data/UCF_test_01 ../../../data/models/$target/c3d_fp32_4b.bmodel $tpu_id > fp32_4b.log 2>&1
./c3d_opencv.pcie ../../../data/UCF_test_01 ../../../data/models/$target/c3d_int8_1b.bmodel $tpu_id > int8_1b.log 2>&1
./c3d_opencv.pcie ../../../data/UCF_test_01 ../../../data/models/$target/c3d_int8_4b.bmodel $tpu_id > int8_4b.log 2>&1

cd ../../../python
rm fp32_*b.log int8_*b.log
python3 c3d_opencv.py --bmodel ../data/models/$target/c3d_fp32_1b.bmodel --tpu_id $tpu_id > fp32_1b.log 2>&1 
python3 c3d_opencv.py --bmodel ../data/models/$target/c3d_fp32_4b.bmodel --tpu_id $tpu_id > fp32_4b.log 2>&1 
python3 c3d_opencv.py --bmodel ../data/models/$target/c3d_int8_1b.bmodel --tpu_id $tpu_id > int8_1b.log 2>&1 
python3 c3d_opencv.py --bmodel ../data/models/$target/c3d_int8_4b.bmodel --tpu_id $tpu_id > int8_4b.log 2>&1 

echo "All done!"
popd
