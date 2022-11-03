function judge_ret() {
  if [[ $1 == 0 ]]; then
    echo "Passed: $2"
    echo ""
  else
    echo "Failed: $2"
    exit 1
  fi
  sleep 3
}

function run_make_pcie() {
  rm -rf ../cpp/CMakeFiles
  rm ../cpp/cmake_install.cmake
  rm ../cpp/CMakeCache.txt
  rm ../cpp/Makefile
  #export TARGET_ARCH="x86"
  cmake ../cpp/CMakeLists.txt -DTARGET_ARCH="x86" -DSAIL_DIR=$1
  make -C ../cpp
}

# function run_make_arm() {
#   rm -rf ../cpp/CMakeFiles
#   rm ../cpp/cmake_install.cmake
#   rm ../cpp/CMakeCache.txt
#   rm ../cpp/Makefile
#   #export TARGET_ARCH="soc"
#   cmake ../cpp/CMakeLists.txt -DTARGET_ARCH="soc" -DSAIL_DIR=$1 -DSDK=$2
#   make -C ../cpp
# }

function build_cpp() {
  pushd ../cpp
  if [[ $1 == "x86" ]]
  then
    OUT_BUILD=$(run_make_pcie $2)
    echo $OUT_BUILD
  else if [[ $1 == "soc" ]]
  then
    #OUT_BUILD=$(run_make_arm $2 $3) 
    else
      OUT_BUILD="failed"
    fi
  fi
  if [[ $OUT_BUILD =~ "failed" ]]
  then
    judge_ret 1 "build_cpp"
  else
    judge_ret 0 "build_cpp"
  fi
  popd 
}

function run_example_cpp() {
  if [ $1 = "x86" ];then
    ../cpp/yolox_sail.pcie pic ../data/image/val2017 ../data/models/$2/yolox_s_fp32_1b.bmodel 1 0.25 0.45 ../cpp/results $3
    judge_ret $? "run_example_cpp [yolox_s_fp32_1b.bmodel]"
    ../cpp/yolox_sail.pcie pic ../data/image/val2017 ../data/models/$2/yolox_s_int8_4b.bmodel 1 0.15 0.45 ../cpp/results $3
    judge_ret $? "run_example_cpp [yolox_s_int8_4b.bmodel]"

  else if [ $1 = "soc" ];then
    ../cpp/yolox_sail.arm pic ../data/image/val2017 ../data/models/$2/yolox_s_fp32_1b.bmodel 1 0.25 0.45 ../cpp/results $3
    judge_ret $? "run_example_cpp [yolox_s_fp32_4b.bmodel]"
    ../cpp/yolox_sail.arm pic ../data/image/val2017 ../data/models/$2/yolox_s_int8_4b.bmodel 1 0.15 0.45 ../cpp/results $3
    judge_ret $? "run_example_cpp [yolox_s_int8_4b.bmodel]"
  fi
  fi
}

function run_example_py() {

  python3 ../python/yolox_bmcv.py \
    --is_video=0 \
    --loops=0 \
    --file_name=../data/image/val2017/ \
    --bmodel_path=../data/models/$2/yolox_s_fp32_1b.bmodel \
    --detect_threshold=0.25 \
    --nms_threshold=0.45 \
    --save_path=../python/results/yolox_bmcv \
    --device_id=$1
  judge_ret $? "yolox_bmcv.py [yolox_s_fp32_1b.bmodel]"

  python3 ../python/yolox_bmcv.py \
    --is_video=0 \
    --loops=0 \
    --file_name=../data/image/val2017/ \
    --bmodel_path=../data/models/$2/yolox_s_int8_4b.bmodel \
    --detect_threshold=0.15 \
    --nms_threshold=0.45 \
    --save_path=../python/results/yolox_bmcv \
    --device_id=$1
  judge_ret $? "yolox_bmcv.py [yolox_s_int8_4b.bmodel]"

  python3 ../python/yolox_opencv.py \
    --is_video=0 \
    --loops=0 \
    --file_name=../data/image/val2017/ \
    --bmodel_path=../data/models/$2/yolox_s_fp32_1b.bmodel \
    --detect_threshold=0.25 \
    --nms_threshold=0.45 \
    --save_path=../python/results/yolox_opencv \
    --device_id=$1
  judge_ret $? "yolox_opencv.py [yolox_s_fp32_1b.bmodel]"

  python3 ../python/yolox_opencv.py \
    --is_video=0 \
    --loops=0 \
    --file_name=../data/image/val2017/ \
    --bmodel_path=../data/models/$2/yolox_s_int8_4b.bmodel \
    --detect_threshold=0.15 \
    --nms_threshold=0.45 \
    --save_path=../python/results/yolox_opencv \
    --device_id=$1
  judge_ret $? "yolox_opencv.py [yolox_s_int8_4b.bmodel]"

}

function verify_result() {

  python3 ../tools/calc_mAP.py \
    --ground_truths=../data/ground_truths/instances_val2017.json \
    --detections=../cpp/results/val2017_yolox_s_fp32_1b_cpp.txt \
    
  judge_ret $? "Verify [cpp-bmcv] [yolox_s_fp32_1b.bmodel]"

  python3 ../tools/calc_mAP.py \
    --ground_truths=../data/ground_truths/instances_val2017.json \
    --detections=../cpp/results/val2017_yolox_s_int8_4b_cpp.txt \
    
  judge_ret $? "Verify [cpp-bmcv] [yolox_s_int8_4b.bmodel]"

  python3 ../tools/calc_mAP.py \
    --ground_truths=../data/ground_truths/instances_val2017.json \
    --detections=../python/results/yolox_bmcv/val2017_yolox_s_fp32_1b_py.txt \
    
  judge_ret $? "Verify [python-bmcv] [yolox_s_fp32_1b.bmodel]"

  python3 ../tools/calc_mAP.py \
    --ground_truths=../data/ground_truths/instances_val2017.json \
    --detections=../python/results/yolox_bmcv/val2017_yolox_s_int8_4b_py.txt \
    
  judge_ret $? "Verify [python-bmcv] [yolox_s_int8_4b.bmodel]"

  python3 ../tools/calc_mAP.py \
    --ground_truths=../data/ground_truths/instances_val2017.json \
    --detections=../python/results/yolox_opencv/val2017_yolox_s_fp32_1b_py.txt \
    
  judge_ret $? "Verify [python-opencv] [yolox_s_fp32_1b.bmodel]"

  python3 ../tools/calc_mAP.py \
    --ground_truths=../data/ground_truths/instances_val2017.json \
    --detections=../python/results/yolox_opencv/val2017_yolox_s_int8_4b_py.txt \
    
  judge_ret $? "Verify [python-opencv] [yolox_s_int8_4b.bmodel]"

}

if [ $1 == "--help" ];then
  sed -rn 's/^### ?//;T;p;' "$0"
fi

shell_dir=$(dirname $(readlink -f "$0"))
platform=$1
target=$2
tpu_id=$3
sail_dir=$4


if [[ $platform != "x86" && $platform != "soc" ]]; then
  echo "please type the right platform, only support x86 or soc"
  exit
fi

if [[ $target != "BM1684" && $target != "BM1684X" ]]; then
  echo "please type the right target, only support BM1684 or BM1684X"
  exit
fi

if [ ! -d "../data/models" ]; then
  download_files
fi

build_cpp $platform $sail_dir

run_example_cpp $platform $target $tpu_id

pip3 install opencv-python==3.4.10.37
pip3 install opencv-python-headless
pip3 install pycocotools

run_example_py $tpu_id $target

mkdir ../mAP
verify_result | tee ../mAP/mAP.txt



### Usages:
###     ./auto_test <plantform> <target> <tpu_id> <sail_dir>
###
### Options:
###     <plantform>   x86 or soc
###     <target>      BM1684 or BM1684X
###     <tpu_id>      tpu id
###     <sail_dir>    sail path 
