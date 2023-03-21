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
  rm -rf ../cpp/bytetrack_bmcv/CMakeFiles
  rm ../cpp/bytetrack_bmcv/cmake_install.cmake
  rm ../cpp/bytetrack_bmcv/CMakeCache.txt
  rm ../cpp/bytetrack_bmcv/Makefile
  #export TARGET_ARCH="x86"
  cmake ../cpp/bytetrack_bmcv/CMakeLists.txt -DSAIL_DIR=$1
  make -C ../cpp/bytetrack_bmcv/
}

function build_cpp() {
  echo "Start to build cpp, please wait......"

  pushd ../cpp/
  if [[ $1 == "x86" ]]
  then
    if [ ! -e "../cpp/bytetrack_bmcv/bytetrack_bmcv.pcie" ]; then
      OUT_BUILD=$(run_make_pcie $2)
      echo $OUT_BUILD
    else
      echo "already have excute file ../cpp/bytetrack_bmcv/bytetrack_bmcv.pcie"
    fi
  else if [[ $1 == "soc" ]]
  then
    if [ ! -e "../cpp/bytetrack_bmcv/bytetrack_bmcv.soc" ]; then
      echo "please ensure you have compiled excute file on soc"
      OUT_BUILD="failed"
    else
      echo "already have ../cpp/bytetrack_bmcv/bytetrack_bmcv.soc"
    fi
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
    chmod +x ../cpp/bytetrack_bmcv/bytetrack_bmcv.pcie
    ../cpp/bytetrack_bmcv/bytetrack_bmcv.pcie image ../datasets/MOT15/ADL-Rundle-6/img1 ../models/$2/bytetrack_s_fp32_1b.bmodel 10 0.1 0.7 ../cpp/bytetrack_bmcv/results $3
    judge_ret $? "run_example_cpp [bytetrack_s_fp32_4b.bmodel]"

  else if [ $1 = "soc" ];then
    chmod +x ../cpp/bytetrack_bmcv/bytetrack_bmcv.soc
    ../cpp/bytetrack_bmcv/bytetrack_bmcv.soc image ../datasets/MOT15/ADL-Rundle-6/img1 ../models/$2/bytetrack_s_fp32_1b.bmodel 10 0.1 0.7 ../cpp/bytetrack_bmcv/results $3
      judge_ret $? "run_example_cpp [bytetrack_s_fp32_4b.bmodel]"
        fi
  fi
}


function run_example_py() {

  python3 ../python/bytetrack_bmcv.py \
    --output_video=0 \
    --is_video=0 \
    --file_name=../datasets/MOT15/ADL-Rundle-6/img1 \
    --bmodel=../models/$2/bytetrack_s_fp32_1b.bmodel \
    --save_path=../python/results/bytetrack_bmcv \
    --score_th=0.1 \
    --nms_th=0.7 \
    --device_id=$1 \
    --track_thresh=0.5 \
    --track_buffer=30 \
    --match_thresh=0.8 \
    --min-box-area=10
  judge_ret $? "yolox_bmcv.py [bytetrack_s_fp32_1b.bmodel]"

  python3 ../python/bytetrack_opencv.py \
    --output_video=0 \
    --is_video=0 \
    --file_name=../datasets/MOT15/ADL-Rundle-6/img1 \
    --bmodel=../models/$2/bytetrack_s_fp32_1b.bmodel \
    --save_path=../python/results/bytetrack_opencv \
    --score_th=0.1 \
    --nms_th=0.7 \
    --device_id=$1 \
    --track_thresh=0.5 \
    --track_buffer=30 \
    --match_thresh=0.8 \
    --min-box-area=10
  judge_ret $? "yolox_opencv.py [bytetrack_s_fp32_1b.bmodel]"

}

function verify_result() {

  python3 ../tools/eval_mot.py \
    --ground_truths=../datasets/MOT15/ADL-Rundle-6/gt/gt.txt \
    --detections=../python/results/bytetrack_bmcv/img1_bytetrack_s_fp32_1b_py.txt \

  judge_ret $? "Verify [python-bmcv] [bytetrack_s_fp32_1b.bmodel]"

  python3 ../tools/eval_mot.py \
    --ground_truths=../datasets/MOT15/ADL-Rundle-6/gt/gt.txt \
    --detections=../python/results/bytetrack_opencv/img1_bytetrack_s_fp32_1b_py.txt \

  judge_ret $? "Verify [python-opencv] [bytetrack_s_fp32_1b.bmodel]"

  python3 ../tools/eval_mot.py \
  --ground_truths=../datasets/MOT15/ADL-Rundle-6/gt/gt.txt \
  --detections=../cpp/bytetrack_bmcv/results/img1_bytetrack_s_fp32_1b_cpp.txt \

  judge_ret $? "Verify [cpp-bmcv] [bytetrack_s_fp32_1b.bmodel]"

}

function download_files(){
  chmod +x ../scripts/download.sh
  ./download.sh
}

if [ $# == 0 ] || [ $1 == "--help" ];then
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

if [ ! -d "../models" ]; then
  download_files
fi

build_cpp $platform $sail_dir

run_example_cpp $platform $target $tpu_id

pip3 install motmetrics

run_example_py $tpu_id $target

verify_result



### Usages:
###     ./auto_test <plantform> <target> <tpu_id> <sail_dir>
### ./auto_test.sh soc BM1684 0 /opt/sophon/sophon-sail
### Options:
###     <plantform>   x86 or soc
###     <target>      BM1684 or BM1684X
###     <tpu_id>      tpu id
###     <sail_dir>    sail path, normally /opt/sophon/sophon-sail
