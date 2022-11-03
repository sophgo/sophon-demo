function judge_ret() {
    if [[ $1 == 0 ]]; then
        echo "Passed: $2"
        echo ""
    else
        echo "Failed: $2"
        exit 1
    fi
    sleep 2
}

function gen_bmodel() {

    ost=$1
    trace_name=${ost##*/}
    temp_name=${trace_name%.*}
    path=${ost%${temp_name}*}"/../middlefiles"
    model=${path}"/"${trace_name%.*}"_bmnetp_test_fp32.prototxt"
    weights=${path}"/"${trace_name%.*}"_bmnetp.fp32umodel"

    batch_size=1
    model_int8=${path}"/"${trace_name%.*}"_bmnetp_deploy_int8_unique_top.prototxt"
    weight_int8=${path}"/"${trace_name%.*}"_bmnetp.int8umodel"
    outdir=${path}"/int8model_bs"${batch_size}

    echo "Start convert to int8_1b bmodel..."
    bmnetu -model ${model_int8} \
        -weight ${weight_int8} \
        -max_n ${batch_size} \
        -prec=INT8 \
        -dyn=0 \
        -cmp=1 \
        -target=$2 \
        -outdir=${outdir}
    judge_ret $? "convert to int8_1b bmodel"

    mkdir -p ${path}"/../"$2
    mv ${outdir}"/compilation.bmodel" ${path}"/../"$2"/"${trace_name%.*}"_int8_"${batch_size}"b.bmodel"
    judge_ret $? "move bmodel"

    batch_size=4
    outdir=${path}"/int8model_bs"${batch_size}

    echo "Start convert to int8_4b bmodel..."
    bmnetu -model ${model_int8} \
        -weight ${weight_int8} \
        -max_n ${batch_size} \
        -prec=INT8 \
        -dyn=0 \
        -cmp=1 \
        -target=$2 \
        -outdir=${outdir}
    judge_ret $? "convert to int8_4b bmodel"

    echo "Start convert to fp32_4b bmodel..."
    mkdir -p ${path}"/../"$2
    mv ${outdir}"/compilation.bmodel" ${path}"/../"$2"/"${trace_name%.*}"_int8_"${batch_size}"b.bmodel"
    judge_ret $? "move bmodel"

}

if [ ! -d "../data/models/torch" ]; then
    get_torch_model
fi

lmdb_folder="../data/image/lmdb"
traced_pt="../data/models/torch/yolox_s.pt"
width=640
height=640

dst_image_path=$lmdb_folder"/../resize_images/"
lmdbfolder=$lmdb_folder"/../lmdb_dataset"
rm -rf lmdbfolder
if [ ! -d "../data/image/lmdb" ]; then
    get_lmdbpics
fi

python3 ../tools/image_resize.py --ost_path=$lmdb_folder --dst_path=$dst_image_path --dst_width=$width --dst_height=$height
judge_ret $? "image_resize"
echo "Start convert_imageset..."
python3 ../tools/convert_imageset.py \
    --imageset_rootfolder=$dst_image_path \
    --imageset_lmdbfolder=$lmdbfolder \
    --resize_height=$height \
    --resize_width=$width \
    --shuffle=True \
    --bgr2rgb=False \
    --gray=False
judge_ret $? "convert_imageset"

python3 ../tools/gen_fp32_umodel.py \
    --trace_model=$traced_pt \
    --data_path=$lmdbfolder \
    --dst_width=$width \
    --dst_height=$height

judge_ret $? "gen_fp32_umodel"

ost=$traced_pt
trace_name=${ost##*/}
temp_name=${trace_name%.*}
path=${ost%${temp_name}*}"../middlefiles"
model=${path}"/"${trace_name%.*}"_bmnetp_test_fp32.prototxt"
weights=${path}"/"${trace_name%.*}"_bmnetp.fp32umodel"

echo "Start calibration..."
echo $model
echo $weights
calibration_use_pb \
    quantize \
    -model=$model \
    -weights=$weights \
    -iterations=100 \
    -bitwidth=TO_INT8 \
    -fpfwd_outputs="15" \
    -save_test_proto 
judge_ret $? "calibration"

gen_bmodel $traced_pt BM1684
judge_ret $? "BM1684"

gen_bmodel $traced_pt BM1684X
judge_ret $? "BM1684X"
