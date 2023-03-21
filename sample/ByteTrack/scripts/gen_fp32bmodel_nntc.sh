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

if [ ! $1 ]; then
    echo "Please set the target chip. Option: BM1684 and BM1684X"
    exit
else
    target=$1
fi

function gen_fp32bmodel() {

    ost=$1
    trace_name=${ost##*/}
    temp_name=${trace_name%.*}
    path=${ost%${temp_name}*}"/../middlefiles"

    batch_size=1

    mkdir -p ${path}"/../"$2
    outdir=${path}"/fp32model_bs"${batch_size}
    python3 -m bmneto --model=$1 \
        --shapes=[1,3,608,1088] \
        --target=$2 \
        --outdir=${outdir}
    mv ${outdir}"/compilation.bmodel" ${path}"/../"$2"/"${trace_name%.*}"_fp32_"${batch_size}"b.bmodel"
    judge_ret $? "convert to fp32_1b bmodel"

}

onnx_model="../models/onnx/bytetrack_s.onnx"

gen_fp32bmodel $onnx_model $target
judge_ret $? $target

