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

    batch_size=1

    mkdir ${path}"/../"$2
    outdir=${path}"/fp32model_bs"${batch_size}
    python3 -m bmnetp --model=$1 \
        --shapes=[1,3,640,640] \
        --target=$2 \
        --outdir=${outdir}
    mv ${outdir}"/compilation.bmodel" ${path}"/../"$2"/"${trace_name%.*}"_fp32_"${batch_size}"b.bmodel"
    judge_ret $? "convert to fp32_1b bmodel"

    batch_size=4    
    python3 -m bmnetp --model=$1 \
        --shapes=[4,3,640,640] \
        --target=$2 \
        --outdir=${outdir}
    mv ${outdir}"/compilation.bmodel" ${path}"/../"$2"/"${trace_name%.*}"_fp32_"${batch_size}"b.bmodel"
    judge_ret $? "convert to fp32_4b bmodel"

}

traced_pt="../data/models/torch/yolox_s.pt"

gen_bmodel $traced_pt BM1684
judge_ret $? "BM1684"

gen_bmodel $traced_pt BM1684X
judge_ret $? "BM1684X"
