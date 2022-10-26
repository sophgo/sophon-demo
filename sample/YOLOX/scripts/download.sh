function download_lmdbpics() {
    pip3 install dfn
    python3 -m dfn --url=http://219.142.246.77:65000/sharing/3zo5a79eW
    mkdir -p $(pwd)/../data/image/
    unzip -d $(pwd)/../data/image/ ./lmdb.zip
    rm ./lmdb.zip
}

function download_torch_model() {
    pip3 install dfn
    python3 -m dfn --url=http://219.142.246.77:65000/sharing/KDEyllzUm
    mkdir -p $(pwd)/../data/models/torch
    mv ./yolox_s.pt $(pwd)/../data/models/torch
}

function download_bmodels() {
    pip3 install dfn
    python3 -m dfn --url=http://219.142.246.77:65000/sharing/e9F9LWimX
    unzip -d ../data ./models.zip
    rm ./models.zip
}

function download_gt() {
    pip3 install dfn
    python3 -m dfn --url=http://219.142.246.77:65000/sharing/R45C88HCn
    mkdir $(pwd)/../data/ground_truths
    mv ./instances_val2017.json $(pwd)/../data/ground_truths/instances_val2017.json
}

function download_val() {
    python3 -m dfn --url=http://219.142.246.77:65000/sharing/9JDrlawwB
    unzip -d ../data/image ./val2017.zip
    rm ./val2017.zip
}

download_lmdbpics
download_torch_model
download_bmodels
download_gt
download_val