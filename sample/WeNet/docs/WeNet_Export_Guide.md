
# WeNet模型导出
 
## 获取源码
```bash
git clone https://github.com/wenet-e2e/wenet
git checkout v2.2.1
```

## 搭建WeNet导出环境
由于wenet的环境依赖较复杂，我们建议在docker或者conda中进行模型导出。
这里的示例是在docker中完成的。
```bash
docker run --name wenet -v $PWD:/workspace/ -it sophgo/tpuc_dev:latest
cd wenet
pip3 install -r requirements.txt
```
## 获取预训练模型
在这个网址中获取：https://github.com/wenet-e2e/wenet/blob/main/docs/pretrained_models.md

这里获取的是`aishell，CN，Checkpoint Model Conformer`，把aishell_u2pp_conformer_exp.tar.gz放到wenet根目录下面。

## 准备导出环境
```bash
export PYTHONPATH=$PWD:$PYTHONPATH
cd examples/aishell/s0
mkdir exp
mv ../../../aishell_u2pp_conformer_exp.tar.gz .
tar xvf aishell_u2pp_conformer_exp.tar.gz
cd ../../../
mkdir -p exp/20210601_u2++_conformer_exp
cp examples/aishell/s0/aishell_u2pp_conformer_exp/global_cmvn exp/20210601_u2++_conformer_exp/
```

## 导出流式wenet encoder/decoder的onnx
```bash
python3 wenet/bin/export_onnx_gpu.py --config examples/aishell/s0/aishell_u2pp_conformer_exp/train.yaml --checkpoint examples/aishell/s0/aishell_u2pp_conformer_exp/final.pt --output_onnx_dir ./onnx --num_decoding_left_chunks 5 --reverse_weight 0.3 --streaming
```
在运行完上述命令后，会在`onnx`目录下生成`config.yaml，decoder.onnx，encoder.onnx`。

## 导出非流式wenet encoder/decoder的onnx
```bash
python3 wenet/bin/export_onnx_gpu.py --config examples/aishell/s0/aishell_u2pp_conformer_exp/train.yaml --checkpoint examples/aishell/s0/aishell_u2pp_conformer_exp/final.pt --output_onnx_dir ./onnx_non_streaming --num_decoding_left_chunks 5 --reverse_weight 0.3
```
在运行完上述命令后，会在`onnx_non_streaming`目录下生成`config.yaml，decoder.onnx，encoder.onnx`。


## author & date: 
 - liheng.fang@sophgo.com, 2023.12.12