# WeNetSpeech

sophon-demo的WeNet使用的源模型是基于aishell的，这里提供wenetspeech模型的导出、编译、测试方法(有提供可下载的测试包)。这里只测试了FP32的流式encoder，如果需要其他精度、decoder或非流式encoder可以自行探索。

## 1. 导出方法：

### 1.1 获取源码
```bash
git clone https://github.com/wenet-e2e/wenet
git check v2.2.1 #注意这一步很重要，因为新版本wenet改了输入输出和模型结构，和当前sophon-demo的WeNet不适配。
```

### 1.2 搭建WeNet导出环境
由于wenet的环境依赖较复杂，我们建议在docker或者conda中进行模型导出。
这里的示例是在docker中完成的(可以使用mlir提供的docker，另起一个容器)。
```bash
docker pull sophgo/tpuc_dev:latest
docker run --name wenet -v $PWD:/workspace/ -it sophgo/tpuc_dev:latest
cd wenet
pip3 install -r requirements.txt #下不下来的话可以用清华源 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 1.3 获取预训练模型
在这个网址中获取：https://github.com/wenet-e2e/wenet/blob/main/docs/pretrained_models.md

这里获取的是`wenetspeech，CN，Checkpoint Model Conformer`，把wenetspeech_u2pp_conformer_exp.tar.gz放到wenet根目录下面。

### 1.4 准备导出环境

当前工作目录应是wenet根目录：
```bash
export PYTHONPATH=$PWD:$PYTHONPATH
mkdir exp 
mv wenetspeech_u2pp_conformer_exp.tar.gz exp
cd exp
tar xvf wenetspeech_u2pp_conformer_exp.tar.gz
cd ..
```

### 1.5 导出流式wenet encoder/decoder的onnx
```bash
python3 wenet/bin/export_onnx_gpu.py --config exp/20220506_u2pp_conformer_exp_wenetspeech/train.yaml --checkpoint exp/20220506_u2pp_conformer_exp_wenetspeech/final.pt --num_decoding_left_chunks 5 --reverse_weight 0.3 --streaming --cmvn_file=exp/20220506_u2pp_conformer_exp_wenetspeech/global_cmvn --output_onnx_dir onnx_streaming_wenetspeech
```
在运行完上述命令后，会在`onnx_streaming_wenetspeech`目录下生成`config.yaml，decoder.onnx，encoder.onnx`。

如果报错:
```
Traceback (most recent call last):
  File "/workspace/open-source/wenet/wenet/bin/export_onnx_gpu.py", line 963, in <module>
    model = init_model(configs)
  File "/workspace/open-source/wenet/wenet/utils/init_model.py", line 35, in init_model
    mean, istd = load_cmvn(configs['cmvn_file'], configs['is_json_cmvn'])
KeyError: 'is_json_cmvn'
```
将`wenet/wenet/utils/init_model.py`的第35行的`configs['is_json_cmvn']`直接改成`True`。

## 2. 编译方法：

### 2.1 配置mlir环境：
另起一个终端：
```bash
docker run --privileged --name tpu_mlir -v $PWD:/workspace -it sophgo/tpuc_dev:latest #注意调整 -v, 映射你自己想要的目录到docker中
#此时已经进入docker，并在/workspace目录下
pip install tpu_mlir -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tpu_mlir[onnx] -i https://pypi.tuna.tsinghua.edu.cn/simple
```
### 2.2 编译命令：

model_transform，注意把--model_def对应的参数换成1.5章节中导出的encoder.onnx:
```bash
    model_transform.py \
        --model_name wenet_encoder_streaming \
        --model_def ../models/onnx/onnx_streaming_wenetspeech/encoder.onnx \
        --input_shapes [[1,67,80],[1],[1,1],[1,12,8,80,128],[1,12,512,14],[1,1,80]] \
        --mlir wenet_encoder_streaming.mlir
```
model_deploy，这里编译的是单核模型：
```bash
    model_deploy.py \
        --mlir wenet_encoder_streaming.mlir \
        --quantize F32 \
        --chip bm1688 \
        --model wenet_encoder_streaming_fp32_wenetspeech.bmodel
```

## 3. 测试方法
参考sophon-demo WeNet的文档，在BM1688 SoC上配置好相应的python/c++测试环境，并拷贝相应的数据集、模型、配置文件到测试环境中。注意dict和config应该传20220506_u2pp_conformer_exp_wenetspeech中的units.txt和train.yaml。

这里提供一个打包好的测试包，可以直接在BM1688上下载并测试，下载命令：
```bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m dfss --url=open@sophgo.com:sophon-demo/WeNet/wenetspeech.tar.gz
tar xvf wenetspeech.tar.gz
cd wenetspeech
```

python测试命令如下：
```bash
pip3 install torch==1.13.1 torchaudio==0.13.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
cd python
python3 wenet.py --encoder_bmodel models_wenetspeech/BM1688/wenet_encoder_streaming_fp32_wenetspeech.bmodel --dict wenetspeech_config/units.txt --config wenetspeech_config/train.yaml --input ../datasets/aishell_S0764/aishell_S0764.list
```
c++测试命令如下：
```bash
cd ../cpp
export LD_LIBRARY_PATH=$PWD/cross_compile_module/ctcdecode-cpp/openfst-1.6.3/src/lib/.libs/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PWD/cross_compile_module/ctcdecode-cpp/build/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PWD/cross_compile_module/ctcdecode-cpp/build/3rd_party/kenlm/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PWD/cross_compile_module/3rd_party/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PWD/cross_compile_module/3rd_party/lib/blas/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PWD/cross_compile_module/3rd_party/lib/lapack/:$LD_LIBRARY_PATH

./wenet.soc --config_file=../python/wenetspeech_config/train.yaml --dict_file=../python/wenetspeech_config/units.txt --encoder_bmodel=../python/models_wenetspeech/BM1688/wenet_encoder_streaming_fp32_wenetspeech.bmodel --input=../datasets/aishell_S0764/aishell_S0764.list
```

