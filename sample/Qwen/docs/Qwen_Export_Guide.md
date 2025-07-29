# Qwen模型导出与编译

## 1. 准备工作

Qwen模型导出需要依赖[Qwen官方仓库](https://huggingface.co/Qwen)。onnx模型导出和转bmodel模型推荐在mlir部分提供的docker中完成。

**注意：** 

- 编译模型需要在x86主机完成。
- Qwen-7B官方库50G左右，转模型需要保证运行内存至少40G以上，导出onnx模型需要存储空间100G以上，请确保有足够的硬件空间完成对应的操作。


## 2. 主要步骤

模型编译前需要安装TPU-MLIR。安装好后需在TPU-MLIR环境中进入例程目录。对于Qwen3,Qwen2无需导出onnx；对于其他模型，需要先导出onnx，然后使用TPU-MLIR将onnx模型编译为BModel。编译的具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

### 2.1 TPU-MLIR环境搭建

### 2.1.1 安装docker

若已安装docker，请跳过本节。
```bash
# 安装docker
sudo apt-get install docker.io
# docker命令免root权限执行
# 创建docker用户组，若已有docker组会报错，没关系可忽略
sudo groupadd docker
# 将当前用户加入docker组
sudo usermod -aG docker $USER
# 切换当前会话到新group或重新登录重启X会话
newgrp docker​ 
```
> **提示**：需要logout系统然后重新登录，再使用docker就不需要sudo了。

### 2.1.2 下载并解压TPU-MLIR

从sftp上获取TPU-MLIR压缩包
```bash
pip3 install dfss --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/tpu-mlir_v1.10.beta.0-31-g896b42e8c-20240809.tar.gz
tar -xf tpu-mlir_v1.10.beta.0-31-g896b42e8c-20240809.tar.gz
#1684X导出Deepseek-R1-Distill-Qwen专用
python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/tpu-mlir_v1.14.beta.0-25-gbca81b22c-20250107.tar.gz
tar -xf tpu-mlir_v1.14.beta.0-25-gbca81b22c-20250107.tar.gz
```

### 2.1.3 创建并进入docker

TPU-MLIR使用的docker是sophgo/tpuc_dev:latest, docker镜像和tpu-mlir有绑定关系，少数情况下有可能更新了tpu-mlir，需要新的镜像。
```bash
docker pull sophgo/tpuc_dev:latest
# 这里将本级目录映射到docker内的/workspace目录,用户需要根据实际情况将demo的目录映射到docker里面
# myname只是举个名字的例子, 请指定成自己想要的容器的名字
docker run --name myname -v $PWD:/workspace -it sophgo/tpuc_dev:latest
# 此时已经进入docker，并在/workspace目录下
# 初始化软件环境
cd /workspace/tpu-mlir_vx.y.z-<hash>-<date>
source ./envsetup.sh
```
此镜像仅onnx模型导出和编译量化模型，程序编译和运行请在开发和运行环境中进行。更多TPU-MLIR的教程请参考[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)的《TPU-MLIR快速入门手册》和《TPU-MLIR开发参考手册》。

### 2.2 获取onnx

### 2.2.1 下载Qwen官方代码

**注：** 
- Qwen1.5-1.8B官方库50G左右，在下载之前，要确认自己有huggingface官网的access token或者SSH key。
- Qwen1.5-7B官方库50G左右，在下载之前，要确认自己有huggingface官网的access token或者SSH key。
- Qwen2.5-1.5B官方库50G左右，在下载之前，要确认自己有huggingface官网的access token或者SSH key。
- Qwen2.5-7B官方库50G左右，在下载之前，要确认自己有huggingface官网的access token或者SSH key。
- Deepseek-R1-Distill-Qwen-1.5B官方库3.5G左右，在下载之前，要确认自己有huggingface官网的access token或者SSH key。
- Deepseek-R1-Distill-Qwen-7B官方库15G左右，在下载之前，要确认自己有huggingface官网的access token或者SSH key。
- Deepseek-R1-Distill-Qwen-14B官方库30G左右，在下载之前，要确认自己有huggingface官网的access token或者SSH key。


```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen1.5-7B-Chat
git clone https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat
git clone https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
```
如果git clone完代码之后出现卡住，可以尝试`ctrl+c`中断，然后进入仓库运行`git lfs pull`。

### 2.2.2 修改官方代码：
本例程的`tools`目录下提供了修改好之后的`config.json`和`modeling_qwen.py`。可以直接替换掉原仓库的文件：

Qwen
```bash
cp tools/Qwen-7B-Chat/config.json Qwen-7B-Chat/
cp tools/Qwen-7B-Chat/modeling_qwen.py Qwen-7B-Chat/
```

Qwen1.5
```bash
pip install transformers_stream_generator einops tiktoken accelerate transformers==4.37.0
cp tools/Qwen1.5-7B-Chat/config.json Qwen1.5-7B-Chat/
cp tools/Qwen1.5-7B-Chat/modeling_qwen2.py /usr/local/lib/python3.10/dist-packages/transformers/models/qwen2/

cp tools/Qwen1.5-1.8B-Chat/config.json Qwen1.5-1.8B-Chat/
cp tools/Qwen1.5-1.8B-Chat/modeling_qwen2.py /usr/local/lib/python3.10/dist-packages/transformers/models/qwen2/
```

Qwen2
```bash
pip install transformers_stream_generator einops tiktoken accelerate torch==2.0.1+cpu torchvision==0.15.2 transformers==4.41.2
cp tools/Qwen2-1.5B-Instruct/config.json Qwen2-1.5B-Instruct/
cp tools/Qwen2-1.5B-Instruct/modeling_qwen2.py /usr/local/lib/python3.10/dist-packages/transformers/models/qwen2/

cp tools/Qwen2-7B-Instruct/config.json Qwen2-7B-Instruct/
cp tools/Qwen2-7B-Instruct/modeling_qwen2.py /usr/local/lib/python3.10/dist-packages/transformers/models/qwen2/
```

Qwen2.5
```bash
pip install transformers_stream_generator einops tiktoken accelerate torch==2.0.1+cpu torchvision==0.15.2 transformers==4.45.2
cp tools/Qwen2.5-7B-Instruct/modeling_qwen2.py /usr/local/lib/python3.10/dist-packages/transformers/models/qwen2/
```

Deepseek-R1-Distill-Qwen-1.5B/7B (BM1684X，BM1688都需要)
```bash
pip install transformers_stream_generator einops tiktoken accelerate torch==2.0.1+cpu torchvision==0.15.2 transformers==4.45.2
cp tools/DeepSeek_R1_Distill_Qwen2.5-1.5B-Instruct/modeling_qwen2.py /usr/local/lib/python3.10/dist-packages/transformers/models/qwen2/
```


### 2.2.3 导出onnx

- 导出所有onnx模型，如果过程中提示缺少某些组件，直接**pip install**组件即可

Qwen
```bash
# bm1684x 单芯
python3 tools/export_onnx_qwen.py --model_path /workspace/Qwen-7B-Chat --seq_length 512 

# bm1684x 多芯
python3 tools/export_onnx_qwen.py --model_path /workspace/Qwen-7B-Chat --seq_length 512 --lm_head_with_topk 1
```

Qwen1.5
```bash
# bm1684x 单芯
python3 tools/export_onnx_qwen1.5.py --model_path /workspace/Qwen1.5-7B-Chat --seq_length 512 

# bm1684x 多芯
python3 tools/export_onnx_qwen1.5.py --model_path /workspace/Qwen1.5-7B-Chat --seq_length 512 --lm_head_with_topk 1

# bm1688/cv186x
python3 tools/export_onnx_qwen1.5.py --model_path /workspace/Qwen1.5-1.8B-Chat --seq_length 512 --lite True
```

Qwen2
```bash
# bm1684x 单芯
python3 tools/export_onnx_qwen2.py --model_path /workspace/Qwen2-7B-Instruct --seq_length 512 

# bm1684x 多芯
python3 tools/export_onnx_qwen2_parallel.py --model_path /workspace/Qwen2-7B-Instruct --seq_length 512 --lm_head_with_topk 1
```

Qwen2.5
```bash
# bm1684x 单芯
python3 tools/export_onnx_qwen2_5.py --model_path /workspace/Qwen2.5-7B-Instruct --seq_length 512 

# bm1688 单芯
python3 tools/export_onnx_qwen2_5.py --model_path /workspace/Qwen2.5-1.5B-Instruct --seq_length 512 --lmhead_with_topk 1
```

Deepseek-R1-Distill-Qwen-1.5B(BM1688，[参考地址](https://github.com/sophgo/LLM-TPU_Lite/tree/main/models/DeepseekR1Distill))
```bash
# bm1688 单芯
python3 tools/export_onnx_DeepSeek_R1_Distill_qwen2_5_BM1688.py --model_path /workspace/DeepSeek-R1-Distill-Qwen-1.5B --seq_length 1024 --lmhead_with_topk 1
```

此时有大量onnx模型被导出到本例程中`Qwen/models/onnx`的目录。

### 2.3 bmodel编译
首先需要在mlir工具下激活环境，[mlir下载地址可参考](./Qwen_Export_Guide.md/#212-下载并解压tpu-mlir)
```bash
cd tpu-mlir_v1.10.beta.0-31-g896b42e8c-20240809
source envsetup.sh
```
目前TPU-MLIR支持1684x对Qwen进行BF16,INT8和INT4量化，使用如下命令生成bmodel。

Qwen
```bash
# bm1684x 单芯
./scripts/gen_bmodel.sh --target bm1684x --mode int4 --name qwen-7b --seq_length 512 --addr_mode io_alone

# bm1684x 多芯
./scripts/gen_bmodel.sh --target bm1684x --mode int4 --name qwen-7b --seq_length 512 --addr_mode io_alone --num_device 2 --dynamic 1
```

Qwen1.5
```bash
# bm1684x 单芯
./scripts/gen_bmodel.sh --target bm1684x --mode int4 --name qwen1.5-7b --seq_length 512 --addr_mode io_alone

# bm1684x 多芯
./scripts/gen_bmodel.sh --target bm1684x --mode int4 --name qwen1.5-7b --seq_length 512 --addr_mode io_alone --num_device 2 --dynamic 1

# bm1688
./scripts/gen_bmodel.sh --target bm1688 --mode int4 --name qwen1.5-1.8b --seq_length 512 --addr_mode io_alone 

# cv186x
./scripts/gen_bmodel.sh --target cv186x --mode int4 --name qwen1.5-1.8b --seq_length 512 --addr_mode io_alone 
```
Qwen2
```bash
# bm1684x 单芯
./scripts/gen_bmodel.sh --target bm1684x --mode int4 --name qwen2-7b --seq_length 512 --addr_mode io_alone

# bm1684x 多芯
./scripts/gen_bmodel_qwen2_parallel.sh --mode int4 --name qwen2-7b --seq_length 512 --addr_mode io_alone --num_device 2 --dynamic 1
```

Qwen2.5
```bash
# bm1684x 单芯
./scripts/gen_bmodel.sh --target bm1684x --mode int4 --name qwen2.5-7b --seq_length 512 --addr_mode io_alone

# bm1688 单芯
./scripts/gen_bmodel_qwen2_1688.sh --name qwen2.5-1.5b --seq_length 512 --mode int4 --addr_mode io_alone
```

Deepseek-R1-Distill-Qwen-1.5B (BM1688，[参考地址](https://github.com/sophgo/LLM-TPU_Lite/tree/main/models/DeepseekR1Distill))
```bash
# bm1688 单芯
./scripts/gen_bmodel_deepseek_r1_distill_qwen_1_5b.sh --name qwen2.5-1.5b --seq_length 1024 --mode int4 --addr_mode io_alone
```

其中，mode可以指定bf16/int8/int4，编译成功之后，BM1684X模型将会存放在`models/BM1684X`目录下，BM1688模型将会存放在`models/BM1688`目录下，CV186X模型将会存放在`models/CV186X`目录下。

### 2.3.1 BM1684X编译 Deepseek-R1-Distill-Qwen bmodel
```bash
# bm1684x 单芯
# 请注意1684X导出Deepseek-R1-Distill-Qwen bmodel仅需要运行该脚本，无需转onnx，请使用2.1.2中专用版本tpu-mlir
python tools/model_export_BM1684X_DS_qwen.py --quantize w4bf16 --tpu_mlir_path /workspace/tpu-mlir/ --torch_path /workspace/models/DeepSeek-R1-Distill-Qwen-1.5B --seq_length 1024  --out_dir deepseek-r1-distill-qwen-1.5b-1024
```
其中，tpu_mlir_path指定tpu-mlir地址，编译成功之后，BM1684X模型将会存放在torch_path指定目录下，该目录下还会存在onnx和bmodel两个中间文件夹可以删除。

### 2.3.2 BM1684X编译QwQ-32B
请参考[LLM-TPU Qwen2_5](https://github.com/sophgo/LLM-TPU/tree/main/models/Qwen2_5/compile)

### 2.3.3 编译Qwen3
Qwen3无需导出onnx，可以使用llm_convert工具直接转换bmodel，可以参考文档[编译LLM模型](https://github.com/sophgo/tpu-mlir/blob/master/docs/quick_start/source_zh/10_llm_convert.rst)。首先需要在Huggingface下载Qwen3
```bash
# 下载模型
git lfs install
git clone git@hf.co:Qwen/Qwen3-4B-AWQ
# 如果是8B，则如下：
git clone git@hf.co:Qwen/Qwen3-8B-AWQ
```
在docker内编译模型生成bmodel
```bash
# bm1684x平台下执行：
llm_convert.py -m /workspace/Qwen3-4B -s 512 --quantize w4bf16 -g 128 -c bm1684x --out_dir qwen3_4b
# bm1688平台下执行：
llm_convert.py -m /workspace/Qwen3-4B -s 512 --quantize w4bf16 -g 128 -c bm1688 --out_dir qwen3_4b
```

> **注意**：
> 1. Qwen2及之后的模型都可以使用此工具直接转换bmodel；
> 2. 推荐下载AWQ量化版本的模型进行转换，可基本消除模型转换过程的精度损失。