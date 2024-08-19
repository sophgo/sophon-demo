# ChatGLM3模型导出与编译

## 1. 准备工作

ChatGLM3模型导出需要依赖[ChatGLM3官方仓库](https://huggingface.co/THUDM/chatglm3-6b)。onnx模型导出和转bmodel模型推荐在mlir部分提供的docker中完成。

**注意：** 

- 编译模型需要在x86主机完成。
- ChatGLM3-6B官方库50G左右，转模型需要保证运行内存至少32G以上，导出onnx模型需要存储空间100G以上，fp16模型转换需要存储空间200G以上，int8和int4模型需要的空间会更少。

## 2. 主要步骤

### 2.1 TPU-MLIR环境搭建

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

### 2.2 获取onnx

### 2.2.1 下载ChatGLM3官方代码

**注：** ChatGLM3-6B官方库50G左右，在下载之前，要确认自己有huggingface官网的access token或者SSH key。
```bash
git lfs install
git clone git@hf.co:THUDM/chatglm3-6b
```
如果git clone完代码之后出现卡住，可以尝试`ctrl+c`中断，然后进入仓库运行`git lfs pull`。

如果无法从官网下载，也可以下载我们之前下好的，压缩包20G左右
```bash
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:ext_model_information/LLM/chatglm3-6b.tar.gz
tar zxvf chatglm3-6b.tar.gz
```

### 2.1.2 修改官方代码：
本例程的`tools`目录下提供了修改好之后的`config.json`和`modeling_chatglm.py`。可以直接替换掉原仓库的文件：
```bash
cp tools/chatglm3-6b/config.json chatglm3-6b/
cp tools/chatglm3-6b/modeling_chatglm.py chatglm3-6b/
```
以下是对应的三处修改：
- 将config.json文件中seq_length配置为512；

- 将modeling_chatglm.py文件中的如下代码：

```bash
if attention_mask is not None:
    attention_scores = attention_scores.masked_fill(attention_mask, float("-inf"))
```
修改为：

```bash
if attention_mask is not None:
    attention_scores = attention_scores + attention_mask
```
这样修改可以提升效率，使用masked_fill效率低下，我们把这一步骤从模型中拿了出来，放到了前处理里面。

- 将modeling_chatglm.py文件中的如下代码：

```bash
pytorch_major_version = int(torch.__version__.split('.')[0])
if pytorch_major_version >= 2:
```
修改为

```bash
pytorch_major_version = int(torch.__version__.split('.')[0])
if False:
```
这是因为ONNX无法支持torch.nn.functional.scaled_dot_product_attention算子的转换。

### 2.1.3 导出onnx

- 指定chatglm3-6B官方仓库的python路径

```bash
# 将/workspace/chatglm3-6b换成docker环境中您的chatglm3-6b仓库的路径
export PYTHONPATH=/workspace/chatglm3-6b:$PYTHONPATH
```

- 导出所有onnx模型，如果过程中提示缺少某些组件，直接**pip install**组件即可

```bash
# 将/workspace/chatglm3-6b换成docker环境中您的chatglm3-6b仓库的路径
python3 tools/export_onnx.py --model_path /workspace/chatglm3-6b --onnx_path ./models/onnx
```
此时有大量onnx模型被导出到本例程中`ChatGLM3/models/onnx`的目录。

### 2.2 bmodel编译

目前TPU-MLIR支持BM1684X/BM1688对ChatGLM3进行F16/INT8/INT4量化，您可以使用如下命令生成bmodel。

```bash
./script/gen_bmodel --mode fp16 --target bm1684x --num_core 1 #bm1688支持--num_core 2
```

其中，mode可以指定fp16/int8/int4，编译成功之后，模型将会存放在`models/BM1684X/`或`models/BM1688/`目录下。

### 2.3 准备tokenizer

如果您之前没有运行过下载脚本，那么您需要运行它以获取tokenizer。经过上面的步骤，现在你的目录下已经存在models文件夹，所以它只会下载tokenizer。
```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```