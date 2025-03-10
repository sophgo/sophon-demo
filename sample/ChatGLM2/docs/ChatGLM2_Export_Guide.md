# ChatGLM2模型导出与编译

## 1. 准备工作

ChatGLM2模型导出需要依赖[ChatGLM2官方仓库](https://huggingface.co/THUDM/chatglm2-6b)。onnx模型导出和转bmodel模型推荐在mlir部分提供的docker中完成。

**注意：** 

- 编译模型需要在x86主机完成。
- ChatGLM2-6B官方库25G左右，转模型需要保证运行内存至少32G以上，导出onnx模型需要存储空间50G以上，fp16模型转换需要存储空间180G以上，int8和int4模型需要的空间会更少。

## 2. 主要步骤

### 2.1 TPU-MLIR环境搭建

建议使用TPU-MLIR编译BModel，模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需**在TPU-MLIR环境中**进入例程目录。

### 2.2 获取onnx

### 2.2.1 下载ChatGLM2官方代码

**注：** ChatGLM2-6B官方库25G左右

```bash
git lfs install
git clone git@hf.co:THUDM/chatglm2-6b
```

### 2.1.2 对官方代码进行三处修改：

- 将config.json文件中seq_length配置为512；

- 将modeling_chatglm.py文件中的如下代码：

```bash
if attention_mask is not None:
    attention_scores = attention_scores.masked_fill(attention_mask, float("-inf"))
```
修改为：

```bash
if attention_mask is not None:
    attention_scores = attention_scores + (attention_mask * -10000.0)
```
这样修改可以提升效率，使用masked_fill效率低下。

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
这样修改可以解决pytorch2.0导出有bug的问题。

### 2.1.3 导出onnx

- 指定chatglm2-6B官方仓库的python路径

```bash
# 将/workspace/chatglm2-6b换成docker环境中您的chatglm2-6b仓库的路径
export PYTHONPATH=/workspace/chatglm2-6b:$PYTHONPATH
```

- 导出所有onnx模型，如果过程中提示缺少某些组件，直接**pip install**组件即可

```bash
# 将/workspace/chatglm2-6b换成docker环境中您的chatglm2-6b仓库的路径
python3 tools/export_onnx.py --path /workspace/chatglm2-6b
```
此时有大量onnx模型被导出到本例程中ChatGLM2/models/onnx的目录。

### 2.2 bmodel编译

目前TPU-MLIR支持1684x对ChatGLM2进行F16, INT8和INT4量化。

- 生成FP16 bmodel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，如：

```bash
./scripts/gen_fp16bmodel_mlir.sh
```

​执行上述命令会在`models/BM1684X/`文件夹下生成`chatglm2-6b_f16.bmodel`文件，即转换好的FP16 BModel。

- 生成INT8 bmodel

​本例程在`scripts`目录下提供了TPU-MLIR编译INT8 BModel的脚本，请注意修改`gen_int8bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，如：

```bash
./scripts/gen_int8bmodel_mlir.sh
```

​执行上述命令会在`models/BM1684X/`文件夹下生成`chatglm2-6b_int8.bmodel`文件，即转换好的INT8 BModel。

- 生成INT4 bmodel

​本例程在`scripts`目录下提供了TPU-MLIR编译INT4 BModel的脚本，请注意修改`gen_int4bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，如：

```bash
./scripts/gen_int4bmodel_mlir.sh
```
​执行上述命令会在`models/BM1684X/`文件夹下生成`chatglm2-6b_int4.bmodel`文件，即转换好的INT4 BModel。

### 2.3 准备tokenizer

将官方代码中chatglm2-6b/tokenizer.model放到BM1684X目录下。

至此导出onnx与转模型部分结束。
