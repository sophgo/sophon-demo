# ChatGLM4模型导出与编译

## 1. 准备工作

ChatGLM4模型导出需要依赖[ChatGLM4官方仓库](https://huggingface.co/THUDM/glm-4-9b-chat)。onnx模型导出和转bmodel模型推荐在mlir部分提供的docker中完成。

**注意：** 

- 编译模型需要在x86主机完成。
- 生成bmodel耗时大概3小时以上，建议64G内存以及200G以上硬盘空间，不然很可能OOM或者no space left。

## 2. 主要步骤

模型编译前需要安装TPU-MLIR。安装好后需在TPU-MLIR环境中进入例程目录。先导出onnx，然后使用TPU-MLIR将onnx模型编译为BModel。编译的具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

### 2.1 TPU-MLIR环境搭建

建议使用TPU-MLIR编译BModel，模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需**在TPU-MLIR环境中**进入例程目录。

### 2.2 获取onnx

### 2.2.1 下载ChatGLM4官方代码

**注：** ChatGLM4-9B官方库18G左右，在下载之前，要确认自己有huggingface官网的access token或者SSH key。
```bash
git lfs install
git clone git@hf.co:THUDM/glm-4-9b-chat
```
如果git clone完代码之后出现卡住，可以尝试`ctrl+c`中断，然后进入仓库运行`git lfs pull`。

### 2.1.2 对齐环境和代码：
本例程的`tools`目录下提供了修改好之后的`config.json`和`modeling_chatglm.py`。可以直接替换掉原仓库的文件：
```bash
sudo apt-get update
sudo apt-get install pybind11-dev
pip install -r tools/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
cp tools/glm-4-9b-chat/config.json glm-4-9b-chat/
cp tools/glm-4-9b-chat/modeling_chatglm.py glm-4-9b-chat/
```

### 2.1.3 导出onnx

- 指定glm-4-9b-chat官方仓库的python路径

```bash
# 将/workspace/glm-4-9b-chat换成docker环境中您的glm-4-9b-chat仓库的路径
export PYTHONPATH=/workspace/glm-4-9b-chat:$PYTHONPATH
```

- 导出所有onnx模型，如果过程中提示缺少某些组件，直接**pip install**组件即可

```bash
# 将/workspace/glm-4-9b-chat换成docker环境中您的glm-4-9b-chat仓库的路径
python3 tools/export_onnx.py --model_path /workspace/glm-4-9b-chat --seq_length 512
```
此时有大量onnx模型被导出到本例程中`ChatGLM4/models/onnx`的目录。

### 2.2 bmodel编译

目前TPU-MLIR支持1684x对ChatGLM4进行INT8和INT4量化，使用如下命令生成bmodel。

```bash
mv ./tmp ./scripts
./scripts/gen_bmodel.sh --mode int4 #int8
```

其中，mode可以指定int8/int4，编译成功之后，模型将会存放在`models/BM1684X/`目录下。

### 2.3 准备tokenizer

如果您之前没有运行过下载脚本，那么您需要运行它以获取tokenizer。经过上面的步骤，现在你的目录下已经存在models文件夹，所以它只会下载tokenizer。
```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```