# Qwen模型导出与编译

## 1. 准备工作

Qwen模型导出需要依赖[Qwen官方仓库](https://huggingface.co/Qwen)。onnx模型导出和转bmodel模型推荐在mlir部分提供的docker中完成。

**注意：** 

- 编译模型需要在x86主机完成。
- Qwen-7B官方库50G左右，转模型需要保证运行内存至少40G以上，导出onnx模型需要存储空间100G以上，请确有足够的内存完成对应的操作。

## 2. 主要步骤

模型编译前需要安装TPU-MLIR。安装好后需在TPU-MLIR环境中进入例程目录。先导出onnx，然后使用TPU-MLIR将onnx模型编译为BModel。编译的具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)相应版本的SDK中获取)。

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

### 2.1.2. 下载并解压TPU-MLIR

    从sftp上获取TPU-MLIR压缩包
    ```bash
    pip3 install dfss --upgrade
    python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/tpu-mlir_v1.6.113-g7dc59c81-20240105.tar.gz 
    tar -xf tpu-mlir_v1.6.113-g7dc59c81-20240105.tar.gz 
    ```

### 2.1.3. 创建并进入docker

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
此镜像仅onnx模型导出和编译量化模型，程序编译和运行请在开发和运行环境中进行。更多TPU-MLIR的教程请参考[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)的《TPU-MLIR快速入门手册》和《TPU-MLIR开发参考手册》。

### 2.2 获取onnx

### 2.2.1 下载Qwen官方代码

**注：** Qwen-7B官方库50G左右，在下载之前，要确认自己有huggingface官网的access token或者SSH key。（Qwen-1.8B / Qwen-14B的操作相同，请保证满足对应内存需求）,以下代码以Qwen-7B为例

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen-7B-Chat
```
如果git clone完代码之后出现卡住，可以尝试`ctrl+c`中断，然后进入仓库运行`git lfs pull`。

如果无法从官网下载，也可以下载我们之前下好的，压缩包20G左右
```bash
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:LLM/Qwen-7B-Chat.zip
unzip Qwen-7B-Chat.zip
```

### 2.1.2 修改官方代码：
本例程的`tools`目录下提供了修改好之后的`config.json`和`modeling_qwen.py`。可以直接替换掉原仓库的文件：

```bash
cp tools/Qwen-7B-Chat/config.json Qwen-7B-Chat/
cp tools/Qwen-7B-Chat/modeling_qwen.py Qwen-7B-Chat/
```
(Qwen-14B, Qwen-1.8B对应文件均已放在`tools`下)

以下是对应的六处修改（包含config.json 和 modeling_qwen.py：
- config.json:
#### 1. 调整`config.json`文件中参数配置

```json
  "bf16": true,
  "max_position_embeddings": 512,
  "seq_length": 512,
```

- modeling_qwen.py:
(由于移植时间较早，下列修改点可能已经被官方完成)

1) 第一点修改如下（这是因为TORCH2的算子转ONNX会失败）：

    ``` python
    # SUPPORT_TORCH2 = hasattr(torch, '__version__') and int(torch.__version__.split(".")[0]) >= 2
    SUPPORT_TORCH2 = False
    ```

2) 第二点修改如下（这是因为转ONNX，提示Shape推导失败）：

    ```python
    # attn_weights = attn_weights / torch.full(
    #     [],
    #     size_temp ** 0.5,
    #     dtype=attn_weights.dtype,
    #     device=attn_weights.device,
    # )
    attn_weights = attn_weights / (size_temp ** 0.5)
    ```

3) 第三点修改如下（这段代码全部注释掉，是因为可以直接采用`attention_mask`，避免复杂逻辑，提升性能）：

    ```python
    # if self.use_cache_quantization:
    #     query_length, key_length = query.size(-2), key[0].size(-2)
    # else:
    #     query_length, key_length = query.size(-2), key.size(-2)
    # causal_mask = registered_causal_mask[
    #     :, :, key_length - query_length : key_length, :key_length
    # ]
    # mask_value = torch.finfo(attn_weights.dtype).min
    # mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(
    #     attn_weights.device
    # )
    # attn_weights = torch.where(
    #     causal_mask, attn_weights.to(attn_weights.dtype), mask_value
    # )
    ```

4) 第四点修改如下（同上原因）：

    ``` python
    # query_length, key_length = query.size(-2), key.size(-2)
    # causal_mask = registered_causal_mask[
    #     :, :, key_length - query_length : key_length, :key_length
    # ]
    # mask_value = torch.finfo(attn_weights.dtype).min
    # mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(
    #     attn_weights.device
    # )
    # attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    ```

5) 第五点修改，将如下代码移至`if layer_past is not None:`之前：

    ``` python
    if use_cache:
        present = (key, value)
    else:
        present = None
    ```

    这是因为kv cache只用输出1个单位就可以了，不用全部输出。提升效率。

### 2.1.3 导出onnx

- 指定Qwen-7B-Chat官方仓库的python路径

```bash
# 将/workspace/Qwen-7B-Chat换成docker环境中您的Qwen-7B-Chat仓库的路径
export PYTHONPATH=/workspace/Qwen-7B-Chat:$PYTHONPATH
```

- 导出所有onnx模型，如果过程中提示缺少某些组件，直接**pip install**组件即可

```bash
# 将/workspace/Qwen-7B-Chat换成docker环境中您的Qwen-7B-Chat仓库的路径
python3 tools/export_onnx.py --model_path /workspace/Qwen-7B-Chat --onnx_path ./models/onnx
```
此时有大量onnx模型被导出到本例程中`Qwen/models/onnx`的目录。

### 2.2 bmodel编译
首先需要在mlir工具下激活环境，[mlir下载地址可参考](./Qwen_Export_Guide.md/#212-下载并解压tpu-mlir)
```bash
cd tpu-mlir_v1.6.113-g7dc59c81-20240105
source envsetup.sh
```
目前TPU-MLIR支持1684x对Qwen进行BF16(仅限Qwen-1.8B),INT8和INT4量化，使用如下命令生成bmodel。

```bash
./scripts/gen_bmodel.sh --mode int4 --name qwen-7b
```

其中，mode可以指定bf16/int8/int4，编译成功之后，模型将会存放在`models/BM1684X/`目录下。

### 2.3 准备tokenizer

如果您之前没有运行过下载脚本，那么您需要运行它以获取tokenizer。经过上面的步骤，现在你的目录下已经存在models文件夹，所以它只会下载tokenizer。
```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```