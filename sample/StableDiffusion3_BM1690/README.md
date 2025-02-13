[简体中文](./README.md)

# StableDiffusion3_for_BM1690

## 目录

- [StableDiffusion3_for_BM1690](#StableDiffusion3_for_BM1690)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 环境准备](#2-环境准备)
  - [3. 例程测试](#3-例程测试)
    - [3.1 例程运行方法](#31-例程运行方法)
    - [3.2 测试结果说明](#32-测试结果说明)
    - [3.3 测试步骤](#33-测试步骤)
      - [3.3.1 测试基础推理时间](#331-测试基础推理时间)
      - [3.3.2 程序运行性能测试](#332-程序运行性能测试)

  
## 1. 简介
- 本例程是专门用于BM1690上[stable-diffusion-3-medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium)的文生图程序。
- 本例程提供了基于gradio的web页面，可控制正、负向提示词，控制cfg参数和随机种子。
- 本例程当前只提供基础的文生图功能，vae decoder（将latent解码为图像）采用蒸馏版的taesd模型，具体可参考https://github.com/madebyollin/taesd。

## 2. 环境准备

从SDK目录里获取sophon-sail压缩包，根据README自行编译，或者按照python版本选择安装已经编译好的whl文件。

*`注意：如果已经安装过适配BM1690的sophon-sail，并且已有基于BM1690的模型可以跳过此步骤`*

```shell
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
# 安装dfss，若已安装请跳过
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
# 下载模型
./download.sh
# 安装第三方库
pip3 install -r requirements.txt
```

下载的模型包括：
```
./clip_l.bmodel		# 使用TPU-MLIR编译的clip_l.bmodel, 用于文本编码, batch size = 1
./clip_g.bmodel       	# 使用TPU-MLIR编译的clip_g.bmodel, 用于文本编码, batch size = 1
./t5.bmodel           	# 使用TPU-MLIR编译的t5.bmodel, 用于文本编码, batch size = 1
./mmdit.bmodel          # 使用TPU-MLIR编译的mmdit.bmodel, 多模态dit, 用于更新隐变量, batch size = 1
./vae_decoder.bmodel	# 使用TPU-MLIR编译的vae_decoder.bmodel, 用于将隐变量解码为图像, batch size = 1
```

## 3. 例程测试

### 3.1 例程运行方法

执行 `python3 run.py --dev_ids 0 1 --prompt "a river flows through the forest"`，可在终端进行测试，`run.py`支持如下命令行参数：

```shell
--model_path BModel的路径
--tokenizer tokenizer的路径
--prompt clip_l的正向提示词，希望生成的图像包含的内容
--negative_prompt clip_l的负向提示词，不希望生成的图像包含的内容
--prompt_2 clip_g的正向提示词，希望生成的图像包含的内容，默认和clip_l的保持一致
--negative_prompt_2 clip_g的负向提示词，不希望生成的图像包含的内容，默认和clip_l的保持一致
--prompt_3 t5的正向提示词，希望生成的图像包含的内容，默认和clip_l的保持一致
--negative_prompt_3 t5的负向提示词，不希望生成的图像包含的内容，默认和clip_l的保持一致
--num_inference_steps 迭代次数
--guidance_scale sd的cfg参数
--dev_ids 设备号，可以传递1或者2个dev_ids，如--dev_ids 0或者--dev_ids 0 1
--seed 随机种子
```

执行`python3 web_gradio.py`可使用网页进行交互，可控制正、负提示词，cfg参数，迭代次数和随机种子，`web_gradio.py`支持如下命令行参数：
```shell
--model_path BModel的路径
--dev_ids 设备号
```
*若gradio界面无法正常显示图像*，可检查当前用户是否有`/tmp/gradio`路径的权限，增加读写权限，或删掉后由gradio默认创建即可。

### 3.2 测试结果说明

`run.py`程序运行结束，会在终端输出stable diffusion加载模型的耗时，以及整个pipeline的耗时，生成的图像保存在`result_{idx}.png`。

### 3.3 测试步骤

#### 3.3.1 测试基础推理时间
使用tpu-model-rt测试模型各个net的理论性能：

```shell
# 请根据实际情况修改要测试的bmodel路径
tpu-model-rt --bmodel mmdit.bmodel
```
测试结果中的`Launch time`就是模型各个net需要的推理时间。

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性；

#### 3.3.2 程序运行性能测试

在测试平台SC11上，参考命令：

```shell
# 双芯并行推理
python3 run.py
# 单芯推理
python3 run.py --dev_ids 0
```

测试结果如下：

| 是否双芯并行 | 系统内存(GB)  | load bmodel time(s) | pipeline (s)/20steps |
|----------- |--------------| ------------------- | -------------------- |
|    是      |     29       |        9.83         |         0.93         |
|    否      |     24       |        9.25         |         1.71         |

> **测试说明**：  
> 1. 多次测试性能会存在一定程度的波动
> 2. 本次测试所用主机为i9-14900K，测试前调整cpu频率为性能模式