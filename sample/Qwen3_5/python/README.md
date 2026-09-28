# Python例程

## 目录

- [Python例程](#python例程)
  - [目录](#目录)
  - [1. 环境准备](#1-环境准备)
    - [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
    - [1.2 SoC平台](#12-soc平台)
  - [2. 推理测试](#2-推理测试)
    - [2.1 参数说明](#21-参数说明)
      - [一张图片占多少Token ?](#一张图片占多少token-)
      - [视频占多少Token ?](#视频占多少token-)
    - [2.2 使用方式](#22-使用方式)
    - [2.3 固定文本前缀缓存测试](#23-固定文本前缀缓存测试)
    - [2.4 Web UI 例程](#24-web-ui-例程)

Qwen3.5能够输入单一图片/视频进行对话，python目录下提供了例程，具体情况如下：

| 序号  |  Python例程       |            说明                 |
| ---- | ---------------- | ------------------------------ |
|   1  | qwen3_5.py       | 使用SAIL推理（命令行交互）        |
|   2  | qwen3_5_prefix_cache.py | 固定文本前缀缓存推理（文字固定、图片变化场景） |
|   3  | webui.py + webui.html | Web UI 例程，浏览器访问，支持流式输出/图片视频上传/停止生成（需额外安装 flask） |

> **注意：**
> 35B-A3B (MoE) 模型与dense模型使用相同的Python推理代码，`qwen3_5.py` 从bmodel自动适配模型层数、hidden_size等参数，无需任何代码改动。35B模型仅需4核编译运行（不支持1core/1dev）。

## 1. 环境准备
> **注意：**
> 无论哪个环境，都要求transformers==5.7.0，该版本要求python版本大于3.10。若不满足，请参考[python3.10安装](../../../docs/FAQ.md#13-se7安装python310)安装。

### 1.1 x86/arm PCIe平台

- 需要**SDK v24.04.01及其以上版本**

- 如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

- 此外您可能还需要安装其他库：

```bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

- 您还需要安装sophon-sail，由于本例程需要的sophon-sail版本较新，可以用如下命令安装sophon-sail。

```bash
python3 -m dfss --install sail
```

### 1.2 SoC平台

- BM1684X 需要**SDK v24.04.01及其以上版本**

  如果您使用BM1684X的SoC平台（如SE7、SM7系列边缘设备），并使用它测试本例程，请使用**SDK V24.04.01及其以上版本**对应的刷机包进行刷机，刷机成功后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。

- BM1688 需要**SDK V2.2及其以上版本**

  如果您使用BM1688的SoC平台（如SE9、SM9系列边缘设备），并使用它测试本例程，请使用**SDK V2.2及其以上版本**对应的刷机包进行刷机，刷机成功后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。

- CV84X6（如SE13-64）使用出厂预装的运行库即可（实测环境：libsophon-0.4.13、sophon-sail 3.11.0），sophon-sail的安装方式见下文`python3 -m dfss --install sail`。


- 此外您可能还需要安装其他库：

```bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
``` 
- 本例程依赖sophon-sail，可直接安装sophon-sail，执行如下命令：

```bash
python3 -m dfss --install sail
```

## 2. 推理测试

python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。

### 2.1 参数说明

```bash
usage: qwen3_5.py [-h] -m MODEL_PATH [-c CONFIG_PATH] [-vr VIDEO_RATIO] [-d DEVID] [-ll {DEBUG,INFO,WARNING,ERROR}] 

options:
  -h, --help            show this help message and exit
  -m MODEL_PATH, --model_path MODEL_PATH
                        path to the bmodel file
  -c CONFIG_PATH, --config_path CONFIG_PATH
                        path to the processor file
  -vr VIDEO_RATIO, --video_ratio VIDEO_RATIO
                        Set video ratio, default is 0.25
  -d DEVID, --devid DEVID
                        device ID to use
  -ll {DEBUG,INFO,WARNING,ERROR}, --log_level {DEBUG,INFO,WARNING,ERROR}
                        log level, default: INFO, option[DEBUG, INFO, WARNING, ERROR]
```


#### 一张图片占多少Token ?

计算公式 $ token数 = 长 \times 宽 \div 32 \div 32 $
比如768x768尺寸图片占token数为576 token

#### 视频占多少Token ?

本例中视频尺寸默认为图片的1/4，比如768x768情况下取尺寸384x384，也就是每两帧(`temporal_patch_size`)占144个token。

默认每秒1帧。

20秒视频取20帧，总token数为 $ 144 \times 20 \div 2 = 1440 $


### 2.2 使用方式


输入`../datasets/test.jpg`测试图片，测试问题为："请描述图片中的内容"，测试命令如下:
```bash
python3 qwen3_5.py -m ../models/BM1684X/qwen3.5-2b-int4-autoround_w4bf16_seq2048_bm1684x_1dev_dynamic_20260415_111517.bmodel -c config/ -d 0
```

```
在Question: 处输入问题，在Image or Video Path: 处输入图片路径（如`test.jpg`），直接回车（不输入路径）则为纯文本对话。图片/视频路径只需输入一次，后续问题默认沿用上一次的附件。

终端将打印FTL、TPS性能数据，并输出回答结果，接着可进一步对图片或者视频进行提问，输入q即可退出。

> **注意：**
> 1. 命令行 `input()` 一次只读取一行：粘贴多行长文本时，后续行会泄漏到接下来的提问里。遇到空行会直接跳过并提示；要粘贴多行文本请先用 Web UI（见 2.4 节），或将文本合并为一行。
> 2. 推理过程中按 Ctrl-C 或发生推理异常时，会自动清空历史以恢复，不会卡死在损坏的 KV 状态上。

> **测试说明**：  
> 1. 图片或者视频尺寸越大，一般精度越高，直到达到一定尺寸，较大输入需要上下文较长的模型。

### 2.3 固定文本前缀缓存测试

针对**文字固定、图片变化**的场景（如对同一批图片反复提问同一个问题），提供[qwen3_5_prefix_cache.py](./qwen3_5_prefix_cache.py)：提问的文本部分只预填充一次并快照 KV/线性状态，之后每张图片仅需执行视觉塔和图片 token + 尾部文本的预填充。

**要求**：bmodel 必须是 `--use_history_kv` 编译的版本（含 `block_kv_<i>` 子图，编译方法见[README](../README.md)第 4.2 节），普通 bmodel 运行会直接报错提示。

```bash
usage: qwen3_5_prefix_cache.py [-h] -m MODEL_PATH [-c CONFIG_PATH] [-vr VIDEO_RATIO] [-d DEVID]
                               --question QUESTION --images IMAGES [IMAGES ...]
                               [--max_tokens MAX_TOKENS] [--do_sample] [-ll {DEBUG,INFO,WARNING,ERROR}]

options:
  -m MODEL_PATH, --model_path MODEL_PATH
                        path to the bmodel file (must be --use_history_kv build)
  -c CONFIG_PATH, --config_path CONFIG_PATH
                        path to the processor config dir
  -vr VIDEO_RATIO, --video_ratio VIDEO_RATIO
                        Set video ratio, default is 0.25
  -d DEVID, --devid DEVID
                        device ID to use
  --question QUESTION   fixed text question used for every image
  --images IMAGES       image paths to query one by one
  --max_tokens MAX_TOKENS
                        max new tokens per image, default 50
  --do_sample           enable sampling (default greedy)
  -ll {DEBUG,INFO,WARNING,ERROR}, --log_level {DEBUG,INFO,WARNING,ERROR}
                        log level, default: INFO, option[DEBUG, INFO, WARNING, ERROR]
```

使用`../datasets/test.jpg`测试图片，测试问题为："请描述图片中的内容"，测试命令如下:
```bash
python3 qwen3_5_prefix_cache.py -m ../models/BM1684X/qwen3.5-9b-int4-autoround_w4bf16_seq2048_bm1684x_1dev_history_dynamic_xxx.bmodel -c config/ --question "请描述图片中的内容" --images ../datasets/test.jpg ../datasets/test.jpg
```

```
程序启动后先对固定文本前缀做一次预填充（一次性开销，约零点几秒），随后逐张推理 --images 指定的图片：
每张图片打印回答内容、总输入 token 数（前缀 + 图片/尾部）、FTL(cached)（视觉塔 + 剩余预填充）和 TPS，
最后汇总输出除首张外的平均 FTL(cached) 与平均 TPS。换一组图片只需换 --images 参数，前缀快照继续复用。
```

> **测试说明**：  
> 1. 首次前缀预填充为一次性开销，图片数量越多摊销越划算；
> 2. 固定文本越长（长系统提示词/固定文档），缓存节省的预填充计算越多，收益越大；
> 3. 吞吐量（decode 速度）不受前缀缓存影响。

### 2.4 Web UI 例程

`webui.py` 在 `qwen3_5.py` 的推理引擎外套了一个极简的 Flask Web 服务，浏览器即可对话：流式输出、图片/视频上传、停止生成、清空对话、性能数据（FTL/TPS/Vision 耗时）展示。相比命令行交互，输入框天然支持多行文本粘贴（`input()` 一次只读一行，粘贴多行长文本会串行）。

**额外依赖**（其余依赖与 `qwen3_5.py` 相同）：

```bash
pip3 install flask -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**启动**（参数与 `qwen3_5.py` 一致，额外多 `--host`/`--port`）：

```bash
python3 webui.py -m ../models/BM1684X/qwen3.5-2b-int4-autoround_w4bf16_seq2048_bm1684x_1dev_dynamic_20260415_111517.bmodel -c config/ -d 0 --host 0.0.0.0 --port 8000
```

启动后在浏览器打开 `http://<设备IP>:8000` 即可使用。如果设备不在本地网段（例如在跳板机后面），可在本地做 SSH 端口转发后访问 `http://localhost:8000`：

```bash
ssh -L 8000:<设备IP>:8000 <user>@<跳板机IP>
```

**界面功能**：

- 直接输入问题，Enter 发送 / Shift+Enter 换行；点 📎 上传图片或视频后再提问（附件对后续问题保持有效，可点 ✕ 移除）；
- 回答流式输出，每条回答底部显示 FTL、TPS、总 token 数、Vision 耗时；
- 生成中可点「停止」中断（在 token 之间停止，KV 历史保持一致，可接着继续对话）；
- 侧栏显示模型名、seq_len、历史 token 数，「清空对话」按钮重置会话。

**HTTP 接口**（返回 NDJSON 流，可直接被其他程序调用）：

| 接口 | 说明 |
| ---- | ---- |
| `GET /api/info` | 模型名、seq_len、是否支持 history |
| `POST /api/upload` | multipart `file` 上传图片/视频，返回 `path` |
| `POST /api/chat` | `{"question": str, "media_path": str}`，NDJSON 流式返回 `delta/info/stats/error/done` 事件 |
| `POST /api/stop` | 请求停止当前生成 |
| `POST /api/clear` | 清空对话历史 |

> **注意：**
> 1. 服务为单会话设计：同一时刻只处理一轮生成，生成中再发 `/api/chat` 会返回 409；
> 2. 监听 `0.0.0.0` 时请自行确认网络环境可信，本例程未做鉴权。
