# Whisper模型的导出与编译
可以直接下载我们已经导出的onnx模型，推荐在mlir部分提供的docker中完成转bmodel模型。
**注意**：
- 编译模型需要在x86主机完成。

## 1 TPU-MLIR环境搭建
参考[TPU-MLIR环境搭建](../../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)完成搭建。

## 2 获取onnx

可以运行如下命令，从源码导出onnx:
```bash
pip3 install -r ./python/requirements.txt
./scripts/gen_onnx.sh --model base #--model可选：small/medium/small.en/distil.small.en
```

## 3 bmodel编译
目前TPU-MLIR支持1684x对Whisper进行F16量化，使用如下命令生成bmodel。
```bash
./scripts/gen_bmodel.sh --model base #--model可选：small/medium/small.en/distil.small.en
```
编译成功之后的模型放置于`./models/BM1684X/`，以base为例，最终会生成模型`bmwhisper_base_1684x_f16.bmodel`