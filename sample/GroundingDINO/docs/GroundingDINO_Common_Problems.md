# GroundingDINO模型导出
## 1. 准备工作
GroundingDINO模型导出是在Pytorch模型生产环境下进行的，需要根据[GroundingDINO官方开源仓库](https://github.com/IDEA-Research/GroundingDINO/tree/main)的要求安装好对应的环境。
由于目前TPU和Pytorch之间实现部分算子的方式不同，同时也考虑到TPU推理的效率问题，我们对在不影响模型结构的情况下对推理过程做了部分重构修改，因此TPU模型的输入个数会和Pytorch模型下的输入个数不同，但不会影响精度。
由于修改和重构的部分过多，我们将在后续直接提供修改/重构后的GroundingDINO Pytorch代码，由于GroundingDINO作者目前只公开了推理部分的代码，保留了训练和finetune的内容，因此此次影响不会对后续推理有影响。

## 2. 如何获取重构后的Pytorch代码
在适配TPU的过程当中为了充分利用TPU的性能,在不影响模型结构的前提下做了相应的优化，因此后续微调后的模型需要从`GroundingDINO_Torch`中的`export_onnx.py`转为对应的onnx模型。
我们提供已经重构后的GroundingDino Pytorch代码，用户可以通过
```bash
python3 -m dfss --url=open@sophgo.com:sophon-demo/GroundingDINO/GroundingDINO_Torch.zip
unzip GroundingDINO_Torch.zip
```
下载得到GroundingDINO_Torch及对应文件.导出onnx方式为：
```bash
./GroundingDINO_Torch/export_onnx.py --args
```
其中`--args`和源仓库中进行模型推理部分的一致,请参考[GroundingDINO官方开源仓库](https://github.com/IDEA-Research/GroundingDINO/tree/main)进行使用

