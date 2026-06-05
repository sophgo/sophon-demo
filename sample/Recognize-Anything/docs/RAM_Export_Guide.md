# RAM模型导出

# 1. 准备工作：

运行如下命令
```bash
git clone https://github.com/xinyu1205/recognize-anything
cp tools/files/inference_ram_plus.py recogize-anything/
cp tools/files/ram_plus.py recognize-anything/ram/models/
cd recognize-anything
pip3 install -r requirements.txt
```

# 2. 下载这个权重：

https://huggingface.co/xinyu1205/recognize-anything-plus-model/blob/main/ram_plus_swin_large_14m.pth

# 3. 运行onnx导出脚本：

```bash
python3 inference_ram_plus.py --pretrained ram_plus_swin_large_14m.pth
```

会在当前文件夹下生成ram.onnx。