# Blip模型的导出与编译

需要通过源码来导出 onnx 文件，基于 blip 原始仓库[BLIP官方开源仓库](https://github.com/salesforce/BLIP)导出。

# 准备导出环境

需要安装对应版本的python模块，在BLIP仓库中执行
```bash
pip3 install -r requirements.txt
```

## 导出blip captioning部分

将tools/export_cap.py复制到仓库运行，导出onnx模型，这里我们使用了model_base_14M.pth，
请从https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_14M.pth下载并放置在checkpoints文件夹内（如果不存在此文件夹，请主动创建或者修改脚本内模型文件路径）。
注意这个onnx模型由于大小超过2GB，因此会生成大量`_Constant`开头的文件，为正常现象。
转换完成后，将`blip_cap.onnx`, `Constant_8841_attr__value`以及`_Constant`开头的文件放置在BLIP/models/onnx/blip_cap文件夹内用于转模型。

## 导出blip image-text matching部分

将tools/export_itm.py复制到仓库运行，导出onnx模型，这里我们使用了model_base_retrieval_coco.pth，
请从https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth下载并放置在checkpoints文件夹内。
转换完成后，将`blip_itm.onnx`文件放置在BLIP/models/onnx文件夹内用于转模型。


## 导出blip visual question answering部分

将tools/export_vqa.py复制到仓库运行，导出onnx模型，这里我们使用了model_vqa.pth，
请从https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_vqa.pth下载并放置在checkpoints文件夹内。
转换完成后，将`blip_vqa_venc.onnx`，`blip_vqa_tenc.onnx`，`blip_vqa_tdec.onnx`文件放置在BLIP/models/onnx文件夹内用于转模型。

## 转换好的onnx
若需要使用转换好的onnx文件，请执行以下命令下载
```bash
python3 -m pip install dfss --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/BLIP/onnx.zip
python3 -m dfss --url=open@sophgo.com:sophon-demo/BLIP/bert-base-uncased.zip
```
解压后onnx文件参照以上说明放置于BLIP/models/onnx中，分词器目录位置为BLIP/models/bert-base-uncased。
