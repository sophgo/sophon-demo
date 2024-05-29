[简体中文](./YOLOv5_Common_Problems.md) | [English](./YOLOv5_Common_Problems_EN.md)

# YOLOv5移植常见问题
## 1. 修改mlir文件参数
执行完`./scripts/gen_mlir.sh`脚本后，生成mlir文件的参数需要修改为实际应用值，包括anchors、class_num、net_input_w、net_input_h、nms_threshold、obj_threshold等
```
    %263 = "top.YoloDetection"(%256, %259, %262) {agnostic_nms = false, anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326], class_num = 80 : i64, keep_topk = 200 : i64, net_input_h = 640 : i64, net_input_w = 640 : i64, nms_threshold = 5.000000e-01 : f64, num_boxes = 3 : i64, obj_threshold = 5.000000e-01 : f64, version = "yolov5"} : (tensor<1x255x80x80xf32>, tensor<1x255x40x40xf32>, tensor<1x255x20x20xf32>) -> tensor<1x1x200x7xf32> loc(#loc264)
    return %263 : tensor<1x1x200x7xf32> loc(#loc)
```

## 2. 其他问题汇总
1. 前处理未对齐，例如yolov5采用灰边填充（letter box）的方式对图片进行放缩，不能直接resize；
2. 如果使用的是单输出的YOLOv5模型，解码部分对应的计算层不能量化；
3. 编译bmodel时，应当指定`--target`参数与将要部署的设备一致(BM1684/BM1684X)；
4. 如果无法正常加载模型，尝试使用`bm-smi`查看tpu状态以及利用率，如果产生异常(tpu状态为Fault或利用率100%)，请联系技术支持或者在GitHub上创建issue。
5. 精度对齐的方法可以参考[精度对齐指导](../../../docs/FP32BModel_Precise_Alignment.md)。