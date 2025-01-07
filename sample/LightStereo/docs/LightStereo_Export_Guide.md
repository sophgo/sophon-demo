# LightStereo ONNX导出

注：导出onnx需要依赖cuda，如果您没有cuda，可以下载我们提供的onnx，也可以自己更改导出脚本。

本例程将OpenStereo仓库中LightStereo的modeling部分单独摘了出来，存放在`tools/core`目录下。并且提供了导出脚本`export_onnx.py`。它的使用方法如下：
```bash
cd tools
python3 export_onnx.py --loadckpt ../models/ckpt/LightStereo-S-SceneFlow.ckpt --onnx_path ../models/onnx/LightStereo-S-SceneFlow.onnx
```
`--loadckpt`对应的文件请自行从OpenStereo源仓库获取，如果您使用不同版本的模型，您还需要更改其他参数，具体请查看`export_onnx.py`脚本中的参数定义。

运行完成后，`--onnx_path`指定的目录下会生成LightStereo-S-SceneFlow.onnx，用户可以使用该onnx进行bmodel编译。

# 模型优化内容：

本例程对LightStereo的源码做了部分修改，在不影响精度的前提下，提升了在tpu上的性能，具体修改如下：

cost_volume.py:
```bash
def correlation_volume(left_features, right_features, max_disp):
    b, c, h, w = left_feature.size()
    cost_volume = left_feature.new_zeros(b, max_disp, h, w)
    for i in range(max_disp):
        if i > 0:
            cost_volume[:, i, :, i:] = (left_feature[:, :, :, i:] * right_feature[:, :, :, :-i]).mean(dim=1)
        else:
            cost_volume[:, i, :, :] = (left_feature * right_feature).mean(dim=1)
    cost_volume = cost_volume.contiguous()
    return cost_volume
```
修改为：
```bash
def correlation_volume(left_features, right_features, max_disp):
    b, _, h, w = left_features.size()
    left_features = left_features.permute(0, 1, 3, 2)
    right_features = right_features.permute(0, 1, 3, 2)
    cost_volume_list = []
    for i in range(max_disp):
      if i > 0:
        cost_volume = (left_features[:, :, i:, :] * right_features[:, :, :-i, :]).mean(dim=1, keepdim=True)
        zeros = torch.zeros(b, 1, i, h)
        cost_volume_list.append(torch.concat([zeros, cost_volume], 2))
      else:
        cost_volume = (left_features * right_features).mean(dim=1, keepdim=True)
        cost_volume_list.append(cost_volume)
    return torch.concat(cost_volume_list, 1).permute(0, 1, 3, 2).contiguous()
```

# cali_data制作方法

如果您需要转不同分辨率的int8模型，那么您需要自己制作新的量化数据集，并更改gen_int8bmodel_mlir.sh里对应的内容。可以通过多种方法制作，比较简单的方法是在lightstereo_opencv.py的LightStereo.predict函数里添加如下代码：

```bash
    def predict(self, left_imgs, right_imgs, img_num):
        input_data = {self.input_name_left: left_imgs,
                      self.input_name_right: right_imgs}
        # 需要添加部分---：
        if not self.count:
            self.count = 1
        else:
            self.count += 1
        np.savez("cali_data_480x736/"+str(self.count), **input_data)
        # ---------------
        outputs = self.net.process(self.graph_name, input_data)[self.output_name]
```

参考python/README.md运行lightstereo_opencv.py，您可以自由选择需要的量化数据集，运行完成后量化数据集就会被处理为.npz的格式，可以供mlir的run_calibration使用。