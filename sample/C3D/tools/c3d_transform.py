#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import torch
import torch.onnx
import os
import torch.nn as nn
import numpy as np

class C3D(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, ckpt, num_classes=101, pretrained=True):
        super(C3D, self).__init__()

        self.ckpt = ckpt
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc_cls = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        x = x.view(-1, 8192)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)

        logits = self.fc_cls(x)

        return logits
    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {
                        # Conv1
                        "backbone.conv1a.conv.weight": "conv1.weight",
                        "backbone.conv1a.conv.bias": "conv1.bias",
                        # Conv2
                        "backbone.conv2a.conv.weight": "conv2.weight",
                        "backbone.conv2a.conv.bias": "conv2.bias",
                        # Conv3a
                        "backbone.conv3a.conv.weight": "conv3a.weight",
                        "backbone.conv3a.conv.bias": "conv3a.bias",
                        # Conv3b
                        "backbone.conv3b.conv.weight": "conv3b.weight",
                        "backbone.conv3b.conv.bias": "conv3b.bias",
                        # Conv4a
                        "backbone.conv4a.conv.weight": "conv4a.weight",
                        "backbone.conv4a.conv.bias": "conv4a.bias",
                        # Conv4b
                        "backbone.conv4b.conv.weight": "conv4b.weight",
                        "backbone.conv4b.conv.bias": "conv4b.bias",
                        # Conv5a
                        "backbone.conv5a.conv.weight": "conv5a.weight",
                        "backbone.conv5a.conv.bias": "conv5a.bias",
                         # Conv5b
                        "backbone.conv5b.conv.weight": "conv5b.weight",
                        "backbone.conv5b.conv.bias": "conv5b.bias",
                        # fc6
                        "backbone.fc6.weight": "fc6.weight",
                        "backbone.fc6.bias": "fc6.bias",
                        # fc7
                        "backbone.fc7.weight": "fc7.weight",
                        "backbone.fc7.bias": "fc7.bias",
                        # fc_cls
                        "cls_head.fc_cls.weight": "fc_cls.weight",
                        "cls_head.fc_cls.bias": "fc_cls.bias"
                        }

        p_dict = torch.load(self.ckpt)
        s_dict = self.state_dict()
        for name in p_dict['state_dict']:
            if name not in corresp_name:
                continue
            print(name)
            print(np.shape(p_dict['state_dict'][name]))    
            s_dict[corresp_name[name]] = p_dict['state_dict'][name]
        print(s_dict.keys())
        self.load_state_dict(s_dict)
    
def pth_to_onnx(input, checkpoint, onnx_path, input_names=['input'], output_names=['output'], device='cpu'):
    if not onnx_path.endswith('.onnx'):
        print('Warning! The onnx model name is not correct,\
              please give a name that ends with \'.onnx\'!')
        return 0

    model = C3D(ckpt=checkpoint) #导入模型
    model.eval()
    # model.to(device)
    
    torch.onnx.export(model, 
                      input, 
                      onnx_path, 
                      verbose=True, 
                      input_names=input_names, 
                      output_names=output_names,
                      dynamic_axes={
                          input_names[0]:{0: 'batch_size', 3: 'in_height', 4: 'in_width'},
                          output_names[0]:{0: 'batch_size'}}) #指定模型的输入，以及onnx的输出路径
    print("Exporting .pth model to onnx model has been successful!")

def pth_to_pt(input, checkpoint, pt_path):
    torch.set_grad_enabled(False)
    # net and model
    net = C3D(ckpt=checkpoint)
    net.eval()
    width = 112
    height = 112
    channels = 3
    output1 = net.forward(input)
    print('output1',output1)
    model = torch.jit.trace(net, input)
    output2 = model.forward(input)
    #对比模型输出结果
    print('output2',output2)
    torch.jit.save(model, pt_path)
    print("Exporting .pth model to torchscript model has been successful!")
    
if __name__ == '__main__':
    checkpoint = '/home/lihengfang/work/open-source/c3d_sports1m_16x1x1_45e_ucf101_rgb_20201021-26655025.pth'
    onnx_path = '../models/onnx/c3d_ucf101.onnx'
    pt_path = '../models/torch/c3d_ucf101.pt'
    
    if not os.path.exists("../models"):
        os.mkdir("../models")
    if not os.path.exists("../models/onnx"):
        os.mkdir("../models/onnx")
    if not os.path.exists("../models/torch"):
        os.mkdir("../models/torch")
    input = torch.randn(1, 3, 16, 112, 112)
    pth_to_onnx(input, checkpoint, onnx_path)
    pth_to_pt(input, checkpoint, pt_path)

