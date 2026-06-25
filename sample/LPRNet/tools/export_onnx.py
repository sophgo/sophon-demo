import torch
import LPRNet

if __name__ == "__main__":
    model = LPRNet.build_lprnet(class_num=68)
    model.load_state_dict(
        torch.load(
            "../models/torch/Final_LPRNet_model.pth", map_location=torch.device("cpu")
        )
    )
    model.eval()
    input = torch.randn(1, 3, 24, 94)
    torch.onnx.export(model, (input), "../models/onnx/lprnet_1b.onnx")
    input1 = torch.randn(4, 3, 24, 94)
    torch.onnx.export(model, (input1), "../models/onnx/lprnet_4b.onnx")
    print("finished")
