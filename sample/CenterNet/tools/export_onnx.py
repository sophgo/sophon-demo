from model import create_model, load_model
import torch

if __name__ == "__main__":
    num_classes = 80
    head_conv = 256
    heads = {"hm": num_classes, "wh": 2, "reg": 2}
    device = "cpu"
    load_model_path = "../models/torch/ctdet_coco_dlav0_1x.pth"
    save_script_1b_onnx = "../models/onnx/centernet_1b.onnx"
    save_script_4b_onnx = "../models/onnx/centernet_4b.onnx"

    model = create_model("dlav0_34", heads, head_conv)
    model = load_model(model, load_model_path)
    model = model.to(device)
    model.eval()
    input_var_1b = torch.zeros([1, 3, 512, 512], dtype=torch.float32)
    torch.onnx.export(model, (input_var_1b), save_script_1b_onnx)
    print("{} exported".format(save_script_1b_onnx))

    model1 = create_model("dlav0_34", heads, head_conv)
    model1 = load_model(model1, load_model_path)
    model1 = model1.to(device)
    model1.eval()
    input_var_4b = torch.zeros([4, 3, 512, 512], dtype=torch.float32)
    torch.onnx.export(model1, (input_var_4b), save_script_4b_onnx)
    print("{} exported".format(save_script_4b_onnx))
