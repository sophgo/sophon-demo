#===----------------------------------------------------------------------===#
# YOLO-World v2 主检测模型导出 (PyTorch -> ONNX)
# 仓库: github.com/AILab-CVC/YOLO-World (v2 权重, ultralytics 导出路径)
# 输入: images [1,3,640,640] float32, txt_feats [1,80,512] float32
# 输出: output (检测张量)
# 运行: 在 flh_mlir 容器内执行
#===----------------------------------------------------------------------===#
import os
import argparse
from copy import deepcopy
import numpy as np
import torch

from ultralytics import YOLOWorld


class ModelExporter(torch.nn.Module):
    """展开 YOLO-World 的 predict 路径, 使 txt_feats 作为显式第二输入导出。"""
    def __init__(self, yoloModel, device="cpu"):
        super().__init__()
        model = yoloModel.model
        model = deepcopy(model).to(device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.float()
        model = model.fuse()
        self.model = model
        self.device = device

    def forward(self, x, txt_feats):
        return self.model.predict(x, txt_feats=txt_feats)

    def export(self, output_dir, model_name, img_w, img_h, num_classes):
        x = torch.randn(1, 3, img_h, img_w, requires_grad=False).to(self.device)
        txt_feats = torch.randn(1, num_classes, 512, requires_grad=False).to(self.device)
        print(f"[export] images={tuple(x.shape)} txt_feats={tuple(txt_feats.shape)}")

        output_path = os.path.join(output_dir, f"{model_name}.onnx")
        with torch.no_grad():
            torch.onnx.export(
                self,
                (x, txt_feats),
                output_path,
                do_constant_folding=True,
                opset_version=12,
                input_names=["images", "txt_feats"],
                output_names=["output"],
                dynamic_axes={"images": {0: "batch_size"}},
            )
        print(f"[export] saved -> {output_path}")
        return output_path


def simplify(onnx_path, out_path):
    import onnx
    from onnxsim import simplify as onnx_simplify
    print(f"[simplify] {onnx_path} -> {out_path}")
    model = onnx.load(onnx_path)
    model_sim, check = onnx_simplify(model)
    assert check, "onnxsim check failed"
    onnx.save(model_sim, out_path)
    onnx.checker.check_model(out_path)
    print(f"[simplify] ok, saved -> {out_path}")
    # 打印 IO
    for i in model_sim.graph.input:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in i.type.tensor_type.shape.dim]
        print(f"  input  {i.name}: {shape}")
    for o in model_sim.graph.output:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in o.type.tensor_type.shape.dim]
        print(f"  output {o.name}: {shape}")


def verify(onnx_path, model_name, img_w, img_h, num_classes):
    """PyTorch vs ONNX 余弦相似度比对 (>0.9999)"""
    import onnxruntime as ort
    yoloModel = YOLOWorld(model_name)
    yoloModel.set_classes([""] * num_classes)
    exporter = ModelExporter(yoloModel)
    x = torch.randn(1, 3, img_h, img_w)
    txt = torch.randn(1, num_classes, 512)
    with torch.no_grad():
        pt_out = exporter(x, txt)
        if isinstance(pt_out, (list, tuple)):
            pt_out = pt_out[0]
        pt_arr = pt_out.detach().cpu().numpy().reshape(-1)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"images": x.numpy(), "txt_feats": txt.numpy()})[0].reshape(-1)
    cos = float(np.dot(pt_arr, onnx_out) / (np.linalg.norm(pt_arr) * np.linalg.norm(onnx_out) + 1e-9))
    maxdiff = float(np.max(np.abs(pt_arr - onnx_out)))
    print(f"[verify] cosine={cos:.6f} max_abs_diff={maxdiff:.6e}")
    assert cos > 0.9999, f"cosine {cos} < 0.9999"
    print("[verify] PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="yolov8s-worldv2")
    ap.add_argument("--img_size", type=int, default=640)
    ap.add_argument("--num_classes", type=int, default=80)
    ap.add_argument("--outdir", default="../models/onnx")
    ap.add_argument("--out_name", default="yoloworld_v2")
    ap.add_argument("--no_verify", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    yoloModel = YOLOWorld(args.model_name)
    yoloModel.set_classes([""] * args.num_classes)

    exporter = ModelExporter(yoloModel)
    raw = exporter.export(args.outdir, args.model_name + "_raw", args.img_size, args.img_size, args.num_classes)
    final = os.path.join(args.outdir, f"{args.out_name}.onnx")
    simplify(raw, final)
    os.remove(raw)

    if not args.no_verify:
        verify(final, args.model_name, args.img_size, args.img_size, args.num_classes)
    print("[done] yoloworld_v2.onnx export complete")


if __name__ == "__main__":
    main()
