#!/usr/bin/env python3
"""Run FearTracker ONNX on host to produce reference bboxes for board comparison."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "python"))
import numpy as np
import cv2
import onnxruntime as ort

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("fear_tracker", os.path.join(base, "python", "fear_tracker.py"))
mod = importlib.util.module_from_spec(spec)
# prevent main() from running
spec.loader.exec_module(mod)

VIDEO = sys.argv[1] if len(sys.argv) > 1 else os.path.join(base, "datasets", "test.mp4")
BBOX = [163, 53, 45, 174]

sess = ort.InferenceSession(os.path.join(base, "models", "onnx", "feartracker.onnx"), providers=["CPUExecutionProvider"])
iname = [i.name for i in sess.get_inputs()]
oname = [o.name for o in sess.get_outputs()]
print("inputs:", iname, "outputs:", oname)

class OrtTracker(mod.FEARTracker):
    def __init__(self):
        self.config = dict(mod.DEFAULT_CONFIG)
        self.state = mod.TrackingState()
        grid_x, grid_y = mod.make_grid(self.config["score_size"], self.config["total_stride"], self.config["instance_size"])
        self.grid_x, self.grid_y = grid_x, grid_y
        self.window = self._make_window(self.config["windowing"], self.config["score_size"])
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.template_img = None
    def infer(self, template, search):
        outs = sess.run(None, {iname[0]: template, iname[1]: search})
        return outs[0], outs[1]
    def update(self, image):
        search_crop, search_bbox, padded_bbox = mod.get_extended_crop(
            image=image, bbox=self.state.bbox,
            crop_size=self.config["instance_size"],
            offset=self.config["search_context"],
            padding_value=self.state.mean_color,
        )
        self.state.mapping = padded_bbox
        self.state.prev_size = search_bbox[2:]
        search_img = self._preprocess(search_crop)
        bbox_pred, cls_pred = self.infer(self.template_img, search_img)
        pred_bbox, _ = self._postprocess(bbox_pred, cls_pred)
        pred_bbox = self._rescale_bbox(pred_bbox, self.state.mapping)
        pred_bbox = mod.clamp_bbox(pred_bbox, image.shape)
        self.state.bbox = pred_bbox
        return pred_bbox

cap = cv2.VideoCapture(VIDEO)
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()
print("frames:", len(frames))

tr = OrtTracker()
tr.initialize(frames[0], np.array(BBOX))
ref = [np.array(BBOX)]
for i, f in enumerate(frames[1:], 1):
    bb = tr.update(f)
    ref.append(bb)
    if i % 50 == 0:
        print(f"  frame {i}: bbox={bb.tolist()}")

np.save(os.path.join(base, "ref_bboxes.npy"), np.stack([np.array(b, dtype=np.float32) for b in ref]))
print("saved ref_bboxes.npy", len(ref))
