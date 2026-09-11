#!/usr/bin/env python3
"""Run FearTracker bmodel on board and compare with ONNX reference bboxes."""
import sys, os, time
import numpy as np
import cv2

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(base, "python"))
import importlib.util
spec = importlib.util.spec_from_file_location("fear_tracker", os.path.join(base, "python", "fear_tracker.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

import sophon.sail as sail

VIDEO = sys.argv[1] if len(sys.argv) > 1 else os.path.join(base, "datasets", "test.mp4")
BMODEL = sys.argv[2] if len(sys.argv) > 2 else os.path.join(base, "models_bm1684x2", "feartracker_fp16_1b.bmodel")
BBOX = [163, 53, 45, 174]
REF = np.load(sys.argv[3] if len(sys.argv) > 3 else "ref_bboxes.npy")

cap = cv2.VideoCapture(VIDEO)
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()
print("frames:", len(frames), "ref:", REF.shape)

tr = mod.FEARTracker(BMODEL, 0)
tr.initialize(frames[0], np.array(BBOX))

pred = [np.array(BBOX)]
times = []
for i, f in enumerate(frames[1:], 1):
    t0 = time.time()
    bb = tr.update(f)
    times.append(time.time() - t0)
    pred.append(bb)

pred = np.stack([np.array(b, dtype=np.float32) for b in pred])
# IoU per frame
x1 = np.maximum(pred[:,0], REF[:,0]); y1 = np.maximum(pred[:,1], REF[:,1])
x2 = np.minimum(pred[:,0]+pred[:,2], REF[:,0]+REF[:,2]); y2 = np.minimum(pred[:,1]+pred[:,3], REF[:,1]+REF[:,3])
inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
union = pred[:,2]*pred[:,3] + REF[:,2]*REF[:,3] - inter
iou = inter / np.maximum(union, 1e-6)
print(f"mean IoU vs ONNX ref: {iou.mean():.4f}  (min {iou.min():.4f})")
print(f"frames IoU>0.5: {(iou>0.5).mean()*100:.1f}%  IoU>0.75: {(iou>0.75).mean()*100:.1f}%")
print(f"avg time: {np.mean(times)*1000:.2f} ms/frame")
