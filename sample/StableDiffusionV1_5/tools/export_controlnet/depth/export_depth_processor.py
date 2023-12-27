from transformers import DPTForDepthEstimation
import torch

device = "cpu"
dtype = torch.float32

model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
model = model.to(device)
def export_depth_processor():
    depth_processor = model.eval()
    for para in depth_processor.parameters():
        para.requires_grad = False
    input = torch.randn(1, 3, 384, 384).to(dtype).to(device)

    def build_flow(input):
        with torch.no_grad():
            predicted_depth = depth_processor(input).predicted_depth
        return predicted_depth
    traced_model = torch.jit.trace(build_flow, (input))
    traced_model.save("depth_processor.pt")

export_depth_processor()