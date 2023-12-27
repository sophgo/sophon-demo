import torch
from controlnet_aux import OpenposeDetector

device = torch.device("cpu")

processor= OpenposeDetector.from_pretrained('lllyasviel/ControlNet', device=device)

def export_body_processor():
    openpose_body_processor = processor.body_estimation.model
    openpose_body_processor.eval()

    for parameter in openpose_body_processor.parameters():
        parameter.requires_grad=False

    input = torch.randn(1,3,184,184).to(torch.float32).to(device)

    def build_body_flow(input):
        with torch.no_grad():
            paf, heat =openpose_body_processor(input)
        return paf, heat
    traced_model=torch.jit.trace(build_body_flow, (input))
    traced_model.save("openpose_body_processor.pt")

export_body_processor()

def export_face_processor():
    openpose_face_processor = processor.face_estimation.model
    openpose_face_processor.eval()
    for parameter in openpose_face_processor.parameters():
        parameter.requires_grad=False

    input = torch.randn(1,3,384,384).to(torch.float32).to(device)
    def build_face_flow(image):
        with torch.no_grad():
            heatmaps = openpose_face_processor(image)
        return heatmaps[0],heatmaps[1],heatmaps[2],heatmaps[3],heatmaps[4],heatmaps[5]
    traced_model=torch.jit.trace(build_face_flow, (input))
    traced_model.save("openpose_face_processor.pt")
# export_face_processor()

def export_hand_processor():
    openpose_hand_processor = processor.hand_estimation.model
    openpose_hand_processor.eval()
    for parameter in openpose_hand_processor.parameters():
        parameter.requires_grad=False

    input = torch.randn(1,3,184,184).to(torch.float32).to(device)
    def build_hand_flow(input):
        with torch.no_grad():
            out =openpose_hand_processor(input)
        return out
    traced_model=torch.jit.trace(build_hand_flow, (input))
    traced_model.save("openpose_hand_processor.pt")
export_hand_processor()