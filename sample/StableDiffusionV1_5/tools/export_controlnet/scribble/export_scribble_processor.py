import torch
from controlnet_aux import HEDdetector

scribble= HEDdetector.from_pretrained('lllyasviel/Annotators')

def export_scribble_processor():
    scribble_processor = scribble.netNetwork
    scribble_processor.eval()

    for parameter in scribble_processor.parameters():
        parameter.requires_grad=False

    input = torch.randn(1,3, 512, 512)

    def build_scribble_flow(input):
        with torch.no_grad():
            out =scribble_processor(input)[0]
        return out
    traced_model=torch.jit.trace(build_scribble_flow, (input))
    traced_model.save("scribble_processor.pt")

export_scribble_processor()