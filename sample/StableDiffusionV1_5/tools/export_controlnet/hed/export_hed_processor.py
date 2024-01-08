import torch
from controlnet_aux import HEDdetector

hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')

def export_hed_processor():
    hed_processor = hed.netNetwork
    hed_processor.eval()

    for parameter in hed_processor.parameters():
        parameter.requires_grad=False

    input = torch.randn(1,3, 512, 512)

    def build_hed_flow(input):
        with torch.no_grad():
            out =hed_processor(input)[0]
        return out
    traced_model=torch.jit.trace(build_hed_flow, (input))
    traced_model.save("hed_processor.pt")

export_hed_processor()