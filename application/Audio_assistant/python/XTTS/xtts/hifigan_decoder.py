import torch
import torchaudio
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
import sophon.sail as sail

LRELU_SLOPE = 0.1


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class PreEmphasis(nn.Module):
    def __init__(self, coefficient=0.97):
        super().__init__()
        self.coefficient = coefficient
        self.register_buffer("filter", torch.FloatTensor([-self.coefficient, 1.0]).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        assert len(x.size()) == 2

        x = torch.nn.functional.pad(x.unsqueeze(1), (1, 0), "reflect")
        return torch.nn.functional.conv1d(x, self.filter).squeeze(1)


class ResNetSpeakerEncoder(nn.Module):
    """This is copied from 🐸TTS to remove it from the dependencies."""

    # pylint: disable=W0102
    def __init__(
        self,
        input_dim=64,
        proj_dim=512,
        layers=[3, 4, 6, 3],
        num_filters=[32, 64, 128, 256],
        encoder_type="ASP",
        log_input=False,
        use_torch_spec=False,
        audio_config=None,
    ):
        super(ResNetSpeakerEncoder, self).__init__()

        self.encoder_type = encoder_type
        self.input_dim = input_dim
        self.log_input = log_input
        self.use_torch_spec = use_torch_spec
        self.audio_config = audio_config
        self.proj_dim = proj_dim

        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_filters[0])

        self.inplanes = num_filters[0]
        self.layer1 = self.create_layer(SEBasicBlock, num_filters[0], layers[0])
        self.layer2 = self.create_layer(SEBasicBlock, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self.create_layer(SEBasicBlock, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self.create_layer(SEBasicBlock, num_filters[3], layers[3], stride=(2, 2))

        self.instancenorm = nn.InstanceNorm1d(input_dim)

        if self.use_torch_spec:
            self.torch_spec = torch.nn.Sequential(
                PreEmphasis(audio_config["preemphasis"]),
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=audio_config["sample_rate"],
                    n_fft=audio_config["fft_size"],
                    win_length=audio_config["win_length"],
                    hop_length=audio_config["hop_length"],
                    window_fn=torch.hamming_window,
                    n_mels=audio_config["num_mels"],
                ),
            )

        else:
            self.torch_spec = None

        outmap_size = int(self.input_dim / 8)

        self.attention = nn.Sequential(
            nn.Conv1d(num_filters[3] * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, num_filters[3] * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
        )

        if self.encoder_type == "SAP":
            out_dim = num_filters[3] * outmap_size
        elif self.encoder_type == "ASP":
            out_dim = num_filters[3] * outmap_size * 2
        else:
            raise ValueError("Undefined encoder")

        self.fc = nn.Linear(out_dim, proj_dim)

        self._init_layers()

    def _init_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def create_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x, l2_norm=False, has_use_torch_spec=False):
        """Forward pass of the model.

        Args:
            x (Tensor): Raw waveform signal or spectrogram frames. If input is a waveform, `torch_spec` must be `True`
                to compute the spectrogram on-the-fly.
            l2_norm (bool): Whether to L2-normalize the outputs.

        Shapes:
            - x: :math:`(N, 1, T_{in})` or :math:`(N, D_{spec}, T_{in})`
        """
        # if you torch spec compute it otherwise use the mel spec computed by the AP
        if not has_use_torch_spec:
            x.squeeze_(1)
            if self.use_torch_spec:
                x = self.torch_spec(x)

        if self.log_input:
            x = (x + 1e-6).log()
        x = self.instancenorm(x).unsqueeze(1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.reshape(x.size()[0], -1, x.size()[-1])

        w = self.attention(x)

        if self.encoder_type == "SAP":
            x = torch.sum(x * w, dim=2)
        elif self.encoder_type == "ASP":
            mu = torch.sum(x * w, dim=2)
            sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5))
            x = torch.cat((mu, sg), 1)

        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        if l2_norm:
            x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x


class HifiDecoder(torch.nn.Module):
    def __init__(
        self,
        input_sample_rate=22050,
        output_sample_rate=24000,
        output_hop_length=256,
        ar_mel_length_compression=1024,
        speaker_encoder_audio_config={
            "fft_size": 512,
            "win_length": 400,
            "hop_length": 160,
            "sample_rate": 16000,
            "preemphasis": 0.97,
            "num_mels": 64,
        },
        tpu_inference_config=None,
    ):
        super().__init__()
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.output_hop_length = output_hop_length
        self.ar_mel_length_compression = ar_mel_length_compression
        self.speaker_encoder_audio_config = speaker_encoder_audio_config
        self.speaker_encoder = ResNetSpeakerEncoder(
            input_dim=64,
            proj_dim=512,
            log_input=True,
            use_torch_spec=True,
            audio_config=speaker_encoder_audio_config,
        )
        waveform_decoder_path = tpu_inference_config["waveform_decoder_path"]
        self.devid = tpu_inference_config["devid"]
        # load bmodel
        self.waveform_decoder_inference = sail.Engine(waveform_decoder_path, self.devid, sail.IOMode.SYSIO)
        self.handle = sail.Handle(0)
        self.graph_name = self.waveform_decoder_inference.get_graph_names()[0]
        self.input_names = self.waveform_decoder_inference.get_input_names(self.graph_name)
        # get output
        self.output_names = self.waveform_decoder_inference.get_output_names(self.graph_name)

    def forward(self, latents, g=None):
        """
        Args:
            x (Tensor): feature input tensor (GPT latent).
            g (Tensor): global conditioning input tensor.

        Returns:
            Tensor: output waveform.

        Shapes:
            x: [B, C, T]
            Tensor: [B, 1, T]
        """

        z = torch.nn.functional.interpolate(
            latents.transpose(1, 2),
            scale_factor=[self.ar_mel_length_compression / self.output_hop_length],
            mode="linear",
        )
        # print(z.shape)
        if z.shape[1] == 1:
            z = z.squeeze(1)
        # upsample to the right sr
        if self.output_sample_rate != self.input_sample_rate:
            z = torch.nn.functional.interpolate(
                z,
                scale_factor=[self.output_sample_rate / self.input_sample_rate],
                mode="linear",
            )
        bmodel_inputs = {self.input_names[0]: z.numpy(), self.input_names[1]: g.numpy()}
        bmodel_outputs = self.waveform_decoder_inference.process(self.graph_name, bmodel_inputs)
        o = (torch.from_numpy(bmodel_outputs[self.output_names[0]]))
        return o

    @torch.no_grad()
    def inference(self, c, g):
        """
        Args:
            x (Tensor): feature input tensor (GPT latent).
            g (Tensor): global conditioning input tensor.

        Returns:
            Tensor: output waveform.

        Shapes:
            x: [B, C, T]
            Tensor: [B, 1, T]
        """
        return self.forward(c, g=g)
