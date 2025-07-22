from typing import Tuple, Union

import torch

class BaseSubsampling(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.right_context = 0
        self.subsampling_rate = 1

    def position_encoding(self, offset: Union[int, torch.Tensor],
                          size: int) -> torch.Tensor:
        return self.pos_enc.position_encoding(offset, size)

class Conv2dSubsampling4(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """
    def __init__(self, idim: int, odim: int, dropout_rate: float):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim))
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 4
        # 6 = (3 - 1) * 1 + (3 - 1) * 2
        self.right_context = 6

    def forward(
            self,
            x: torch.Tensor,
            x_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
            torch.Tensor: positional encoding

        """
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))

        return x, x_mask[:, :, 2::2][:, :, 2::2]
    
    def infer(self, x, buffer, buffer_index, buffer_out):
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))

        return x, buffer, buffer_index, buffer_out

class Subsampling(torch.nn.Module):
    @staticmethod
    def add_arguments(group):
        """Add Subsampling common arguments."""
        group.add_argument('--subsampling-rate', default=4, type=int)
        group.add_argument('--subsampling-input-dim', default=256, type=int)
        group.add_argument('--subsampling-output-dim', default=256, type=int)
        group.add_argument('--subsampling-dropout-rate', default=0.1, type=float)

        return group
    
    def __init__(self, args):
        super().__init__()
        self.subsampling_rate = args.subsampling_rate
        self.subsampling_input_dim = args.subsampling_input_dim
        self.subsampling_output_dim = args.subsampling_output_dim
        self.subsampling_dropout_rate = args.subsampling_dropout_rate

        if self.subsampling_rate == 4:
            self.core = Conv2dSubsampling4(self.subsampling_input_dim, 
                                           self.subsampling_output_dim, 
                                           self.subsampling_dropout_rate)

    def forward(self, xs, ilens, masks):
        xs, masks = self.core(xs, masks)
        ilens = masks.squeeze(1).sum(1)
        return xs, ilens, masks
    
    def infer(self, x, buffer, buffer_index, buffer_out, pe_index):
        x, buffer, buffer_index, buffer_out = self.core.infer(x, 
                                    buffer, buffer_index, buffer_out)
        return x, buffer, buffer_index, buffer_out, pe_index
