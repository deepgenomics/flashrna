import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from flash_rna.models.layers.conv import ConvBlock


class UNetEncoder(nn.Module):
    """UNet encoder which applies a downsampling by a factor of 2 and a ConvBlock"""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel_size: int,
        activation: str,
        norm: str,
        dropout: float = 0.0,
        separable: bool = False,
        bias: bool = False,
    ):
        super().__init__()

        # Max pooling is used for sharper features over avg pooling that can blur
        # features
        self.downsampling = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv = ConvBlock(
            dim_in=dim_in,
            dim_out=dim_out,
            kernel_size=kernel_size,
            norm=norm,
            activation=activation,
            dropout=dropout,
            separable=separable,
            bias=bias,
        )

    def forward(
        self, x: Float[Tensor, "b d_in l"]
    ) -> Float[Tensor, "b d_out l_downsampled"]:
        x = self.downsampling(x)
        x = self.conv(x)

        return x


class UNetDecoder(nn.Module):
    """UNet decoder which applies the following operations:
        1. Upsamples the input by a factor of 2
        2. Applies a pointwise ConvBlock
        3. Adds a skip connection after applying a linear projection to match the
        dimension
        4. Applies the final separable ConvBlock

    This decoder is unconventional in that it keeps the dimension constant, rather
    than reducing after upsampling and skip connection is added, rather than
    concatentation.

    Also, a pointwise convolution is applied before upsampling and a separable
    convolution is applied after the skip connection. This results in depthwise
    separable convolution when decoder blocks are stacked.
    """

    def __init__(
        self,
        dim: int,
        dim_skip: int,
        kernel_size: int,
        activation: str,
        norm: str,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        self.conv_pointwise = ConvBlock(
            dim_in=dim,
            dim_out=dim,
            kernel_size=1,
            norm=norm,
            activation=activation,
            dropout=dropout,
            bias=bias,
        )

        self.upsample = nn.Upsample(scale_factor=2)

        self.conv_skip = ConvBlock(
            dim_in=dim_skip,
            dim_out=dim,
            kernel_size=1,
            norm=norm,
            activation=activation,
            dropout=dropout,
            bias=bias,
        )

        self.conv_separable = ConvBlock(
            dim_in=dim,
            dim_out=dim,
            kernel_size=kernel_size,
            norm=norm,
            activation=activation,
            dropout=dropout,
            separable=True,
            bias=bias,
        )

    def forward(
        self, x: Float[Tensor, "b d l"], x_skip: Float[Tensor, "b d l"]
    ) -> Float[Tensor, "b d l"]:
        x = self.upsample(x)
        x = self.conv_pointwise(x)

        x += self.conv_skip(x_skip)

        x = self.conv_separable(x)

        return x


class UNetModule(nn.Module):
    """UNet module wrapping around a trunk (often a transformer based module)

    The final output is of shape (batch, dim_input, L). Input is padded to ensure
    that L is a multiple of total_pool_size but the returned output is cropped to
    the original length L, removing the padded regions.
    """

    def __init__(
        self,
        trunk: nn.Module,
        num_downsampling: int,
        dim_input: int,
        dim_trunk: int,
        encoder_kernel_size: int = 5,
        decoder_kernel_size: int = 3,
        activation: str = "gelu_tanh",
        norm: str = "group_32",
        dropout: float = 0.0,
        # Generally not necessary to have bias, especially with normalization and
        # residual connections
        bias: bool = False,
    ):
        super().__init__()

        if encoder_kernel_size % 2 == 0 or decoder_kernel_size % 2 == 0:
            raise ValueError(
                f"Only odd kernel sizes are supported, got {encoder_kernel_size} "
                f"and {decoder_kernel_size}. This constraint is applied to make "
                "the cropping logic easier to implement."
            )

        self.downsampling = nn.MaxPool1d(kernel_size=2, stride=2)
        self.trunk = trunk

        # num_downsampling - 1 encoders apply downsampling and an additional
        # downsampling is applied right after the encoders, before the trunk
        self.total_pool_size = 2**num_downsampling

        # UNet channels are geometric progression from dim_input to dim_hidden,
        # rounded to the nearest multiple of 128. This is to ensure that at each
        # level, embedding dimension can be divided by 128 and be compatible with
        # group norm, which often uses a group dimension of 32, 64, or 128.
        unet_channels = np.geomspace(dim_input, dim_trunk, num=num_downsampling)
        unet_channels = (128 * np.round(unet_channels / 128)).astype(np.int32).tolist()

        conv_block_kwargs = dict(
            norm=norm,
            activation=activation,
            bias=bias,
            dropout=dropout,
        )

        self.encoders = nn.ModuleList(
            [
                UNetEncoder(
                    dim_in=d_in,
                    dim_out=d_out,
                    kernel_size=encoder_kernel_size,
                    **conv_block_kwargs,
                )
                for d_in, d_out in zip(unet_channels[:-1], unet_channels[1:])
            ]
        )

        unet_decoder_channels = list(reversed(unet_channels))
        self.decoders = nn.ModuleList(
            [
                UNetDecoder(
                    dim=dim_trunk,
                    dim_skip=d_skip,
                    kernel_size=decoder_kernel_size,
                    **conv_block_kwargs,
                )
                for d_skip in unet_decoder_channels
            ]
        )

        assert len(self.encoders) + 1 == len(
            self.decoders
        ), "There should be one more decoder than encoder"

    def forward(
        self,
        input: Float[Tensor, "b d l"],
        transpose_for_trunk: bool = True,
    ):
        """Forward pass for the UNet.

        Args:
            input: Input tensor of shape (batch_size, hidden_dim, length)
            transpose_for_trunk: Whether to apply transpose right before the trunk
                to convert the embedding shape to (batch_size, length, hidden_dim).
                This is necessary if this is the shape the trunk expects. After the
                trunk, the transpose is undone.

        Returns:
            output: Output tensor of shape (batch_size, hidden_dim, length)
        """
        assert (
            len(input.shape) == 3
        ), f"Input shape must be (batch_size, hidden_dim, length), got {input.shape}"

        L = input.shape[-1]

        # Pad input so that sequence length is divisible by total pool size
        padding_size = (
            0,
            (
                0
                if L % self.total_pool_size
                == 0  # no padding needed if already divisible
                else self.total_pool_size - L % self.total_pool_size
            ),
        )
        x = F.pad(input, padding_size, mode="constant", value=0)

        # Padded input x is used as a skip connection for reconstructing the original
        # input length
        skips = [x]

        # Downsample
        for encoder in self.encoders:
            x = encoder(x)  # downsample -> conv
            skips.append(x)

        x = self.downsampling(x)  # final downsampling before trunk

        if transpose_for_trunk:
            x = x.mT  # (b, d, l) -> (b, l, d)

        x = self.trunk(x)

        if transpose_for_trunk:
            x = x.mT  # (b, l, d) -> (b, d, l)

        # Upsample
        for decoder in self.decoders:
            skip = skips.pop()
            x = decoder(x, skip)

        assert not skips

        # Crop to original sequence length
        output = x[:, :, :L]

        return output
