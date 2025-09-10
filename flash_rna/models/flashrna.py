"""FlashRNA model"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float, Integer
from lightning.pytorch.core.mixins import HyperparametersMixin
from lightning.pytorch.utilities import rank_zero_info
from torch import Tensor

import wandb
from flash_rna.config import MODELS_DIR
from flash_rna.data.borzoi import BorzoiHumanTracksMapping, BorzoiMouseTracksMapping
from flash_rna.data.seqtrack import TargetTracks
from flash_rna.data.sequences import RCSequences, Sequence
from flash_rna.data.tracks_mapping import TracksMapping
from flash_rna.models.modules.transformer import TransformerModule
from flash_rna.models.modules.unet import UNetEncoder, UNetModule
from flash_rna.models.utils import OneHotEmbedding, import_class_from_path


@dataclass(frozen=True, kw_only=True)
class FlashRNA_Config:
    """Default values are chosen to match Borzoi as closely as possible"""

    model_type: str = "flashrna"

    species: Literal["hg38", "mm10", "multi-species"] = "multi-species"

    # Dict for multi-species models
    tracks_mapping: TracksMapping | Dict[str, TracksMapping] = field(
        default_factory=lambda: {
            "hg38": BorzoiHumanTracksMapping(),
            "mm10": BorzoiMouseTracksMapping(),
        }
    )  # defaults to Borzoi multi-species TracksMapping

    # Transformer config
    dim: int = 1536
    transformer_num_layers: int = 8
    head_dim: int = 192
    expansion_factor: int = 2
    transformer_norm: Literal["ln", "rms"] = "ln"
    use_rope: bool = True
    rope_base: float = 10000.0
    use_alibi: bool = False
    window_size: Tuple[int, int] = (-1, -1)
    mlp_activation: str = "gelu_tanh"
    mlp_swiglu_match_params: bool = False
    mlp_dropout: float = 0.3
    attention_dropout: float = 0.2
    post_attn_dropout: float = 0.3
    post_mlp_dropout: float = 0.3

    # Conv tower config
    conv_num_layers: int = 5
    conv_dim_in: int = 512
    conv_dim_out: int = 1280
    conv_kernel_size: int = 5
    conv_activation: str = "gelu_tanh"
    conv_norm: str = "group_32"
    stem_dropout: float = 0.0

    # UNet config
    unet_num_downsampling: int = 2
    encoder_kernel_size: int = 5
    decoder_kernel_size: int = 3
    unet_activation: str = "gelu_tanh"
    unet_norm: str = "group_32"
    unet_dropout: float = 0.0

    # Input embedding config
    embedding_conv_kernel_size: int = 15

    # Head config
    output_hidden_dim: int = 1920
    output_dropout: float = 0.1
    output_dim_human: int = 2449
    output_dim_mouse: int = 828

    # Configurations for model training
    use_flash_attn: bool = True
    prediction_crop_margin: int | None = None
    prediction_resolution: int = 32


class DownsampleConvTower(nn.Module):
    def __init__(
        self,
        num_layers: int,
        dim_in: int,
        dim_out: int,
        kernel_size: int = 5,
        activation: str = "gelu_tanh",
        norm: str = "group_64",
        dropout: float = 0.0,
        # Generally not necessary to have bias, especially with normalization and
        # residual connections
        bias: bool = False,
    ):
        super().__init__()

        if num_layers == 0:
            self.conv_tower = nn.ModuleList([nn.Identity()])
        else:
            # Channels are changed in geometric progression from dim_input to dim_out,
            # rounded to the nearest multiple of 128. This is to ensure that at each
            # level, embedding dimension can be divided by 128 and be compatible with
            # group norm, which often uses a group dimension of 32, 64, or 128.
            conv_channels = np.geomspace(dim_in, dim_out, num=num_layers + 1)
            conv_channels = (
                (128 * np.round(conv_channels / 128)).astype(np.int32).tolist()
            )

            conv_block_kwargs = dict(
                kernel_size=kernel_size,
                norm=norm,
                activation=activation,
                bias=bias,
                dropout=dropout,
            )

            self.conv_tower = nn.ModuleList(
                [
                    UNetEncoder(d_in, d_out, **conv_block_kwargs)
                    for d_in, d_out in zip(conv_channels[:-1], conv_channels[1:])
                ]
            )

    def forward(self, x: Float[Tensor, "b d l"]) -> Float[Tensor, "b d l"]:
        for layer in self.conv_tower:
            x = layer(x)

        return x


class FlashRNA(HyperparametersMixin, nn.Module):
    def __init__(
        self,
        config: FlashRNA_Config,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = self.hparams.config

        # Initial sanity checks =======================================================
        if config.species == "multi-species":
            # Expect {"hg38": HumanTracksMapping, "mm10": MouseTracksMapping}
            if not isinstance(config.tracks_mapping, dict):
                raise ValueError(
                    f"{self.__class__.__name__} model with multi-species support "
                    "requires a dictionary of species name to tracks mapping."
                )

            if set(config.tracks_mapping.keys()) != {"hg38", "mm10"}:
                raise ValueError(
                    f"Currently, {self.__class__.__name__} only supports multi-species "
                    "models for hg38 and mm10."
                )
        elif config.species == "hg38" or config.species == "mm10":
            if not isinstance(config.tracks_mapping, TracksMapping):
                raise ValueError(
                    f"{self.__class__.__name__} model with hg38 or mm10 species "
                    "support requires a TracksMapping object."
                )
        else:
            raise ValueError(f"Invalid species: {config.species}")

        if config.species == "hg38" and config.output_dim_mouse != 0:
            raise ValueError(
                "hg38 model must have a zero output dimension for mouse tracks."
            )
        elif config.species == "mm10" and config.output_dim_human != 0:
            raise ValueError(
                "mm10 model must have a zero output dimension for human tracks."
            )

        # Model layers ================================================================
        self.embedding = OneHotEmbedding(num_classes=4)

        transformer = TransformerModule(
            num_blocks=config.transformer_num_layers,
            dim=config.dim,
            head_dim=config.head_dim,
            expansion_factor=config.expansion_factor,
            norm=config.transformer_norm,
            use_rope=config.use_rope,
            rope_base=config.rope_base,
            use_alibi=config.use_alibi,
            window_size=config.window_size,
            mlp_activation=config.mlp_activation,
            mlp_dropout=config.mlp_dropout,
            mlp_swiglu_match_params=config.mlp_swiglu_match_params,
            attention_dropout=config.attention_dropout,
            post_attn_dropout=config.post_attn_dropout,
            post_mlp_dropout=config.post_mlp_dropout,
            use_flash_attn=config.use_flash_attn,
        )

        self.conv_tower = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=config.conv_dim_in,
                kernel_size=config.embedding_conv_kernel_size,
                padding="same",
                bias=False,
            ),
            DownsampleConvTower(
                num_layers=config.conv_num_layers,
                dim_in=config.conv_dim_in,
                dim_out=config.conv_dim_out,
                kernel_size=config.conv_kernel_size,
                activation=config.conv_activation,
                norm=config.conv_norm,
                dropout=config.stem_dropout,
                bias=False,
            ),
        )

        self.core = UNetModule(
            trunk=transformer,
            num_downsampling=config.unet_num_downsampling,
            dim_input=config.conv_dim_out,
            dim_trunk=config.dim,
            encoder_kernel_size=config.encoder_kernel_size,
            decoder_kernel_size=config.decoder_kernel_size,
            activation=config.unet_activation,
            norm=config.unet_norm,
            dropout=config.unet_dropout,
            bias=False,
        )

        # Final head
        self.final_joined_convs = nn.Sequential(
            nn.Linear(config.dim, config.output_hidden_dim),
            nn.Dropout(config.output_dropout),
            nn.GELU(approximate="tanh"),
        )

        # Rather than having separate heads for each species, use a unified head and
        # slice appropriately. This is more efficient and easier to implement,
        # especially with DDP (otherwise conditional computation can cause issues and
        # need find_unused_parameters which can slow down computation).
        total_output_dim = config.output_dim_human + config.output_dim_mouse
        self.unified_head = nn.Linear(config.output_hidden_dim, total_output_dim)

        # Store slice indices for each species
        self.human_slice = slice(0, config.output_dim_human)
        self.mouse_slice = slice(config.output_dim_human, total_output_dim)

        self.final_softplus = nn.Softplus()

    def predict_embeddings(
        self,
        x: Integer[Tensor, "b l"],
        cropped_length: int | None = None,
    ) -> Float[Tensor, "b l_cropped d"]:
        """Predict embeddings before the final head layer

        Args:
            x: Input sequence tensor
            cropped_length: Margin to crop out of the model predictions near the
                edges of the sequence. If None, the default from the model
                config is used. To explicitly disable cropping, set to 0.
        """
        if cropped_length is None:
            cropped_length = self.config.prediction_crop_margin

        x = self.embedding(x)

        x = rearrange(x, "b l d -> b d l")

        x = self.conv_tower(x)

        out = self.core(x)

        out = rearrange(out, "b d l -> b l d")

        out_cropped = self.center_crop(out, cropped_length=cropped_length)

        return out_cropped

    def forward(
        self,
        x: Integer[Tensor, "b l"] | Sequence,
        species: Literal["hg38", "mm10"] = "hg38",
        cropped_length: int | None = None,
        example_id: str | None = None,
    ) -> TargetTracks:
        """Forward pass

        Returns a TargetTracks dataclass which makes it easy to keep track of indices
        for each target track.

        Args:
            x: Input sequence tensor of shape (b, l) or Sequence object
            species: Species for the output predictions. Defaults to "hg38" since
                this is expected to be the most common use case
            cropped_length: Margin to crop out of the model predictions near the
                edges of the sequence. If None, the default from the model
                config is used. To explicitly disable cropping, set to 0
            example_id: Optional example ID to attach to the output

        Returns:
            TargetTracks object with predictions.
        """
        if isinstance(x, Sequence):
            x = x.tensor

        # Initial sanity checks =======================================================
        if species not in ["hg38", "mm10"]:
            raise ValueError(f"Invalid species: {species}")

        if self.config.species != "multi-species" and species != self.config.species:
            raise ValueError(
                f"This model is for {self.config.species}, but `species` set to "
                f'"{species}". Please ensure `config.species` is consistent with '
                "`species`."
            )

        # Main forward pass ===========================================================
        x = self.predict_embeddings(x, cropped_length=cropped_length)

        x = self.final_joined_convs(x)

        # Disable autocast for full precision in the final layer
        with torch.amp.autocast(device_type="cuda", enabled=False):
            # Always compute full output, then slice appropriately
            full_output = self.unified_head(x.float())

            if species == "hg38":
                output = full_output[..., self.human_slice]
            else:
                output = full_output[..., self.mouse_slice]

            output_softplus = self.final_softplus(output)

        # Handle tracks_mapping access based on species
        if self.config.species == "multi-species":
            tracks_mapping = self.config.tracks_mapping[species]
        else:
            tracks_mapping = self.config.tracks_mapping

        return TargetTracks(
            tracks=output_softplus,
            tracks_mapping=tracks_mapping,
            example_id=example_id,
        )

    def predict(
        self,
        x: Integer[Tensor, "b l"] | Sequence | RCSequences,
        cropped_length: int | None = None,
        example_id: str | None = None,
        pred_reverse_complement: bool = False,
        **kwargs,
    ) -> TargetTracks:
        """Forward pass for inference

        With `pred_reverse_complement` set to True, prediction will be merged from
        both forward and reverse complement of the input sequence.

        Args:
            x: Input sequence tensor of shape (b, l) or Sequence object or RCSequences
                object (only when `pred_reverse_complement` is True)
            cropped_length: Margin to crop out of the model predictions near the
                edges of the sequence. If None, the default from the model
                config is used.
            example_id: Optional example ID to attach to the output.
            pred_reverse_complement: Whether to predict the reverse complement of the
                input sequence.
        """
        if isinstance(x, RCSequences) and not pred_reverse_complement:
            raise ValueError(
                "RCSequences object provided but `pred_reverse_complement` is False"
            )

        if pred_reverse_complement:
            if isinstance(x, Sequence):
                x = RCSequences(x)
            elif isinstance(x, Tensor):
                x = RCSequences(Sequence(x))
            elif not isinstance(x, RCSequences):
                raise ValueError(
                    f"Invalid input type: {type(x)}, "
                    "must be Sequence, Tensor, or RCSequences"
                )

            return self._predict_with_rc(
                x,
                cropped_length=cropped_length,
                example_id=example_id,
                **kwargs,
            )
        else:
            return self.forward(
                x,
                cropped_length=cropped_length,
                example_id=example_id,
                **kwargs,
            )

    @classmethod
    def center_crop(
        cls, x: Float[Tensor, "b l d"], cropped_length: int | None
    ) -> Float[Tensor, "b l_cropped d"]:
        """Center crop a tensor by a given margin 'cropped_length',
        returning a tensor of shape '(b, l - 2 * crop_margin, d)'

        This is used to crop out model predictions near the edges of the
        sequence, where the model is expected to be worse due to limited
        sequence context.
        """
        if cropped_length is None or cropped_length == 0:
            return x

        assert len(x.shape) == 3, f"Expected (b l d), got {x.shape}"

        L = x.shape[1]
        assert (
            cropped_length > 0
        ), f"Cropped length must be positive, got {cropped_length}"
        assert (
            L - 2 * cropped_length > 0
        ), f"Cropped length must be positive, got {L - 2 * cropped_length}"

        return x[:, cropped_length:-cropped_length, :]

    @classmethod
    def from_ckpt(
        cls,
        ckpt_path: str | Path | None = None,
        wandb_artifact: str | None = None,
        wandb_filename: str = "model.ckpt",
        model_class_key: str = "model_class",
        model_config_key: str = "model_config",
        weights_prefix: str | None = None,
        load_weights: bool = True,
        device: str | torch.device | None = None,
    ) -> FlashRNA:
        """Load a model from a checkpoint file or a wandb artifact

        This assumes the model class and model config were saved under
        `hyper_parameters`. `model_class_key` is the key under which the model
        class is found in the `hyper_parameters`, and `model_config_key` is
        the key under which the model config is found.

        A model is initialized using the model class with the model config.
        If `load_weights` is False, only the model is initialized according to
        the model config without loading saved weights.

        `weights_prefix` is the prefix added to the keys of the model weights in
        the checkpoint (e.g. when the model was saved as self.model in the
        LightningModule). This is removed before loading the model weights.

        Args:
            ckpt_path: Path to the checkpoint file
            wandb_artifact: Wandb artifact name
            wandb_filename: Wandb filename in the artifact
            model_class_key: Key for the model class in the checkpoint
            model_config_key: Key for the model config in the checkpoint
            weights_prefix: Prefix for the model weights in the checkpoint
            load_weights: Whether to load the saved model weights
        """
        if ckpt_path is None and wandb_artifact is None:
            raise ValueError("Either ckpt_path or wandb_artifact must be provided")
        elif ckpt_path is not None and wandb_artifact is not None:
            raise ValueError("Only one of ckpt_path or wandb_artifact must be provided")

        if wandb_artifact is not None:
            ckpt_path = MODELS_DIR / wandb_artifact / wandb_filename
            rank_zero_info(f"Loading from wandb artifact: {wandb_artifact}")
            if not ckpt_path.exists():
                rank_zero_info(
                    f"Local checkpoint not found at {ckpt_path}. "
                    f"Downloading from wandb artifact: {wandb_artifact}"
                )
                artifact = wandb.Api().artifact(wandb_artifact)
                artifact.download(root=ckpt_path.parent)

            # If there is a wandb run, track this model artifact is consumed during
            # this run
            if wandb.run is not None:
                wandb.run.use_artifact(wandb_artifact)
        else:
            ckpt_path = Path(ckpt_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

        rank_zero_info(
            f"Loading model from {ckpt_path} (loading weights: {load_weights})"
        )

        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        model_cls = ckpt["hyper_parameters"][model_class_key]
        if isinstance(model_cls, str):
            model_cls = import_class_from_path(model_cls)
        elif not issubclass(model_cls, FlashRNA):
            raise ValueError(f"Model is not a subclass of FlashRNA: {type(model_cls)}")

        model_config = ckpt["hyper_parameters"][model_config_key]

        if not isinstance(model_cls, type) or not issubclass(model_cls, FlashRNA):
            raise ValueError(f"Model is not a subclass of FlashRNA: {type(model_cls)}")

        if not isinstance(model_config, FlashRNA_Config):
            raise ValueError(
                "Model config is not an instance of "
                f"FlashRNA_Config: {type(model_config)}"
            )

        model = model_cls(model_config)

        if load_weights:
            # If LightningModule is used, there can be additional weights_prefix added
            # depending on how the model is saved in the LightningModule.
            # e.g. if self.model = Model(...) inside the LightningModule,
            # weights_prefix will be "model."
            if weights_prefix:
                weights = {
                    k.removeprefix(weights_prefix): v
                    for k, v in ckpt["state_dict"].items()
                    if k.startswith(
                        weights_prefix
                    )  # only load weights starting with weights_prefix
                }
            else:
                weights = ckpt["state_dict"]

            model.load_state_dict(weights)

        return model

    @property
    def tracks_mapping(self) -> Dict[str, TracksMapping]:
        """Dictionary of species name to tracks mapping

        If `config.species` is "multi-species", the underlying `config.tracks_mapping`
        is a dictionary of species name to tracks mapping, but for single-species
        models, it is a TracksMapping object.

        This property always returns a dictionary of species name to tracks mapping to
        provide a consistent interface regardless of single- or multi-species training.
        """
        if self.config.species == "multi-species":
            # Should have already been validated in __init__
            assert isinstance(self.config.tracks_mapping, dict)

            return self.config.tracks_mapping

        return {self.config.species: self.config.tracks_mapping}

    @property
    def output_dim(self) -> Dict[str, int]:
        """Dictionary of species name to output dimension"""

        return {
            "hg38": self.config.output_dim_human,
            "mm10": self.config.output_dim_mouse,
        }

    def _predict_with_rc(
        self,
        x: RCSequences,
        cropped_length: int | None = None,
        example_id: str | None = None,
        **kwargs,
    ) -> TargetTracks:
        """Predict with both forward and reverse complement of the input sequence and
        merge the predictions

        Args:
            x: RCSequences object
            cropped_length: Margin to crop out of the model predictions near the
                edges of the sequence. If None, the default from the model
                config is used.
            example_id: Optional example ID to attach to the output.
        """
        if not isinstance(x, RCSequences):
            raise ValueError("x must be a RCSequences object")

        preds = self.forward(
            x.tensor,
            cropped_length=cropped_length,
            example_id=example_id,
            **kwargs,
        )

        preds_fw, preds_rc = torch.chunk(preds.tracks, 2, dim=0)

        # Flip 'preds_rc' and using 'strand_pairs' to map stranded
        # tracks to its complement to align with 'preds_fw'.
        # 'strand_pairs' is a list of track indices that swaps
        # forward and reverse complement tracks while keeping unstranded tracks
        # in the same position
        preds_rc = preds_rc.flip(-2)[:, :, preds.tracks_mapping.strand_pairs]

        preds_merged = (preds_fw + preds_rc) * 0.5

        return TargetTracks(
            tracks=preds_merged,
            tracks_mapping=preds.tracks_mapping,
            example_id=example_id,
        )
