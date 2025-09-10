from pathlib import Path
from typing import List

import torch
from jaxtyping import Bool, Float, Integer
from numpy.typing import NDArray
from torch import Tensor

from flash_rna.config import (
    BORZOI_HUMAN_TARGETS,
    BORZOI_MOUSE_TARGETS,
    FLASH_RNA_HUMAN_TARGETS,
    FLASH_RNA_MOUSE_TARGETS,
)
from flash_rna.data.seqtrack import TargetTracks
from flash_rna.data.tracks_mapping import TracksMapping

# Util functions for Borzoi dataset transformations ===================================


def inverse_squash_transform(
    tracks: Float[Tensor, "l d"] | Float[Tensor, "b l d"],
    scale: Float[Tensor, " d"] | Float[NDArray, " d"],
    clip_soft: Float[Tensor, " d"] | Float[NDArray, " d"],
    sqrt_mask: Bool[Tensor, " d"] | Bool[NDArray, " d"],
) -> Float[Tensor, "b l d"]:
    """Inverse of the squash transform used in the Borzoi paper

    Adopted from Baskerville repo:
    https://github.com/calico/baskerville/blob/af9e2df999b6727d02d4fd5378c9dcdc7ac1482d/src/baskerville/dataset.py#L397

    Args:
        tracks: Tracks tensor to be untransformed
        scale: Scale factor for each track
        clip_soft: Clip soft factor for each track
        sqrt_mask: Mask for tracks that are sqrt transformed

    Returns:
        Untransformed tracks
    """
    assert tracks.ndim == 2 or tracks.ndim == 3
    assert scale.ndim == 1
    assert clip_soft.ndim == 1
    assert sqrt_mask.ndim == 1

    # Convert to Tensors if not already (for sqrt_mask, not necessary)
    if not isinstance(scale, Tensor):
        scale = torch.from_numpy(scale)
    if not isinstance(clip_soft, Tensor):
        clip_soft = torch.from_numpy(clip_soft)

    # Move to same device (for sqrt_mask, not necessary)
    scale = scale.to(device=tracks.device)
    clip_soft = clip_soft.to(device=tracks.device)

    # Reverse scaling
    tracks = tracks / scale

    # Reverse soft clip
    tracks_unclipped = clip_soft + (tracks - clip_soft) ** 2
    tracks = torch.where(tracks > clip_soft, tracks_unclipped, tracks)

    # Reverse 3 / 4 exponentiation
    tracks[..., sqrt_mask] = tracks[..., sqrt_mask] ** (4 / 3)

    return tracks


# =====================================================================================


class BorzoiTracksMappingBase(TracksMapping):
    GTEX_TISSUE_COL = "gtex_tissue"
    CLIP_SOFT_COL = "clip_soft"
    SUM_STAT_COL = "sum_stat"
    DESCRIPTION_COL = "description"

    EXPECTED_COLUMNS = TracksMapping.EXPECTED_COLUMNS + [
        GTEX_TISSUE_COL,
        CLIP_SOFT_COL,
        SUM_STAT_COL,
        DESCRIPTION_COL,
    ]

    def reverse_transform(
        self, tracks: Float[Tensor, "b l d"], log2p1: Bool[NDArray, " d"]
    ) -> Float[Tensor, "b l d"]:
        """Reverse the squash transform used in the Borzoi paper

        Args:
            tracks: Tensor of tracks to be untransformed
            log2p1: Numpy array of booleans (or equivalent list, tensor, etc.), on
                whether to apply log2p1 transform to the tracks where the boolean is
                True.

        Returns:
            Untransformed tracks
        """
        tracks_untransformed = inverse_squash_transform(
            tracks,
            scale=self.scale,
            clip_soft=self.clip_soft,
            sqrt_mask=self.sqrt_mask,
        )

        tracks_untransformed = torch.where(
            log2p1, torch.log2(tracks_untransformed + 1), tracks_untransformed
        )

        return tracks_untransformed

    def indices_by_tissue(self, tissue: str) -> Integer[NDArray, " n"]:
        if tissue not in self.gtex_tissues:
            raise ValueError(
                f"Invalid tissue: {tissue}. Available tissues: {self.gtex_tissues}"
            )

        tissue_tracks = self.metadata_df[
            self.metadata_df[self.GTEX_TISSUE_COL] == tissue
        ]

        return tissue_tracks.index.values

    @property
    def gtex_tissues(self) -> List[str]:
        """GTEX tissues in the tracks mapping"""
        if not hasattr(self, "_gtex_tissues"):
            gtex_tissues = self.metadata_df[self.GTEX_TISSUE_COL].dropna().unique()

            if len(gtex_tissues) == 0:
                raise ValueError("No GTEX tissues found in the tracks mapping")

            self._gtex_tissues = gtex_tissues.tolist()

        return self._gtex_tissues

    @property
    def clip_soft(self) -> Float[Tensor, " d"]:
        """Clip soft factor for each track"""
        return self.metadata_df[self.CLIP_SOFT_COL].values

    @property
    def sqrt_mask(self) -> Bool[NDArray, " d"]:
        """Mask for tracks that are sqrt transformed"""
        return self.metadata_df[self.SUM_STAT_COL].str.contains("_sqrt").values


class BorzoiHumanTracksMapping(BorzoiTracksMappingBase):
    DEFAULT_METADATA_FILE: Path = BORZOI_HUMAN_TARGETS


class BorzoiMouseTracksMapping(BorzoiTracksMappingBase):
    DEFAULT_METADATA_FILE: Path = BORZOI_MOUSE_TARGETS


class FlashRNAHumanTracksMapping(BorzoiTracksMappingBase):
    DEFAULT_METADATA_FILE: Path = FLASH_RNA_HUMAN_TARGETS


class FlashRNAMouseTracksMapping(BorzoiTracksMappingBase):
    DEFAULT_METADATA_FILE: Path = FLASH_RNA_MOUSE_TARGETS


class BorzoiTargetTracks(TargetTracks):
    """Borzoi-specific implementation of TargetTracks

    Borzoi tracks are stored in a squash-transformed raw counts space.
    """

    def get_raw_counts(self, track_name: str | None = None) -> Float[Tensor, "b l d"]:
        """Get tracks in raw counts space after reversing the squash-transformation

        Borzoi tracks are stored in a squash-transformed space, and this method
        returns the tracks after reversing the squash-transformation.

        Args:
            track_name: Name of the track to retrieve. If None, returns all tracks.

        Returns:
            Tensor of tracks in raw counts space after reversing the squash-transformation
            with shape (b, l, d)
        """
        tracks = self.get_tracks(track_name)

        scale = self.tracks_mapping.scale
        clip_soft = self.tracks_mapping.clip_soft
        sqrt_mask = self.tracks_mapping.sqrt_mask

        if track_name is not None:
            track_indices = self.tracks_mapping.indices_by_type(track_name)
            scale = scale[track_indices]
            clip_soft = clip_soft[track_indices]
            sqrt_mask = sqrt_mask[track_indices]

        return inverse_squash_transform(
            tracks,
            scale=scale,
            clip_soft=clip_soft,
            sqrt_mask=sqrt_mask,
        )

    def get_log2p1(self, track_name: str | None = None) -> Float[Tensor, "b l d"]:
        """Convert tracks to log2(counts + 1) space

        First, converts tracks to raw counts space. Then, applies log2(x + 1).
        This is commonly used for visualization and downstream analysis as it
        provides better dynamic range for both low and high count values.

        Args:
            track_name: Name of the track to retrieve. If None, returns all tracks.

        Returns:
            Tensor of tracks in log2p1 space with shape (b, l, d)
        """

        return torch.log2(self.get_raw_counts(track_name) + 1)
