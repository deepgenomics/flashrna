from pathlib import Path
from typing import Any, ClassVar, List, Set

import pandas as pd
from jaxtyping import Float
from numpy.typing import NDArray


class TracksMapping:
    """Mapping of tracks to indices in the merged tracks tensor

    This loads tracks metadata in a Pandas DataFrame and applies appropriate
    transformations to it, including filtering out tracks not in `tracks_to_load`
    and re-mapping indices.

    The underlying dataset can store all track types together or separately.
    `orginal_index` is the index of how the track is stored in the underlying dataset.

    From all the tracks in the underlying dataset, `loaded_tracks` specifies which
    tracks are actually loaded and stored in the merged tracks tensor.

    The mapping indices are stored in the underlying `metadata_df` DataFrame.
    Commonly used indices are conveniently accessible via methods, but custom
    indices need to be fetched directly from the DataFrame.

    Child classes can add additional transformations to the dataframe by extending
    `load_tracks_metadata` and `process_tracks_metadata` methods.

    Args:
        metadata_file: Path to metadata file. If None, the child class must define
            `metadata_file` as a class variable. If `metadata_file` is specified,
            it will override `self.metadata_file` in the child class.
        tracks_to_load: Tracks to load. If None, load all tracks. Default is
            ["rna", "dnase", "atac"], which are the tracks used in the FlashRNA model.
    """

    # Columns in metadata file
    ORIGINAL_INDEX_COL: ClassVar[str] = "original_index"
    IDENTIFIER_COL: ClassVar[str] = "identifier"
    TRACK_TYPE_COL: ClassVar[str] = "track_type"
    STRAND_COL: ClassVar[str] = "strand"
    STRAND_PAIR_COL: ClassVar[str] = "strand_pair"
    SCALE_COL: ClassVar[str] = "scale"

    EXPECTED_COLUMNS: ClassVar[List[str]] = [
        ORIGINAL_INDEX_COL,
        IDENTIFIER_COL,
        TRACK_TYPE_COL,
        STRAND_COL,
        STRAND_PAIR_COL,
        SCALE_COL,
    ]

    def __init__(
        self,
        metadata_file: str | Path | None = None,
        tracks_to_load: Set[str] | List[str] | None = ["rna", "dnase", "atac"],
    ):
        if metadata_file is not None:
            metadata_file = Path(metadata_file)
        elif not hasattr(self, "DEFAULT_METADATA_FILE"):
            raise ValueError(
                "`DEFAULT_METADATA_FILE` must be specified in the child class or "
                "passed as an argument to the constructor"
            )
        else:
            metadata_file = self.DEFAULT_METADATA_FILE

        if not metadata_file.exists():
            raise FileNotFoundError(f"`metadata_file` not found: {metadata_file}")

        df = self.load_tracks_metadata(metadata_file)
        df = self.process_tracks_metadata(df, tracks_to_load=tracks_to_load)

        self._metadata_df = df

        # Ensure `_loaded_tracks` preserves order of appearance specified in
        # `_metadata_df`
        self._loaded_tracks = df[self.TRACK_TYPE_COL].unique().tolist()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TracksMapping):
            return False

        # This can be quite restrictive. Can relax this depending on the
        # specific use case in the child class.
        return self.metadata_df.equals(other.metadata_df)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(loaded_tracks: {self.loaded_tracks}, "
            f"num_tracks: {self.num_tracks})"
        )

    @property
    def loaded_tracks(self) -> List[str]:
        """Tracks that are actually loaded and stored in the merged tensor"""
        return self._loaded_tracks.copy()

    @property
    def metadata_df(self) -> pd.DataFrame:
        """Metadata DataFrame"""
        if not hasattr(self, "_metadata_df"):
            raise AttributeError(
                f"{self.__class__.__name__} does not have metadata DataFrame loaded."
            )

        return self._metadata_df

    @property
    def num_tracks(self) -> int:
        """Number of tracks"""
        return len(self.metadata_df)

    @property
    def strand_pairs(self) -> List[int]:
        """Strand pairs"""
        return self.metadata_df[self.STRAND_PAIR_COL].to_list()

    @property
    def scale(self) -> Float[NDArray, " d"]:
        """Scale factor used to transform the original values of each track

        Depending on the dataset, what exactly this means and how it is used
        can vary, but often it is used to ensure dynamic range is within a desired
        range.
        """
        return self.metadata_df[self.SCALE_COL].values

    def indices_by_type(self, track_type: str) -> List[int]:
        """Get tracks indices by track type"""
        if track_type not in self.metadata_df[self.TRACK_TYPE_COL].unique():
            raise KeyError(
                f"Invalid track type: '{track_type}'. "
                f"Available types: {self.metadata_df[self.TRACK_TYPE_COL].unique()}"
            )
        return self.metadata_df[
            self.metadata_df[self.TRACK_TYPE_COL] == track_type
        ].index.to_list()

    def load_tracks_metadata(self, metadata_file: str | Path) -> pd.DataFrame:
        """Load tracks metadata from underlying mapping file (e.g. a CSV file)

        This loads the raw metadata file without much processing (done in
        `process_tracks_metadata`).

        Returns:
            DataFrame with tracks metadata
        """
        df = pd.read_csv(metadata_file, sep="\t", index_col=None)

        # Sanity check
        if not set(self.EXPECTED_COLUMNS).issubset(df.columns):
            raise ValueError(
                f"Expected columns {self.EXPECTED_COLUMNS} not found in DataFrame"
            )

        return df

    def process_tracks_metadata(
        self, df: pd.DataFrame, tracks_to_load: List[str] | None = None
    ) -> pd.DataFrame:
        """Process tracks metadata to filter out tracks not in `tracks_to_load` and
        update `strand_pair` accordingly.

        Args:
            df: DataFrame with tracks metadata
            tracks_to_load: List of tracks to load. If None, load all tracks

        Returns:
            DataFrame with the new tracks metadata with updated `strand_pair`
        """
        # If `tracks_to_load` is not specified, load all tracks
        if tracks_to_load is None:
            # If not specified, load all tracks
            tracks_to_load = df[self.TRACK_TYPE_COL].unique().tolist()
        else:
            if len(tracks_to_load) != len(set(tracks_to_load)):
                raise ValueError(
                    "`tracks_to_load` contains duplicates. Must contain unique tracks"
                )

            # Validate that all requested tracks exist
            available_tracks = set(df[self.TRACK_TYPE_COL].unique())
            missing_tracks = set(tracks_to_load) - available_tracks
            if missing_tracks:
                raise ValueError(
                    f"Requested tracks not found in DataFrame: {sorted(missing_tracks)}"
                )

        # Collect tracks of the same type together and update `strand_pair` with
        # the updated indices of the strand pair (otherwise, it will still point
        # to the original indices)

        df_track_list = []
        track_idx_offset = 0  # offset where each track type starts in the merged tensor
        for track_type in tracks_to_load:
            df_track = df[df[self.TRACK_TYPE_COL] == track_type].reset_index(drop=True)
            # `track_index` will be the absolute index in the merged tensor
            track_index = df_track.index + track_idx_offset

            # Sanity check before processing `strand_pair`: make sure all are valid
            self._validate_strand_pairs(df_track)

            # Map `strand_pair` to the `track_index` of the pair
            pair_idx_mapping = dict(zip(df_track[self.ORIGINAL_INDEX_COL], track_index))
            df_track[self.STRAND_PAIR_COL] = df_track[self.STRAND_PAIR_COL].map(
                pair_idx_mapping
            )

            df_track_list.append(df_track)
            track_idx_offset += len(df_track)  # update offset for the next track type

        df = pd.concat(df_track_list).reset_index(drop=True)

        return df

    def _validate_strand_pairs(self, df: pd.DataFrame) -> None:
        """Validate that strand pairing is consistent and correct.

        This validates only within a single track type. This is because depending on
        how tracks are stored in the source data, strand pair might be indices within
        a single track type, not global indices. Then, `df` from multiple track types
        will have clashing strand pairs.

        Validates:
        - All strand_pair values reference existing original_index values
        - Unstranded tracks (".") point to themselves
        - Stranded tracks ("+"/"-") have equal counts and point to each other
        - All pairing relationships are bidirectional

        Args:
            df: DataFrame of a single track type with required columns
        """
        # Initial sanity checks
        if df[self.TRACK_TYPE_COL].nunique() > 1:
            raise ValueError("DataFrame must contain a single track type")

        if set(df[self.STRAND_PAIR_COL]) != set(df[self.ORIGINAL_INDEX_COL]):
            raise ValueError("`strand_pair` and `original_index` do not match")

        self._validate_unstranded_tracks(df[df[self.STRAND_COL] == "."])
        self._validate_stranded_tracks(df[df[self.STRAND_COL] != "."])

    def _validate_unstranded_tracks(self, df: pd.DataFrame) -> None:
        if df.empty:
            return

        if not df[self.STRAND_PAIR_COL].equals(df[self.ORIGINAL_INDEX_COL]):
            raise ValueError(
                "Unstranded tracks must have strand pair pointing to themselves"
            )

    def _validate_stranded_tracks(self, df: pd.DataFrame) -> None:
        df_plus = df[df[self.STRAND_COL] == "+"]
        df_minus = df[df[self.STRAND_COL] == "-"]

        if not len(df_plus) == len(df_minus):
            raise ValueError(
                "Plus and minus strands must have the same number of tracks"
            )

        if df_plus.empty:
            return

        # Check that plus and minus strands point to each other
        # Create dictionaries for O(1) lookups instead of O(n) DataFrame searches
        original_to_strand = dict(zip(df[self.ORIGINAL_INDEX_COL], df[self.STRAND_COL]))
        original_to_strand_pair = dict(
            zip(df[self.ORIGINAL_INDEX_COL], df[self.STRAND_PAIR_COL])
        )

        for _, plus_row in df_plus.iterrows():
            plus_idx = plus_row[self.ORIGINAL_INDEX_COL]
            paired_idx = plus_row[self.STRAND_PAIR_COL]

            # Check if the paired track exists
            if paired_idx not in original_to_strand:
                raise ValueError(
                    f"Plus strand {plus_idx} points to non-existent " f"{paired_idx}"
                )

            # Check if the paired track is a minus strand
            paired_strand = original_to_strand[paired_idx]
            if paired_strand != "-":
                raise ValueError(
                    f"Plus strand {plus_idx} should point to minus strand, "
                    f"but points to strand='{paired_strand}'"
                )

            # Check that the minus strand points back to the plus strand
            minus_strand_pair = original_to_strand_pair[paired_idx]
            if minus_strand_pair != plus_idx:
                raise ValueError(
                    f"Plus strand {plus_idx} and minus strand {paired_idx} do not "
                    "point to each other"
                )
