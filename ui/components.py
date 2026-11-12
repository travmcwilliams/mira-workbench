"""Reusable Streamlit components (image gallery, status badges, etc.)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import streamlit as st


def image_gallery(
    image_paths: Sequence[Path | str],
    title: str | None = None,
    max_cols: int = 2,
    caption_prefix: str = "",
) -> None:
    """
    Display a grid of images from disk paths.

    Args:
        image_paths: List of paths to PNG/JPG files.
        title: Optional header.
        max_cols: Number of columns in the grid.
        caption_prefix: Prefix for image captions.
    """
    valid_paths = [Path(p) for p in image_paths if Path(p).exists()]
    if not valid_paths:
        st.caption("No images to display.")
        return

    if title:
        st.subheader(title)

    cols = st.columns(max_cols)
    for idx, p in enumerate(valid_paths):
        col = cols[idx % max_cols]
        with col:
            caption = f"{caption_prefix}{p.name}" if caption_prefix else p.name
            # use_column_width, not width="stretch": the latter needs a newer
            # Streamlit than the 1.39 pin and raises TypeError there.
            st.image(str(p), caption=caption, use_column_width=True)
