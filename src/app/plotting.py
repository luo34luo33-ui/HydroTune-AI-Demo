"""Bound chart rendering memory for Streamlit Community Cloud."""
from io import BytesIO
import math

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

PLOT_DPI = 120
MAX_IMAGE_PIXELS = 4_000_000
EVENTS_PER_FIGURE = 4


def create_event_figure(event_count):
    """Create at most four event panels; callers render batches sequentially."""
    if not 1 <= event_count <= EVENTS_PER_FIGURE:
        raise ValueError("event_count must be between 1 and 4")
    cols = min(2, event_count)
    rows = math.ceil(event_count / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5.8 * rows), dpi=PLOT_DPI)
    axes = np.asarray(axes, dtype=object).reshape(-1)
    for ax in axes[event_count:]:
        ax.set_visible(False)
    return fig, axes


def figure_png(fig):
    """Encode one bounded PNG and always unregister its Matplotlib figure."""
    try:
        width, height = fig.get_size_inches()
        dpi = min(PLOT_DPI, math.sqrt(MAX_IMAGE_PIXELS / (width * height)) * 0.99)
        fig.set_dpi(dpi)
        with BytesIO() as buffer:
            # A tight bounding box can expand beyond the pixel budget.
            with plt.rc_context({"savefig.bbox": None}):
                fig.savefig(buffer, format="png", dpi=dpi, bbox_inches=None)
            return buffer.getvalue()
    finally:
        plt.close(fig)


def render_figure(fig, *, label=None, file_name=None, key=None):
    """Reuse the display PNG for downloads instead of rendering a second copy."""
    png = figure_png(fig)
    st.image(png, width="stretch")
    if file_name is not None:
        st.download_button(
            label, data=png, file_name=file_name, mime="image/png",
            key=key, on_click="ignore",
        )
