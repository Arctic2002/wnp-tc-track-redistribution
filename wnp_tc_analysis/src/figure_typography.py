"""Shared final-scale typography for WNP TC analysis figures."""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.text import Text


FINAL_FONT_SCALE = 1.10
AGU_FONT = "Arial"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [AGU_FONT, "Microsoft YaHei"],
        "mathtext.fontset": "custom",
        "mathtext.rm": AGU_FONT,
        "mathtext.it": f"{AGU_FONT}:italic",
        "mathtext.bf": f"{AGU_FONT}:bold",
        "mathtext.sf": AGU_FONT,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def scale_figure_typography(fig, scale: float = FINAL_FONT_SCALE) -> None:
    """Increase all visible text artists once immediately before export.

    Axis labels, ticks, legends, colourbars, annotations, and panel letters
    retain their existing hierarchy. This affects presentation only.
    """
    fig.canvas.draw()
    for text in fig.findobj(match=Text):
        if text.get_visible() and text.get_fontsize() > 0:
            text.set_fontfamily(AGU_FONT)
            text.set_fontsize(text.get_fontsize() * scale)
    fig.canvas.draw()
