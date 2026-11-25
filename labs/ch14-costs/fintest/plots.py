"""The book's chart style for matplotlib figures (light surface, for print)."""

from __future__ import annotations

INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"
SHADE = "#efeeea"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")


def percent_axis(ax, axis: str = "y"):
    """Format an axis of decimal returns as percentages without rounding 12.5% to 12%."""
    import matplotlib.ticker as mt
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v * 100:g}%"))


def new_figure(width: float = 9.0, height: float = 4.6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    return fig, ax, plt


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_3)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def titles(fig, title: str, subtitle: str = "", width_chars: int | None = None) -> int:
    """Title and subtitle at the top left; long subtitles wrap. Returns subtitle line count."""
    import textwrap
    fig.text(0.012, 0.975, title, ha="left", va="top", fontsize=12, color=INK, fontweight="bold")
    if not subtitle:
        return 0
    width_chars = width_chars or int(fig.get_figwidth() * 15)
    lines = textwrap.wrap(subtitle, width_chars)
    fig.text(0.012, 0.915, "\n".join(lines), ha="left", va="top", fontsize=9, color=INK_2, linespacing=1.4)
    fig._subtitle_lines = len(lines)
    return len(lines)


def finish(fig, path, top: float | None = None):
    if top is None:
        top = 0.86 - 0.045 * (getattr(fig, "_subtitle_lines", 1) - 1)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=200)
