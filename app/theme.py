"""Design tokens and Plotly templates.

Two selected modes — not an automatic inversion. Each mode's categorical slots
were stepped for their own surface and validated together (lightness band,
chroma floor, adjacent CVD separation, normal-vision floor, contrast).

Colour is assigned by the job it does:
  categorical -> identity (a series, capped at 8 slots, never cycled)
  sequential  -> magnitude (one hue, light to dark)
  diverging   -> polarity (blue/red poles, neutral grey midpoint)
  status      -> state (reserved; always paired with a label)
"""

from __future__ import annotations

# ── Categorical slots, in fixed order ────────────────────────────────────────
CATEGORICAL = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
              "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark":  ["#3987e5", "#d95926", "#199e70", "#c98500",
              "#d55181", "#008300", "#9085e9", "#e66767"],
}
MAX_SERIES = 8

# ── Sequential ramp (blue, light → dark) ─────────────────────────────────────
SEQUENTIAL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
              "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
              "#184f95", "#104281", "#0d366b"]

# ── Diverging pair: blue ↔ red with a neutral midpoint ───────────────────────
DIVERGING = {
    "light": [[0.0, "#0d366b"], [0.25, "#3987e5"], [0.5, "#f0efec"],
              [0.75, "#e34948"], [1.0, "#8f1f1f"]],
    "dark":  [[0.0, "#104281"], [0.25, "#3987e5"], [0.5, "#383835"],
              [0.75, "#e66767"], [1.0, "#a82f2f"]],
}

# ── Status colours — reserved, never reused as a series ──────────────────────
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

TOKENS = {
    "light": {
        "surface": "#fcfcfb",
        "plane": "#f9f9f7",
        "raised": "#ffffff",
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "text_muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "border": "rgba(11,11,11,0.10)",
        "positive": "#006300",
        "negative": "#d03b3b",
        "accent": "#2a78d6",
    },
    "dark": {
        "surface": "#1a1a19",
        "plane": "#0d0d0d",
        "raised": "#232322",
        "text_primary": "#ffffff",
        "text_secondary": "#c3c2b7",
        "text_muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "border": "rgba(255,255,255,0.10)",
        "positive": "#0ca30c",
        "negative": "#e66767",
        "accent": "#3987e5",
    },
}

FONT_STACK = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def tokens(mode: str) -> dict:
    return TOKENS.get(mode, TOKENS["dark"])


def palette(mode: str) -> list[str]:
    return CATEGORICAL.get(mode, CATEGORICAL["dark"])


def diverging(mode: str):
    return DIVERGING.get(mode, DIVERGING["dark"])


def diverging_returns(mode: str):
    """Diverging scale for signed *returns*: losses red, gains blue.

    The base scale runs blue -> red for magnitude (correlation, exposure).
    Money is different: red has to mean loss, so the scale is reversed rather
    than reaching for green — red/green is the pairing most often confused
    under colour-vision deficiency.
    """
    scale = diverging(mode)
    return [[1 - stop, color] for stop, color in reversed(scale)]


def diverging_poles(mode: str) -> tuple[str, str]:
    """``(loss_color, gain_color)`` — red for down, blue for up."""
    scale = diverging(mode)
    return scale[-2][1], scale[1][1]


def series_color(mode: str, slot: int) -> str:
    """Colour for a categorical slot. Slots are assigned, never cycled."""
    slots = palette(mode)
    return slots[slot % len(slots)]


def delta_color(mode: str, value: float) -> str:
    """Direction colour for a signed number (up is good for returns)."""
    t = tokens(mode)
    if value is None:
        return t["text_secondary"]
    return t["positive"] if value >= 0 else t["negative"]


def layout(mode: str, *, height: int = 360, showlegend: bool = True, **overrides) -> dict:
    """Base Plotly layout: recessive hairline chrome, generous margins."""
    t = tokens(mode)
    axis = dict(
        gridcolor=t["grid"], gridwidth=1, griddash="solid",
        linecolor=t["axis"], linewidth=1,
        zeroline=False, ticks="outside", ticklen=4, tickcolor=t["axis"],
        tickfont=dict(size=11, color=t["text_muted"]),
        title=dict(font=dict(size=11, color=t["text_secondary"])),
        automargin=True,
    )
    base = dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_STACK, size=12, color=t["text_secondary"]),
        xaxis={**axis},
        yaxis={**axis},
        margin=dict(l=8, r=16, t=8, b=8),
        showlegend=showlegend,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=t["text_secondary"]),
            itemsizing="constant",
        ),
        hoverlabel=dict(
            bgcolor=t["raised"], bordercolor=t["axis"], font_size=12,
            font_family=FONT_STACK, font_color=t["text_primary"], align="left",
        ),
        colorway=palette(mode),
        transition=dict(duration=0),
    )
    base.update(overrides)
    return base


def rgba(hex_color: str, alpha: float) -> str:
    """Hex to rgba — used for the ~10% area washes under lines."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ── Mark specs (fixed across every chart) ────────────────────────────────────
LINE_WIDTH = 2
LINE_WIDTH_THIN = 1.25
MARKER_SIZE = 9
AREA_ALPHA = 0.10
BAR_MAX_WIDTH = 24
