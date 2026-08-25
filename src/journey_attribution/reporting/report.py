"""
HTML report: attribution credit compared across methods, plus journey
visualizations (channel flow, path frequency, journey length). Built with
Plotly so the output is a single interactive HTML file — no server, open it
in a browser.

Colors are the fixed 8-slot categorical palette (see the dataviz skill /
references/palette.md) assigned by role, never cycled, so the same channel
or method always gets the same color across every chart in the report.

Author: Anastasiia Bakhtoiarova
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from journey_attribution.schemas import Journey, AttributionResult
from journey_attribution.attribution.markov import transition_summary, START, CONVERSION, NULL

CATEGORICAL = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
STATUS_GOOD = "#0ca30c"
MUTED = "#898781"
SURFACE = "#fcfcfb"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"

CHANNEL_ORDER = [
    "paid_search", "organic_search", "email", "referral",
    "social", "direct", "display", "other",
]
METHOD_ORDER = [
    "first_touch", "last_touch", "linear", "time_decay", "position_based",
    "markov_removal_effect", "datadriven_shap",
]


def _assign_colors(keys: list[str], preferred_order: list[str]) -> dict[str, str]:
    ordered = [k for k in preferred_order if k in keys] + [k for k in keys if k not in preferred_order]
    return {k: CATEGORICAL[i % len(CATEGORICAL)] for i, k in enumerate(ordered)}


def _layout(title: str, xaxis_title: str, yaxis_title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=16, color=INK_PRIMARY, family=FONT)),
        font=dict(family=FONT, color=INK_SECONDARY, size=12),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        xaxis=dict(title=xaxis_title, gridcolor=GRIDLINE, linecolor=AXIS, tickfont=dict(color=INK_SECONDARY)),
        yaxis=dict(title=yaxis_title, gridcolor=GRIDLINE, linecolor=AXIS, tickfont=dict(color=INK_SECONDARY)),
        margin=dict(l=60, r=30, t=50, b=60),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )


def fig_credit_by_channel(results: dict[str, AttributionResult]) -> go.Figure:
    channels_seen = sorted({c for r in results.values() for c in r.credits})
    channels = [c for c in CHANNEL_ORDER if c in channels_seen] + [c for c in channels_seen if c not in CHANNEL_ORDER]
    methods = [m for m in METHOD_ORDER if m in results] + [m for m in results if m not in METHOD_ORDER]
    colors = _assign_colors(methods, METHOD_ORDER)

    fig = go.Figure()
    for m in methods:
        credits = results[m].credits
        fig.add_bar(name=m, x=channels, y=[credits.get(c, 0.0) for c in channels], marker_color=colors[m])
    fig.update_layout(barmode="group", **_layout(
        "Attributed credit by channel, across methods", "Channel", "Credit (share of conversions)",
    ))
    return fig


def fig_channel_frequency(journeys: list[Journey]) -> go.Figure:
    counts: Counter[str] = Counter()
    for j in journeys:
        counts.update(j.channels_in_order)
    channels = [c for c, _ in counts.most_common()]
    colors = _assign_colors(channels, CHANNEL_ORDER)

    fig = go.Figure(go.Bar(
        x=channels, y=[counts[c] for c in channels],
        marker_color=[colors[c] for c in channels], showlegend=False,
    ))
    fig.update_layout(**_layout("Touchpoint frequency by channel", "Channel", "Touchpoints (all journeys)"))
    return fig


def fig_journey_length_hist(journeys: list[Journey]) -> go.Figure:
    lengths = [len(j.channels_in_order) for j in journeys]
    fig = go.Figure(go.Histogram(x=lengths, marker_color=CATEGORICAL[0], xbins=dict(size=1)))
    fig.update_layout(bargap=0.1, **_layout("Journey length distribution", "Touchpoints per journey", "Journeys"))
    return fig


def fig_top_paths(journeys: list[Journey], top_n: int = 10) -> go.Figure:
    def deduped(order: list[str]) -> tuple[str, ...]:
        out: list[str] = []
        for c in order:
            if not out or out[-1] != c:
                out.append(c)
        return tuple(out)

    counts = Counter(deduped(j.channels_in_order) for j in journeys)
    top = counts.most_common(top_n)
    labels = [" → ".join(path) for path, _ in top][::-1]
    values = [c for _, c in top][::-1]

    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=CATEGORICAL[0]))
    fig.update_layout(**_layout(f"Top {top_n} journey paths", "Journeys", ""))
    fig.update_layout(height=max(320, 40 * len(labels)))
    return fig


def fig_journey_flow_sankey(journeys: list[Journey], min_share: float = 0.002) -> go.Figure:
    states, counts = transition_summary(journeys)
    total = sum(counts.values()) or 1.0
    threshold = total * min_share
    counts = {k: v for k, v in counts.items() if v >= threshold}

    channels = [s for s in states if s not in (START, CONVERSION, NULL)]
    channel_colors = _assign_colors(channels, CHANNEL_ORDER)
    node_colors = {START: MUTED, NULL: MUTED, CONVERSION: STATUS_GOOD, **channel_colors}

    idx = {s: i for i, s in enumerate(states)}
    sources, targets, values, link_colors = [], [], [], []
    for (a, b), v in counts.items():
        sources.append(idx[a])
        targets.append(idx[b])
        values.append(v)
        hex_color = node_colors[a]
        r, g, b_ = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        link_colors.append(f"rgba({r},{g},{b_},0.35)")

    fig = go.Figure(go.Sankey(
        node=dict(
            label=states, color=[node_colors[s] for s in states],
            pad=14, thickness=16, line=dict(color=AXIS, width=0.5),
        ),
        link=dict(source=sources, target=targets, value=values, color=link_colors),
    ))
    fig.update_layout(
        title=dict(
            text=f"Journey flow between channels (edges below {min_share:.1%} of total flow omitted)",
            font=dict(size=16, color=INK_PRIMARY, family=FONT),
        ),
        font=dict(family=FONT, color=INK_SECONDARY, size=12),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def _lerp_hex(lo: str, hi: str, t: float) -> str:
    lo_rgb = [int(lo[i:i + 2], 16) for i in (1, 3, 5)]
    hi_rgb = [int(hi[i:i + 2], 16) for i in (1, 3, 5)]
    mixed = [round(a + (b - a) * t) for a, b in zip(lo_rgb, hi_rgb)]
    return f"#{mixed[0]:02x}{mixed[1]:02x}{mixed[2]:02x}"


def _comparison_table_html(results: dict[str, AttributionResult]) -> str:
    channels_seen = sorted({c for r in results.values() for c in r.credits})
    channels = [c for c in CHANNEL_ORDER if c in channels_seen] + [c for c in channels_seen if c not in CHANNEL_ORDER]
    methods = [m for m in METHOD_ORDER if m in results] + [m for m in results if m not in METHOD_ORDER]
    df = pd.DataFrame({m: [results[m].credits.get(c, 0.0) for c in channels] for m in methods}, index=channels)
    vmax = df.values.max() or 1.0

    header = "<th>channel</th>" + "".join(f"<th>{m}</th>" for m in methods)
    rows = []
    for channel in channels:
        cells = [f"<td>{channel}</td>"]
        for m in methods:
            v = df.loc[channel, m]
            bg = _lerp_hex("#cde2fb", "#0d366b", min(v / vmax, 1.0))
            ink = INK_PRIMARY if v / vmax < 0.55 else "#ffffff"
            cells.append(f"<td style='background:{bg};color:{ink}'>{v:.3f}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return f"<table class='cmp-table'><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def build_report(
    journeys: list[Journey],
    results: dict[str, AttributionResult],
    output_path: str,
    diagnostics: dict | None = None,
) -> None:
    n = len(journeys)
    conv_rate = sum(j.converted for j in journeys) / n if n else 0.0
    diagnostics = diagnostics or {}

    figs = [
        fig_credit_by_channel(results),
        fig_journey_flow_sankey(journeys),
        fig_channel_frequency(journeys),
        fig_top_paths(journeys),
        fig_journey_length_hist(journeys),
    ]
    chart_divs = []
    for i, fig in enumerate(figs):
        chart_divs.append(fig.to_html(
            full_html=False, include_plotlyjs=("inline" if i == 0 else False), div_id=f"chart-{i}",
        ))

    diag_line = "".join(f"<div class='stat'><div class='v'>{v:.3f}</div><div class='l'>{k}</div></div>"
                         for k, v in diagnostics.items())

    html = f"""<meta charset="utf-8">
<title>Journey Attribution Report</title>
<style>
  body {{ background:{SURFACE}; color:{INK_PRIMARY}; font-family:{FONT}; margin:0; padding:32px 40px 64px; }}
  h1 {{ font-size:22px; margin:0 0 4px; }}
  .meta {{ color:{INK_SECONDARY}; font-size:13px; margin-bottom:24px; }}
  .stats {{ display:flex; gap:28px; margin-bottom:32px; flex-wrap:wrap; }}
  .stat {{ background:#fff; border:1px solid {GRIDLINE}; border-radius:8px; padding:12px 18px; min-width:120px; }}
  .stat .v {{ font-size:22px; font-weight:600; }}
  .stat .l {{ font-size:12px; color:{INK_SECONDARY}; margin-top:2px; }}
  .chart {{ border:1px solid {GRIDLINE}; border-radius:8px; margin-bottom:28px; padding:8px; overflow-x:auto; }}
  table.cmp-table {{ border-collapse:collapse; font-size:13px; margin-bottom:32px; }}
  table.cmp-table th, table.cmp-table td {{ padding:6px 10px; border:1px solid {GRIDLINE}; text-align:right; }}
  table.cmp-table th:first-child, table.cmp-table td:first-child {{ text-align:left; }}
  h2 {{ font-size:15px; color:{INK_SECONDARY}; margin:0 0 8px; font-weight:600; }}
</style>
<h1>Journey Attribution Report</h1>
<div class="meta">Generated {datetime.now():%Y-%m-%d %H:%M} &middot; {n:,} journeys</div>
<div class="stats">
  <div class="stat"><div class="v">{n:,}</div><div class="l">journeys</div></div>
  <div class="stat"><div class="v">{conv_rate:.2%}</div><div class="l">empirical conversion rate</div></div>
  {diag_line}
</div>
<h2>Attribution comparison</h2>
<div class="chart">{chart_divs[0]}</div>
{_comparison_table_html(results)}
<h2>Journey flow</h2>
<div class="chart">{chart_divs[1]}</div>
<h2>Channel touchpoint frequency</h2>
<div class="chart">{chart_divs[2]}</div>
<h2>Most common journey paths</h2>
<div class="chart">{chart_divs[3]}</div>
<h2>Journey length distribution</h2>
<div class="chart">{chart_divs[4]}</div>
"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html, encoding="utf-8")
