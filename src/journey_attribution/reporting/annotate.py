"""
Post-processes an already-generated report.html to insert manually-written
insight callouts at fixed points in the page — without touching the
chart-rendering scripts, the Plotly library, or any other existing content
in the file.

Workflow: generate a fresh report (`journey-attribution --mode real
--report reports/report.html`), then run this script against it to layer
commentary on top:

    python -m journey_attribution.reporting.annotate reports/report.html

To add or change commentary, edit INSIGHTS below — each entry is
(heading, html), and the callout is inserted immediately before the
`<h2>{heading}</h2>` it precedes. Use heading=None for a callout appended
at the very end of the report, after the last chart. Re-running this
script against an already-annotated file is safe — each callout is
inserted at most once.

Author: Anastasiia Bakhtoiarova
"""
from __future__ import annotations
import argparse
from pathlib import Path

INSIGHT_CSS = """
.insight { background:#f6f5f0; border-left:3px solid #52514e;
  border-radius:6px; padding:10px 16px; margin:0 0 28px;
  font-size:16px; line-height:1.5; color:#2b2a27; }
.insight b { color:#0b0b0b; }
"""

INSIGHTS: list[tuple[str | None, str]] = [
    (
        "Attribution comparison",
        "<b>83%</b> of the 267,084 journeys are single-touchpoint (mean length 1.31), "
        "and the blended conversion rate is 1.63%. Multi-touch attribution only "
        "meaningfully applies to the <b>~17%</b> of journeys with 2+ touches — most "
        "of what follows is about a minority of traffic.",
    ),
    (
        "Journey flow",
        "Every heuristic model (first-touch, last-touch, linear, time-decay, "
        "position-based, Markov removal effect) agrees: organic_search leads "
        "(28–35% credit), direct is second (~22–23%), paid_search is reliably last "
        "(2–5%). That ranking tracks almost exactly with raw touchpoint volume "
        "(organic_search is 34.4% of all touches), which suggests these models are "
        "rewarding frequency, not causal contribution. The data-driven (Shapley) "
        "model disagrees sharply: organic_search drops to 12.8%, direct to 10.3%, "
        "while 'other' jumps to 43% and referral to 25.5% — <b>a near-total "
        "reversal</b>. Worth noting <b>the data-driven model's AUC is only "
        "0.712</b>, so treat this as directional, not definitive.",
    ),
    (
        "Channel touchpoint frequency",
        "In the flow data, every channel's largest outbound edge goes to "
        "<b>drop-off ('Null'), not conversion</b> — e.g. organic_search → Null is "
        "93,681 vs. organic_search → Conversion at 1,220 (the only single-hop "
        "conversion edge large enough to clear the chart's 0.2%-of-flow display "
        "threshold). Per-channel conversion counts are understated here because "
        "smaller conversion edges were suppressed for readability — <b>the 1.63% "
        "empirical rate is the only fully reliable conversion figure in this "
        "report</b>.",
    ),
    (
        "Most common journey paths",
        "Touchpoint share (organic_search 34.4%, direct 23.2%, referral 17.4%, "
        "display 14.7%, other 5.8%, paid_search 4.4%) <b>lines up almost exactly "
        "with the heuristic attribution ranking above</b> — supporting evidence "
        "that those models are largely tracking exposure volume.",
    ),
    (
        "Journey length distribution",
        "Single-channel paths dominate: organic_search alone (87,082) and direct "
        "alone (54,267) each dwarf the best multi-hop combination (organic_search "
        "→ referral, at 3,262). <b>Multi-touch sequences are a rounding error next "
        "to single-touch volume.</b>",
    ),
    (
        None,
        "Journey length distribution: 83.0% length 1, 10.8% length 2, 3.0% length "
        "3, and a long thin tail out to 12 touches. This is the clearest evidence "
        "that <b>most traffic isn't 'journeying' anywhere — it converts or drops "
        "in a single touch.</b>",
    ),
]


def apply_insights(html: str, insights: list[tuple[str | None, str]] = INSIGHTS) -> str:
    if ".insight {" not in html:
        html = html.replace("</style>", INSIGHT_CSS + "</style>", 1)

    for heading, text in insights:
        box = f'<div class="insight">{text}</div>\n'
        if box in html:
            continue
        if heading is None:
            html = html.rstrip() + "\n" + box
        else:
            marker = f"<h2>{heading}</h2>"
            if marker not in html:
                raise ValueError(f"Anchor heading not found in report: {heading!r}")
            html = html.replace(marker, box + marker, 1)
    return html


def main() -> None:
    parser = argparse.ArgumentParser(description="Insert insight callouts into a generated report.html.")
    parser.add_argument("report", nargs="?", default="reports/report.html",
                         help="Path to the report.html to annotate (edited in place)")
    args = parser.parse_args()

    path = Path(args.report)
    html = apply_insights(path.read_text(encoding="utf-8"))
    path.write_text(html, encoding="utf-8")
    print(f"Inserted {len(INSIGHTS)} insight callouts into {path}")


if __name__ == "__main__":
    main()
