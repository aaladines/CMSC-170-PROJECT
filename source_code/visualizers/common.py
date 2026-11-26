# export the chart and its existing test-record display states

import json
from pathlib import Path

import plotly.graph_objects as plotly


OUTPUT_FOLDER = Path(__file__).resolve().parents[2] / "tables_and_charts"


def record_info(record):
    return dict(
        crop=record["crop"],
        region=record["region"],
        quarter=int(record["quarter"]),
        actual=int(record["production"]),
    )


def save_chart(fig, records, filename, base_count):
    figure = json.loads(fig.to_json())
    layout = figure["layout"]
    layout["uirevision"] = "2025"
    base_data = figure["data"][:base_count]
    first = records[0]
    initial = plotly.Figure(data=base_data + first["traces"], layout={**layout, **first.get("layout", {})})

    OUTPUT_FOLDER.mkdir(exist_ok=True)
    path = OUTPUT_FOLDER / filename
    initial.write_html(
        path,
        include_plotlyjs="cdn",
        div_id="method-plot",
        post_script='connectMethodChart("{plot_id}");',
    )

    # keep the display states inside the chart instead of adding a data file
    state = json.dumps(dict(base_data=base_data, layout=layout, records=records), separators=(",", ":"), allow_nan=False)
    state = state.replace("<", r"\u003c")
    head = (
        '<style>html, body { margin: 0; padding: 0; height: auto; }</style>'
        '<script src="../method_selector.js"></script>'
        '<script type="application/json" id="method-state">' + state + '</script></head>'
    )

    html = path.read_text(encoding="utf-8").replace("</head>", head, 1)
    path.write_text(html, encoding="utf-8")

    print("Saved " + str(path))
