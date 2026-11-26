# plot the nearest neighbors of an existing test record

import numpy as np
import plotly.graph_objects as plotly

from data_loader import split_per_year
from k_nearest_neighbor import make_knn_features, standardize_features, predict_minkowski_knn

from .common import record_info, save_chart


FEATURES = ["Previous-year log volume", "Prior-year annual change", "Prior-quarter annual change"]
CLASSES = ["Decrease / unchanged (0)", "Increase (1)"]
COLORS = ["#1f77b4", "#2ca02c"]


def plot_knn(training, labels, selected, neighbors, record, predicted, year):
    fig = plotly.Figure()

    for label, name in enumerate(CLASSES):
        points = training[labels == label]

        fig.add_trace(
            plotly.Scatter3d(
                x=points[:, 0].tolist(),
                y=points[:, 1].tolist(),
                z=points[:, 2].tolist(),
                mode="markers",
                name=name,
                legendgroup=str(label),
                marker=dict(color=COLORS[label], size=4, opacity=0.35),
                hovertemplate=(name + "<br>" + FEATURES[0] + ": %{x:.3f}"
                               + "<br>" + FEATURES[1] + ": %{y:.3f}"
                               + "<br>" + FEATURES[2] + ": %{z:.3f}<extra></extra>"),
            )
        )

    # separate each connector with None so unrelated neighbors are not joined
    links = [
        [value for index in neighbors for value in (selected[d], training[index, d], None)]
        for d in range(3)
    ]

    fig.add_trace(
        plotly.Scatter3d(
            x=links[0], y=links[1], z=links[2],
            mode="lines",
            line=dict(color="#7f7f7f", width=2),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    for label, name in enumerate(CLASSES):
        indices = [index for index in neighbors if labels[index] == label]
        points = training[indices]

        fig.add_trace(
            plotly.Scatter3d(
                x=points[:, 0].tolist(),
                y=points[:, 1].tolist(),
                z=points[:, 2].tolist(),
                mode="markers",
                name=name + " neighbor",
                legendgroup=str(label),
                showlegend=False,
                marker=dict(color=COLORS[label], size=7, line=dict(color="#263238", width=2)),
            )
        )

    fig.add_trace(
        plotly.Scatter3d(
            x=[float(selected[0])], y=[float(selected[1])], z=[float(selected[2])],
            mode="markers",
            name="Test record",
            marker=dict(color="#263238", size=8, symbol="diamond"),
            hovertemplate=(record["crop"] + " · " + record["region"]
                           + f" · Q{record['quarter']}<br>Actual: {CLASSES[int(record['production'])]}"
                           + "<br>Predicted: " + CLASSES[predicted] + "<extra></extra>"),
        )
    )

    votes = np.bincount(labels[neighbors], minlength=2)
    axes = {axis: dict(title=dict(text=feature.replace(" ", "<br>", 1), font=dict(size=12)))
            for axis, feature in zip(("xaxis", "yaxis", "zaxis"), FEATURES)}

    fig.update_layout(
        title=dict(
            text=f"<b>K-Nearest Neighbors ({year})</b><br><sup>{len(neighbors)} neighbors: {votes[0]} decrease / unchanged, {votes[1]} increase</sup>",
            x=0, xanchor="left", y=0.97, yanchor="top",
        ),
        scene=dict(aspectmode="data", **axes),
        legend=dict(orientation="h", x=0, y=1.03, yanchor="bottom"),
        font=dict(family="system-ui, sans-serif", size=14),
        title_font_size=20,
        height=700,
        margin=dict(t=140, r=70, b=65, l=70),
    )

    return fig


def show_all(data, *, k, p):
    data_table = {
        (r["crop"], r["region"], int(r["year"]), int(r["quarter"])): (
            float(r["same_quarter_previous_year"]), float(r["this_quarter"]), int(r["production"])
        )
        for r in data
    }

    features, labels, years = make_knn_features(data_table)
    year = int(max(years))
    training, testing, _, test_indices = split_per_year(features, labels, years, year)
    train_x, test_x = standardize_features(training["features"], testing["features"])

    k = min(k, len(train_x))
    predictions = predict_minkowski_knn(train_x, training["labels"], test_x, k=k, p=p)
    states = []
    first_figure = None

    for index, selected in enumerate(test_x):
        distances = np.sum(np.abs(train_x - selected) ** p, axis=1) ** (1.0 / p)
        neighbors = np.argsort(distances)[:k]
        record = data[int(test_indices[index])]
        fig = plot_knn(train_x, training["labels"], selected, neighbors, record, predictions[index], year)

        if first_figure is None:
            first_figure = fig

        states.append(dict(
            record=record_info(record),
            traces=[trace.to_plotly_json() for trace in fig.data[2:]],
            layout=dict(title=fig.layout.title.to_plotly_json()),
        ))

    save_chart(first_figure, states, "knn.html", base_count=2)
