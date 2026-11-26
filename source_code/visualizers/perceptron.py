# plot the final trained Perceptron plane using the existing training split

from itertools import product
import numpy as np
import plotly.graph_objects as plotly

from data_loader import split_per_year
from perceptron import make_perceptron_features, standardize_features, train_perceptron, predict_perceptron

from .common import record_info, save_chart


FEATURES = ["Previous-year log volume", "Prior-year annual change", "Prior-quarter annual change"]
CLASSES = ["Decrease / unchanged (0)", "Increase (1)"]
COLORS = ["#1f77b4", "#2ca02c"]


def separating_plane(weights, bias, ranges):
    # find where the plane meets the twelve edges of the plot box
    normal = np.array(weights, dtype=float)

    if not np.any(normal):
        return []

    vertices = []

    def add(point):
        if not any(np.linalg.norm(point - other) < 1e-9 for other in vertices):
            vertices.append(point.copy())

    for corner in product((0, 1), repeat=3):
        start = np.array([ranges[d][corner[d]] for d in range(3)])

        for d in range(3):
            if corner[d]:
                continue

            end = start.copy()
            end[d] = ranges[d][1]
            a = float(normal @ start + bias)
            b = float(normal @ end + bias)

            if a == 0:
                add(start)

            if b == 0:
                add(end)

            if (a < 0 < b) or (b < 0 < a):
                add(start + a / (a - b) * (end - start))

    if len(vertices) < 3:
        return []

    # order the vertices around the plane so its triangles do not cross
    normal /= np.linalg.norm(normal)
    reference = np.eye(3)[np.argmin(np.abs(normal))]
    horizontal = np.cross(normal, reference)
    horizontal /= np.linalg.norm(horizontal)
    vertical = np.cross(normal, horizontal)
    center = np.mean(vertices, axis=0)
    vertices.sort(key=lambda point: np.arctan2((point - center) @ vertical, (point - center) @ horizontal))

    return [point.tolist() for point in vertices]


def plot_perceptron(training, labels, weights, bias, year, testing=None):
    fig = plotly.Figure()
    bounds = training if testing is None else np.vstack((training, testing))
    low, high = bounds.min(axis=0), bounds.max(axis=0)
    padding = np.where(high > low, (high - low) * 0.08, 1)
    ranges = np.column_stack((low - padding, high + padding))
    vertices = separating_plane(weights, bias, ranges)

    if vertices:
        fig.add_trace(
            plotly.Mesh3d(
                x=[point[0] for point in vertices],
                y=[point[1] for point in vertices],
                z=[point[2] for point in vertices],
                i=[0] * (len(vertices) - 2),
                j=list(range(1, len(vertices) - 1)),
                k=list(range(2, len(vertices))),
                name="Separating plane",
                color="#d62728",
                opacity=0.18,
                flatshading=True,
                hoverinfo="skip",
                showlegend=True,
            )
        )

    for label, name in enumerate(CLASSES):
        points = training[labels == label]

        fig.add_trace(
            plotly.Scatter3d(
                x=points[:, 0].tolist(),
                y=points[:, 1].tolist(),
                z=points[:, 2].tolist(),
                mode="markers",
                name=name,
                marker=dict(color=COLORS[label], size=4, opacity=0.7),
                hovertemplate=(name + "<br>" + FEATURES[0] + ": %{x:.3f}"
                               + "<br>" + FEATURES[1] + ": %{y:.3f}"
                               + "<br>" + FEATURES[2] + ": %{z:.3f}<extra></extra>"),
            )
        )

    axes = {
        axis: dict(title=dict(text=feature.replace(" ", "<br>", 1), font=dict(size=12)), range=ranges[d].tolist())
        for d, (axis, feature) in enumerate(zip(("xaxis", "yaxis", "zaxis"), FEATURES))
    }

    fig.update_layout(
        title=dict(text=f"<b>Perceptron separating plane ({year})</b>", x=0, xanchor="left", y=0.97, yanchor="top"),
        scene=dict(aspectmode="data", **axes),
        legend=dict(orientation="h", x=0, y=1.03, yanchor="bottom"),
        font=dict(family="system-ui, sans-serif", size=14),
        title_font_size=20,
        height=700,
        margin=dict(t=105, r=70, b=65, l=70),
    )

    return fig


def show_all(data, *, learning_rate, max_epochs):
    features, labels, records = make_perceptron_features(data)
    years = np.array([int(record["year"]) for record in records])
    year = int(max(years))
    training, testing, _, test_indices = split_per_year(features, labels, years, year)
    train_x, test_x = standardize_features(training["features"], testing["features"])

    weights, bias, _, _ = train_perceptron(train_x, training["labels"], learning_rate=learning_rate, max_epochs=max_epochs)
    fig = plot_perceptron(train_x, training["labels"], weights, bias, year, testing=test_x)
    predictions = predict_perceptron(test_x, weights, bias)
    states = []

    for index, point in enumerate(test_x):
        record = record_info(records[int(test_indices[index])])
        predicted = int(predictions[index])
        trace = plotly.Scatter3d(
            x=[float(point[0])], y=[float(point[1])], z=[float(point[2])],
            mode="markers", name="Test record",
            marker=dict(color="#263238", size=8, symbol="diamond"),
            hovertemplate=(record["crop"] + " · " + record["region"]
                           + f" · Q{record['quarter']}<br>Actual: {CLASSES[record['actual']]}"
                           + "<br>Predicted: " + CLASSES[predicted] + "<extra></extra>"),
        )
        title = fig.layout.title.to_plotly_json()
        title["text"] += "<br><sup>Actual: " + CLASSES[record["actual"]] + " · Predicted: " + CLASSES[predicted] + "</sup>"
        states.append(dict(record=record, traces=[trace.to_plotly_json()], layout=dict(title=title)))

    save_chart(fig, states, "perceptron.html", base_count=len(fig.data))
