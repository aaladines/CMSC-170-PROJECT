# plot the fitted Naive Bayes distributions using the existing training split

import numpy as np
import plotly.graph_objects as plotly
from plotly.subplots import make_subplots

from data_loader import split_per_year
from gaussian_naive_bayes import make_gnb_features, calculate_mean_and_var, gaussian_pdf, class_priors, predict_increaase_decrease

from .common import record_info, save_chart


FEATURES = [
    "Previous-year log volume",
    "Prior-year annual change",
    "Prior-quarter annual change",
]
CLASSES = ["Decrease / unchanged (0)", "Increase (1)"]
COLORS = ["#1f77b4", "#2ca02c"]


def plot_gaussian_nb(mean_var, epsilon, year):
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=FEATURES,
        vertical_spacing=0.13,
    )

    for dimension, feature in enumerate(FEATURES):
        means = [mean_var[label]["mean"][dimension] for label in (0, 1)]
        variances = [mean_var[label]["var"][dimension] + epsilon for label in (0, 1)]
        deviations = np.sqrt(variances)

        low = min(mean - 4 * deviation for mean, deviation in zip(means, deviations))
        high = max(mean + 4 * deviation for mean, deviation in zip(means, deviations))

        # add samples around each mean so narrow curves stay smooth
        values = list(np.linspace(low, high, 200))

        for mean, deviation in zip(means, deviations):
            values.extend(np.linspace(mean - 4 * deviation, mean + 4 * deviation, 100))

        values = np.array(sorted(values))

        for label, name in enumerate(CLASSES):
            densities = gaussian_pdf(values, means[label], variances[label])

            fig.add_trace(
                plotly.Scatter(
                    x=values.tolist(),
                    y=densities.tolist(),
                    mode="lines",
                    name=name,
                    legendgroup=str(label),
                    showlegend=dimension == 0,
                    line=dict(color=COLORS[label], width=3),
                    hovertemplate=(
                        feature + ": %{x:.3f}<br>Density: %{y:.4f}"
                        + "<extra>" + name + "</extra>"
                    ),
                ),
                row=dimension + 1,
                col=1,
            )

        fig.update_xaxes(title_text=feature, row=dimension + 1, col=1)
        fig.update_yaxes(title_text="Density", rangemode="tozero", row=dimension + 1, col=1)

    fig.update_layout(
        title=dict(
            text=f"<b>Gaussian Naive Bayes distributions ({year})</b>",
            x=0,
            xanchor="left",
            y=0.98,
            yanchor="top",
        ),
        legend=dict(orientation="h", x=0, y=1.06, yanchor="bottom"),
        font=dict(family="system-ui, sans-serif", size=14),
        title_font_size=20,
        height=1200,
        margin=dict(t=150, r=30, b=75, l=70),
    )

    return fig


def show_all(data, *, gnb_smoothing):
    data_table = {
        (record["crop"], record["region"], int(record["year"]), int(record["quarter"])): (
            float(record["same_quarter_previous_year"]),
            float(record["this_quarter"]),
            int(record["production"]),
        )
        for record in data
    }

    features, labels, years = make_gnb_features(data_table)
    year = int(max(years))
    training, testing, _, test_indices = split_per_year(features, labels, years, year)
    mean_var = calculate_mean_and_var(training["features"], training["labels"])

    all_variances = np.concatenate([mean_var[label]["var"] for label in (0, 1)])
    epsilon = gnb_smoothing * all_variances.max()

    fig = plot_gaussian_nb(mean_var, epsilon, year)
    predicted = predict_increaase_decrease(testing["features"], mean_var, class_priors(mean_var), epsilon)
    states = []

    for index, source_index in enumerate(test_indices):
        record = record_info(data[int(source_index)])
        traces = []
        shapes = []

        for dimension, value in enumerate(testing["features"][index]):
            xref = "x" + (str(dimension + 1) if dimension else "")
            yref = "y" + (str(dimension + 1) if dimension else "")

            for label, name in enumerate(CLASSES):
                density = gaussian_pdf(value, mean_var[label]["mean"][dimension], mean_var[label]["var"][dimension] + epsilon)
                traces.append(plotly.Scatter(
                    x=[float(value)], y=[float(density)],
                    mode="markers", name=name, showlegend=False,
                    xaxis=xref, yaxis=yref,
                    marker=dict(color=COLORS[label], size=9),
                    hovertemplate=FEATURES[dimension] + ": %{x:.3f}<br>Density: %{y:.4f}<extra>" + name + "</extra>",
                ).to_plotly_json())

            shapes.append(dict(
                type="line", xref=xref, yref=yref + " domain",
                x0=float(value), x1=float(value), y0=0, y1=1,
                line=dict(color="#263238", width=1.5, dash="dash"),
            ))

        title = fig.layout.title.to_plotly_json()
        title["text"] += "<br><sup>Actual: " + CLASSES[record["actual"]] + " · Predicted: " + CLASSES[int(predicted[index])] + "</sup>"
        states.append(dict(record=record, traces=traces, layout=dict(shapes=shapes, title=title)))

    save_chart(fig, states, "gaussian_nb.html", base_count=6)
