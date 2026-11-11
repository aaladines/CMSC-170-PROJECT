# CMSC 170: Fourcast - plotting module (plotly, no sklearn)
# One chart per metric (Accuracy, Precision, Recall, F1-Score), saved as its own HTML file.

import os # used to create the output folder

import plotly.graph_objects as plotly #plots the graph in html (for interactivity)

METRICS = ["Accuracy", "Precision", "Recall", "F1-Score"]
FILENAMES = ["accuracy", "precision", "recall", "f1_score"]
FONT_SIZE = 18   # base font size for all charts
OUTPUT_FOLDER = "tables_and_charts"
# styling for the graph
COLORS = {
    "Gaussian Naive Bayes": "#1f77b4",
    "K-Nearest Neighbors": "#2ca02c",
    "Perceptron": "#d62728",
}

# saves the files
SAVE_FILES = True


def plot_metric_per_year(scores, metric_index):
    # Line chart for one metric
    
    # the metric's name. for ex. "Accuracy"
    metric = METRICS[metric_index]
    fig = plotly.Figure() #starts the empty chart

    for name in scores:
        years = sorted(scores[name]) #gets the model's years and sorts it
        
        values = [] #metric value for each year

        for year in years: 
            values.append(scores[name][year][metric_index])

        # colors
        if name in COLORS:
            color = COLORS[name]
        else:
            color = "#7f7f7f"

        # Text shown when hovering over a point.
        hover_text = "<b>" + name + "</b><br>"
        hover_text = hover_text + "Year: %{x}<br>"
        hover_text = hover_text + metric + ": %{y:.2f}"
        hover_text = hover_text + "<extra></extra>"   # hides plotly's extra side box

        # Add the line (with a dot at each year) to the chart
        fig.add_trace(
            plotly.Scatter(
                x=years, # x-axis values
                y=values, # y-axis values
                mode="lines+markers", # draw lines and dots
                name=name, # label in the legend
                line=dict(color=color, width=3), # line color and thickness
                marker=dict(size=10), # dot size
                hovertemplate=hover_text,
            )
        )

    # titles, axes, fonts, and size
    fig.update_layout(
        title=metric + " per Year",
        xaxis=dict(title="Year", dtick=1), # dtick=1: one tick per year
        yaxis=dict(
            title=metric,
            range=[0, 1.05], # scores are 0 to 1, with a little room at the top
            dtick=0.1, # tick every 0.1
            tickformat=".2f", # show two decimal place
        ),
        font=dict(size=FONT_SIZE),
        title_font_size=FONT_SIZE + 6, # title is a bit bigger than other text
        height=650, # chart height in pixels
    )

    return fig


def show_all(scores, *unused):
    # makes the four metric charts

    # creates a folder
    if SAVE_FILES:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Go through the four metrics by position: 0, 1, 2, 3
    for i in range(len(FILENAMES)):
        filename = FILENAMES[i]

        # builds the chart for this metric
        fig = plot_metric_per_year(scores, i)

        if SAVE_FILES:
            # the .html file where the chart is shown
            path = OUTPUT_FOLDER + "/" + filename + ".html"
            fig.write_html(path, include_plotlyjs="cdn")
            print("Saved " + path)
        else:
            # open the chart in the browser instead of saving
            fig.show()