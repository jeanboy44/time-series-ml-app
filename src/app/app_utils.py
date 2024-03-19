import numpy as np
import plotly.graph_objs as go
import pandas as pd


def plot_plotly_with_testset(
    m,
    fcst,
    future_real=None,
    uncertainty=True,
    plot_cap=True,
    trend=False,
    changepoints=False,
    changepoints_threshold=0.01,
    xlabel="ds",
    ylabel="y",
    figsize=(900, 600),
):
    prediction_color = "#0072B2"
    error_color = "rgba(0, 114, 178, 0.2)"  # '#0072B2' with 0.2 opacity
    actual_color = "black"
    cap_color = "black"
    trend_color = "#B23B00"
    line_width = 2
    marker_size = 4

    data = []
    # Add actualforecast
    if future_real is None:
        x_real = m.history["ds"]
        y_real = m.history["y"]
        color_real = actual_color
    else:
        x_real = pd.concat([m.history["ds"], future_real["ds"]])
        y_real = pd.concat([m.history["y"], future_real["y"]])
        color_real = pd.Series(
            [actual_color] * len(m.history["ds"]) + ["red"] * len(future_real["y"])
        )

    data.append(
        go.Scatter(
            name="Actual",
            x=x_real,
            y=y_real,
            marker=dict(color=color_real, size=marker_size),
            mode="markers",
        )
    )
    # Add lower bound
    if uncertainty and m.uncertainty_samples:
        data.append(
            go.Scatter(
                x=fcst["ds"],
                y=fcst["yhat_lower"],
                mode="lines",
                line=dict(width=0),
                hoverinfo="skip",
            )
        )
    # Add prediction
    data.append(
        go.Scatter(
            name="Predicted",
            x=fcst["ds"],
            y=fcst["yhat"],
            mode="lines",
            line=dict(color=prediction_color, width=line_width),
            fillcolor=error_color,
            fill="tonexty" if uncertainty and m.uncertainty_samples else "none",
        )
    )
    # Add upper bound
    if uncertainty and m.uncertainty_samples:
        data.append(
            go.Scatter(
                x=fcst["ds"],
                y=fcst["yhat_upper"],
                mode="lines",
                line=dict(width=0),
                fillcolor=error_color,
                fill="tonexty",
                hoverinfo="skip",
            )
        )
    # Add caps
    if "cap" in fcst and plot_cap:
        data.append(
            go.Scatter(
                name="Cap",
                x=fcst["ds"],
                y=fcst["cap"],
                mode="lines",
                line=dict(color=cap_color, dash="dash", width=line_width),
            )
        )
    if m.logistic_floor and "floor" in fcst and plot_cap:
        data.append(
            go.Scatter(
                name="Floor",
                x=fcst["ds"],
                y=fcst["floor"],
                mode="lines",
                line=dict(color=cap_color, dash="dash", width=line_width),
            )
        )
    # Add trend
    if trend:
        data.append(
            go.Scatter(
                name="Trend",
                x=fcst["ds"],
                y=fcst["trend"],
                mode="lines",
                line=dict(color=trend_color, width=line_width),
            )
        )
    # Add changepoints
    if changepoints and len(m.changepoints) > 0:
        signif_changepoints = m.changepoints[
            np.abs(np.nanmean(m.params["delta"], axis=0)) >= changepoints_threshold
        ]
        data.append(
            go.Scatter(
                x=signif_changepoints,
                y=fcst.loc[fcst["ds"].isin(signif_changepoints), "trend"],
                marker=dict(
                    size=50,
                    symbol="line-ns-open",
                    color=trend_color,
                    line=dict(width=line_width),
                ),
                mode="markers",
                hoverinfo="skip",
            )
        )

    layout = dict(
        showlegend=False,
        width=figsize[0],
        height=figsize[1],
        yaxis=dict(title=ylabel),
        xaxis=dict(
            title=xlabel,
            type="date",
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all"),
                    ]
                )
            ),
            rangeslider=dict(visible=True),
        ),
    )
    fig = go.Figure(data=data, layout=layout)
    return fig
