import streamlit as st
import pandas as pd

from prophet.plot import plot_components_plotly
from app_utils import plot_plotly_with_testset
from prophet.serialize import model_from_json

URI = "file:///Users/jeanboy/workspace/time-series-ml-app/notebooks/mlruns"


def prepare_dataset():
    # Python
    df = pd.read_csv(
        "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv"
    )
    df_train = df[df.ds < "2015-01-01"]
    df_test = df[(df.ds >= "2015-01-01") & (df.ds < "2016-01-01")]

    return df_train, df_test


def load_model():
    with open("prophet_model.json", "r") as fin:
        m = model_from_json(fin.read())  # Load model

    return m


def predict_model(m, periods=365):
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    return forecast


def plot_results(model, forecast):
    _, df_test = prepare_dataset()
    st.subheader("Plot Prediction")
    st.plotly_chart(plot_plotly_with_testset(model, forecast, df_test))
    st.subheader("Plot Components")
    st.plotly_chart(plot_components_plotly(model, forecast))


def show_results():
    model = load_model()
    forecast = predict_model(model)
    plot_results(model, forecast)


def main():
    show_results()
