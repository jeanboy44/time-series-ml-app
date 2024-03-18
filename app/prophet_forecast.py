import streamlit as st
import pandas as pd
from prophet import Prophet

from app.utils import plot_plotly


def prepare_dataset():
    # Python
    df = pd.read_csv(
        "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv"
    )
    df_train = df[df.ds < "2015-01-01"]
    df_test = df[(df.ds >= "2015-01-01") & (df.ds < "2016-01-01")]

    return df_train, df_test


def train_model(df_train):
    m = Prophet()
    m.fit(df_train)

    return m


def predict_model(m, periods=365):
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    return forecast


def plot(model, forecast, df_test):
    return plot_plotly(model, forecast, future_real=df_test)


# Config Page
def prophet_forecast():
    df_train, df_test = prepare_dataset()
    model = train_model(df_train)
    forecast = predict_model(model)
    fig = plot(model, forecast, df_test)
    st.plotly_chart(fig, theme="streamlit")
