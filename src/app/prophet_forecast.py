import streamlit as st
import pandas as pd
from prophet import Prophet

# from app.utils import plot_plotly
import mlflow
from pathlib import Path
from prophet.plot import plot_plotly, plot_components_plotly
from app.utils import plot_plotly_with_testset
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository

URI = "file:///Users/jeanboy/workspace/time-series-ml-app/notebooks/mlruns"


def prepare_dataset():
    # Python
    df = pd.read_csv(
        "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv"
    )
    df_train = df[df.ds < "2015-01-01"]
    df_test = df[(df.ds >= "2015-01-01") & (df.ds < "2016-01-01")]

    return df_train, df_test


def download_model(local_dir):
    mlflow.set_tracking_uri(URI)
    ModelsArtifactRepository("models:/prophet_model/latest").download_artifacts(
        "", dst_path=local_dir
    )


def load_model():
    local_dir = Path("tmp")
    if not local_dir.is_dir():
        local_dir.mkdir(parents=True)

    download_model(local_dir)
    return mlflow.prophet.load_model(local_dir)


def predict_model(m, periods=365):
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    return forecast


def plot_results(model, forecast):

    st.subheader("Plot Prediction")
    st.plotly_chart(plot_plotly(model, forecast))
    st.subheader("Plot Components")
    st.plotly_chart(plot_components_plotly(model, forecast))


def plot_comparison(model, forecast):
    _, df_test = prepare_dataset()
    st.plotly_chart(
        plot_plotly_with_testset(model, forecast, df_test), use_container_width=True
    )


def show_results():
    model = load_model()
    forecast = predict_model(model)
    plot_results(model, forecast)


def show_comparision():
    rmse_ = 0.12411
    with st.chat_message("ai"):
        st.write(f"RMSE: {rmse_}")
    model = load_model()
    forecast = predict_model(model)
    plot_comparison(model, forecast)


def prophet_forecast():
    st.sidebar.markdown("----")
    menu = ["Show Results", "Compare Model"]
    choice = st.sidebar.selectbox("Choose", menu)

    if choice == "Show Results":
        show_results()
    if choice == "Compare Model":
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox("model A", ["model1", "model2"])
            show_comparision()
        with col2:
            st.selectbox("model B", ["model1", "model2"])
            show_comparision()
