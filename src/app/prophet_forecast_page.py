import streamlit as st
import pandas as pd
import mlflow
from prophet.plot import plot_components_plotly
from app_utils import plot_plotly_with_testset
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from pathlib import Path
from typing import Dict
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

URI = "file:///Users/jeanboy/workspace/time-series-ml-app/notebooks/mlruns"


def prepare_dataset():
    # Python
    df = pd.read_csv(
        "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv"
    )
    df_train = df[df.ds < "2015-01-01"]
    df_test = df[(df.ds >= "2015-01-01") & (df.ds < "2016-01-01")]

    return df_train, df_test


def download_model(local_dir, version):
    mlflow.set_tracking_uri(URI)
    ModelsArtifactRepository(f"models:/prophet_model/{version}").download_artifacts(
        "", dst_path=local_dir
    )


def list_models(model_name="prophet_model"):
    models_list: Dict[str, ModelVersion] = {}
    for mv in mlflow.search_model_versions(filter_string=f"name='{model_name}'"):
        models_list[mv.version] = mv

    return models_list


def load_model(version):
    local_dir = Path("tmp")
    if not local_dir.is_dir():
        local_dir.mkdir(parents=True)

    download_model(local_dir, version)
    return mlflow.prophet.load_model(local_dir)


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


def plot_comparison(model, forecast):
    _, df_test = prepare_dataset()
    st.plotly_chart(
        plot_plotly_with_testset(model, forecast, df_test), use_container_width=True
    )


def show_results(mv):
    model = load_model(mv.version)
    forecast = predict_model(model)
    plot_results(model, forecast)


def show_comparision(mv):
    client = MlflowClient()
    run_info = client.get_run(mv.run_id)
    with st.chat_message("ai"):
        st.write(f"RMSE: {run_info.data.metrics['rmse']}")
    model = load_model(mv.version)
    forecast = predict_model(model)
    plot_comparison(model, forecast)


def main():
    st.sidebar.markdown("----")
    menu = ["Show Results", "Compare Model"]
    choice = st.sidebar.selectbox("Choose", menu)
    models_list = list_models()
    if st.button("Refresh"):
        models_list = list_models()

    if choice == "Show Results":
        model_version = st.selectbox("model version", models_list.keys())
        show_results(mv=models_list.get(model_version))
    if choice == "Compare Model":
        col1, col2 = st.columns(2)
        with col1:
            model_a_version = st.selectbox("model A", models_list.keys())
            show_comparision(mv=models_list.get(model_a_version))
        with col2:
            model_b_version = st.selectbox("model B", models_list.keys())
            show_comparision(mv=models_list.get(model_b_version))
