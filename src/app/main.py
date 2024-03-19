# Core Pkgs
import streamlit as st
from yolov5_page import main as yolov5_page
from prophet_forecast_page import main as prophet_forecast_page

PAGE_CONFIG = {
    "page_title": "Machine Learning Application Demo",
    "page_icon": "😃",
    "layout": "wide",
}
st.set_page_config(**PAGE_CONFIG)


def main():
    st.title("ML model apps")
    menu = ["Home", "Object Detection", "Time Series Forecasting"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
    if choice == "Object Detection":
        yolov5_page()
    if choice == "Time Series Forecasting":
        prophet_forecast_page()


if __name__ == "__main__":
    main()
