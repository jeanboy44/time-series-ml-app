# Core Pkgs
import streamlit as st
from app.prophet_forecast import prophet_forecast
from app.yolov5 import run_yolov5

# Additional Pkgs

# Import File/Data

# More Fxn

# Config Page
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
        run_yolov5()
    if choice == "Time Series Forecasting":
        prophet_forecast()


if __name__ == "__main__":
    main()
