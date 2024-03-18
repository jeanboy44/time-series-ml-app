# Core Pkgs
import streamlit as st
from app.prophet_forecast import prophet_forecast

# Additional Pkgs

# Import File/Data

# More Fxn

# Config Page
PAGE_CONFIG = {
    "page_title": "StreamlitTutorial",
    "page_icon": "😃",
    "layout": "centered",
}
st.set_page_config(**PAGE_CONFIG)


def main():
    st.title("Streamlit is Awesome")
    menu = ["Home", "Prophet", "Applications"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Prophet":
        prophet_forecast()


if __name__ == "__main__":
    main()
