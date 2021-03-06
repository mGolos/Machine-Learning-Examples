import streamlit as st
import examples
from examples.utils import st_query_radio
from examples.question_tagging.app import SimpleModel


def main():
    st_query_radio("Menu", "p", {
        "Acceuil": examples.home,
        "Question Tagging": examples.question_tagging,
        "Breed Classifier": examples.breed_classifier,
    })()


if __name__ == "__main__":
    st.set_page_config(
        page_title='MLE',  # String or None. Strings get appended with "• Streamlit". 
        initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
        page_icon=None,   # String, anything supported by st.image, or None.
        layout="centered"  # Can be "centered" or "wide". In the future also "dashboard", etc.
    )
    st.sidebar.image('hands.png')
    with open('.streamlit/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        
    main()
    st.sidebar.markdown("[Portefolio](https://share.streamlit.io/mgolos/portfolio/main/main.py)")