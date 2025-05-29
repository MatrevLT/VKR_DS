import streamlit as st
from analysis_and_model import analysis_and_model_page
from presentation import presentation_page
# Настройка навигации
pages = {
 "Анализ и модель": analysis_and_model_page,
 "Презентация": presentation_page,
}
# Отображение навигации
selected_page = st.sidebar.selectbox("Навигация", list(pages.keys()))
pages[selected_page]()
