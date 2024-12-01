import streamlit as st
from src import chat_with_pdf
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Vitche | Chat de Atendimento ao Cliente")


class MultiApp:
    def __init__(self) -> None:
        self.apps = []

    def add_apps(self, title, function):
        self.apps.append({"title": title, "function": function})

    def run():
        with st.sidebar:
            st.image('./images/logo.png')
            app = option_menu(
                menu_title="Menu Vitche",
                # menu_icon='chat-text-fill',
                options=['Atendimento', ],
                icons=['house-fill', 'person-circle'],
                default_index=0,
                styles={
                    "container": {"padding": "5!important", "background-color": "black"},
                    "icon": {"color": "white", "font-size": "23px"},
                    "nav-link": {"color": "white", "font-size": "20px", "text-align": "left", "margin": "0px"},
                    "nav-link-selected": {"background-color": "#02ab21"},
                }
            )

        # if app=='Conta':
            # accounts.app()
        if app == 'Atendimento':
            chat_with_pdf.app()

    run()
