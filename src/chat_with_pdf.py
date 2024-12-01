import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from . import utils

DIRECTORY = 'Docs/'


def app():
    load_dotenv()

    if 'chain' not in st.session_state:
        st.session_state['chain'] = None
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'API_KEY' not in st.session_state:
        st.session_state['API_KEY'] = ''
    if 'db' not in st.session_state:
        st.session_state['db'] = None
    if 'unique_id' not in st.session_state:
        st.session_state['unique_id'] = ''
    if 'is_ready' not in st.session_state:
        st.session_state['is_ready'] = False
    st.session_state['API_KEY'] = os.getenv("OPENAI_API_KEY")

    st.title("Vitche 🌹 - Assistente de Atendimento ao Cliente🏪")
    # st.header("Faça o upload dos documentos pdf 📄")
    pdfs_uploaded = st.file_uploader("Carregue o guia de atendimento aqui e clica em 'Processar'",
                                     accept_multiple_files=True, type=["pdf"])
    btn_upload = st.button("Processar")

    if btn_upload:
        with st.spinner("Processando..."):
            st.session_state['unique_id'] = uuid.uuid4().hex

            # get pdf text
            docs = utils.create_docs(pdfs_uploaded, st.session_state['unique_id'])
            # docs = utils.load_docs(DIRECTORY)
            st.success("Documentos carregados✅")

            # get the text chunks
            chunks = utils.split_docs(docs)
            # chunks = utils.split_docs(docs)
            st.success("Chunks criados✅")

            # embeddings
            embeddings = utils.get_embeddings(st.session_state['API_KEY'])
            st.success("Embedding criados✅")

            # create vectorstore
            db = utils.get_vectorstore(chunks, embeddings)
            st.session_state['db'] = db
            st.success("Vector Store criado✅")

            # create conversation chain
            st.session_state['chain'] = utils.get_chain(st.session_state['API_KEY'])
            st.success("Chain criado✅")

            st.session_state['is_ready'] = True

    if st.session_state['is_ready']:
        response_container = st.container()
        container = st.container()

        with container:
            question = st.chat_input("Digite a sua pergunta:")
            if question:
                st.session_state['chat_history'].append(question)
                relevant_docs = utils.get_similar_docs(db=st.session_state['db'], query=question, k=1)
                print(f"{'='*200}\nDocumentos Base da resposta:{relevant_docs}")
                answer = utils.get_answer(chain=st.session_state['chain'], query=question, relevant_docs=relevant_docs)
                st.session_state['chat_history'].append(answer)

                with response_container:
                    for i in range(len(st.session_state['chat_history'])):
                        if (i % 2) == 0:
                            with st.chat_message("user"):
                                st.write(st.session_state['chat_history'][i])
                        else:
                            with st.chat_message("assistant"):
                                st.write(st.session_state['chat_history'][i])
