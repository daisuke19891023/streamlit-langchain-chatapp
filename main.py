"""Python file to serve as the frontend"""
import os
import datetime
import streamlit as st
from streamlit_chat import message
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.chains import ConversationChain, RetrievalQAWithSourcesChain, QAWithSourcesChain, ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")


@st.cache_resource
def load_chain() -> ConversationChain:
    """Logic for loading the chain you want to use should go here."""
    # template = "{history} let's think step by step"
    # prompt = PromptTemplate(input_variables=["history"], template=template)
    llm = OpenAI(temperature=0)
    chain = ConversationChain(llm=llm)
    return chain

@st.cache_resource
def load_docs():
    llm = OpenAI(temperature=0)
    loader = DirectoryLoader('uploaded_files/', glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    # index = VectorstoreIndexCreator().from_loaders([docs])
    # text_splitter = CharacterTextSplitter()
    # texts = text_splitter.split_documents(docs)
    # embeddings = OpenAIEmbeddings()
    # db: Chroma = Chroma.from_documents(texts, embeddings)
    # retriever = db.as_retriever(search_type="similarity", search_kwargs={"k:2"})
    # # chain = RetrievalQAWithSourcesChain(llm=llm, retriever=load_docs())
    # qa = ConversationalRetrievalChain(llm=llm, retriever=retriever)
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    return chain, docs

chain: ConversationChain = load_chain()

# From here down is all the StreamLit UI.

st.header("LangChain Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []



with st.form(key="form", clear_on_submit=True):
    user_input: str = st.text_area("You: ", "", key="input_text", placeholder="please type here")
    submit: bool = st.form_submit_button("Submit")



if submit:
    output: str = chain.run(input=f"{user_input}. let's think step by step")

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)


if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
