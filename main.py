"""Python file to serve as the frontend"""
import os
import datetime
import streamlit as st
from streamlit_chat import message
from langchain.document_loaders import PyPDFLoader
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.chains import ConversationChain
from langchain import PromptTemplate
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.agents import load_tools, initialize_agent
load_dotenv()
search = GoogleSearchAPIWrapper()
tools = load_tools(["google-search"])
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
@st.cache_resource
def load_chain() -> ConversationChain:
    """Logic for loading the chain you want to use should go here."""
    # template = "{history} let's think step by step"
    # prompt = PromptTemplate(input_variables=["history"], template=template)
    llm = OpenAI(temperature=0)
    chain = ConversationChain(llm=llm)
    return chain

chain: ConversationChain = load_chain()

# From here down is all the StreamLit UI.

st.header("LangChain Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []

search_button: bool = st.button("search")
with st.form(key="form", clear_on_submit=True):
    user_input: str = st.text_area("You: ", "", key="input_text", placeholder="please type here")
    submit: bool = st.form_submit_button("Submit")

# Implement the sidebar
with st.sidebar:
    st.header("Upload files")

    # Allow multiple files of any type to be uploaded
    new_uploaded_files = st.file_uploader("Upload multiple files", accept_multiple_files=True)

    # Button to append new uploaded files to the session state
    if st.button("Add files"):
        if new_uploaded_files:
            for file in new_uploaded_files:
                st.session_state.uploaded_files.append((file, datetime.datetime.now()))
        else:
            st.warning("No files selected for upload.")

if submit:
    output: str = chain.run(input=f"{user_input}. let's think step by step")

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if search_button:
    search_results: str = search.run("What is langchain?")   
    st.write(search_results)
if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

# Define target directory for saving files
target_directory = "uploaded_files"

if not os.path.exists(target_directory):
    os.makedirs(target_directory)

if st.session_state["uploaded_files"]:
    file_counter = 1
    for file, timestamp in st.session_state["uploaded_files"]:
        try:
            # Save the uploaded file to the target directory
            file_path = os.path.join(target_directory, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getvalue())

            # Display the file name, upload timestamp, and saved file path
            st.write(f"File {file_counter}: {file.name} (uploaded at {timestamp.strftime('%Y-%m-%d %H:%M:%S')})")
            st.write(f"Saved to: {file_path}")

        except Exception as e:
            st.error(f"Error processing file {file.name}: {str(e)}")

        file_counter += 1
else:
    st.write("No files uploaded.")