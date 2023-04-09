import streamlit as st
from langchain.llms import OpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import load_tools, initialize_agent, AgentExecutor

from dotenv import load_dotenv
load_dotenv()
st.set_page_config(page_title="LangChain Search Google Demo", page_icon=":robot:")

@st.cache_resource
def lead_agent()  -> AgentExecutor:
    llm = OpenAI(temperature=0)
    tools = load_tools(["google-search", "llm-math"], llm=llm)
    agent: AgentExecutor = initialize_agent(tools, llm, agent="zero-shot-react-description")

    return agent

agent = lead_agent()
search_button: bool = st.button("search")

if search_button:
    search_results: str = agent.run("What is langchain?")   
    st.write(search_results)