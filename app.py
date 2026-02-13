import streamlit as st
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Load .env (if exists)
load_dotenv()

# LangSmith tracing (optional)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A CHATBOT WITH OPENAI"

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Question: {question}")
])

def generate_response(question, api_key, engine, temperature, max_tokens):
    # ✅ Set env var so LangChain can read it
    os.environ["OPENAI_API_KEY"] = api_key

    llm = ChatOpenAI(
        model=engine,
        temperature=temperature,
        max_tokens=max_tokens
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})


# ---------- Streamlit UI ----------

st.title("Q&A Chatbot with OpenAI")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
engine = st.sidebar.selectbox(
    "Select Model",
    ["gpt-3.5-turbo", "gpt-4"]
)
temperature = st.sidebar.slider("Select Temperature", 0.0, 1.0, 0.5)
max_tokens = st.sidebar.slider("Select Max Tokens", 1, 1024, 200)

st.write("Go ahead, ask any question 👇")
user_input = st.text_input("Enter your question:")

if user_input and api_key:
    response = generate_response(
        user_input,
        api_key,
        engine,
        temperature,
        max_tokens
    )
    st.write("### Response")
    st.write(response)
else:
    st.write("Please enter a question and API key.")
