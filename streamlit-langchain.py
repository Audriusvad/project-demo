from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import os
from langchain import hub

from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader, TextLoader
import bs4  # BeautifulSoup for parsing HTML

load_dotenv()  # take environment variables

# from .env file
# Load environment variables from .env file

token = os.getenv("SECRET")  # Replace with your actual token
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-nano"

# Load all document sources
file_loader = TextLoader("project-demo/data/hometown_info.txt", encoding="utf-8")
file_docs = file_loader.load()

# Add Anykščiai info as a new file source
anyksciai_path = "project-demo/data/anyksciai_info.txt"
if not os.path.exists(anyksciai_path):
    with open(anyksciai_path, "w", encoding="utf-8") as f:
        f.write("Anykščiai yra miestas šiaurės rytų Lietuvoje, Aukštaitijos regione, prie Šventosios upės. Miestas garsėja savo gamta, kultūros paveldu, literatūros ir menų tradicijomis. Anykščiai žinomi dėl Puntuko akmens, Anykščių šilelio, siauruko geležinkelio, bažnyčios su apžvalgos bokštu ir daugybės muziejų. Tai populiari turistų lankoma vieta, ypač vasarą.")

anyksciai_loader = TextLoader(anyksciai_path, encoding="utf-8")
anyksciai_docs = anyksciai_loader.load()


wiki_loader = WebBaseLoader(
    web_paths=("https://en.wikipedia.org/wiki/Anykščiai",)
)
wiki_docs = wiki_loader.load()

# Function to filter sources based on user selection
def filter_docs(selected_sources):
    docs = []
    if "File" in selected_sources:
        docs += file_docs
    if "Wikipedia" in selected_sources:
        docs += wiki_docs
    return docs

# Formatter to separate source contents
def format_docs(docs):
    # Show up to 3 different chunks (documents) from the selected sources
    if not docs:
        return "_No content found from selected sources._"
    output = []
    for i, doc in enumerate(docs[:3]):
        src = doc.metadata.get("source", "Unknown source")
        output.append(f"### Chunk {i+1} from {src}\n" + doc.page_content[:1000])  # Show up to 1000 chars per chunk
    return "\n\n".join(output)

st.title("Streamlit LangChain Demo - 3 chunk")

def generate_response(input_text, sources_selected):
    # Step 1: Filter documents
    selected_docs = filter_docs(sources_selected)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    splits = text_splitter.split_documents(selected_docs)
    # Step 2: Create vectorstore and retriever
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(
            model="text-embedding-3-small",
            base_url="https://models.inference.ai.azure.com",
            api_key=token,
        ),
    )
    retriever = vectorstore.as_retriever()
    # Step 3: Set up LLM and RAG chain
    llm = ChatOpenAI(
        base_url=endpoint,
        temperature=0.7,
        api_key=token,
        model=model
    )
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | hub.pull("rlm/rag-prompt")
        | llm
        | StrOutputParser()
    )
    # Step 4: Show result
    return rag_chain.invoke(input_text), selected_docs

with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "Anykščių istorija",
    )
    sources_selected = st.multiselect(
        "Select sources:",
        ["File", "Anykščiai", "Web", "Wikipedia"],
        default=["File", "Anykščiai", "Web", "Wikipedia"]
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        response, used_docs = generate_response(text, sources_selected)
        st.info(response)
        # Show the sources
        st.markdown("### Sources used:")
        st.markdown(format_docs(used_docs))