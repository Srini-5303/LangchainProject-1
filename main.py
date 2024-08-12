import os
from sentence_transformers import SentenceTransformer
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


from dotenv import load_dotenv
load_dotenv()

st.title("News Research Tool")

st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_index_vector.pkl"

main_placefolder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500, model="gpt-3.5-turbo-instruct")

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)

    main_placefolder.text("Data Loading....Started")
    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n","\n",".", ","],
        chunk_size = 1000
    )

    main_placefolder.text("Text Splitting....Started")


    docs = splitter.split_documents(data)


    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)


    main_placefolder.text("Embedding Vector started building")
    time.sleep(2)

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

query = main_placefolder.text_input("Question : ")
if query:
    if os.path.exists:
        with open(file_path,"rb") as f:
            vectorstore = pickle.load(f)

            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever = vectorstore.as_retriever())
            result = chain({"question" : query}, return_only_outputs=True)

            st.header("Answer")
            st.write(result["answer"])

            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)