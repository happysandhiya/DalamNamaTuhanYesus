import streamlit as st
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# openai / langchain Const
RETRIEVER_K_ARG = 3
OPENA_AI_MODEL = "gpt-3.5-turbo-1106" #"gpt-4-0314"
PRE_PROMPT_INSTRUCTIONS = "Use the context to answer the prompt"
PERSIST_DIRECTORY = "db"
HUGGINGFACE_MODEL = "sentence-transformers/all-mpnet-base-v2"
MODEL_KWARGS = {"device": "cuda"}

# Google Const
CLIENT_SECRET_FILE = r"C:\Users\Pluto_06\Documents\Happy\Chatbot\.credentials\credentials.json" #"credentials.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] =r"C:\Users\Pluto_06\Documents\Happy\Chatbot\.credentials\credentials.json"
TOKEN_FILE = r"C:\Users\Pluto_06\Documents\Happy\Chatbot\token.json"
# GOOGLE_DRIVER_FOLDER_ID = "1fUsasyT4uTrJzo2tnVLwbJ8_A3iDZh6r"
# GOOGLE_DRIVER_FOLDER_ID = "1M9Firi8Iqth5IPoUAhZnRu_5Z3BTMXqd"
GOOGLE_DRIVER_FOLDER_ID = "10vCfmWKC88nU4ETvoKE-bi-fo_gs5pck"

os.environ["OPENAI_API_KEY"] ="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"


def load_documents():
    loader = GoogleDriveLoader(
        credentials_path=CLIENT_SECRET_FILE,
        token_path=TOKEN_FILE,
        folder_id=GOOGLE_DRIVER_FOLDER_ID,
        recursive=False,
        file_types=["sheet", "document", "pdf"],
    )
    return loader.load()


def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=0, separators=[" ", ",", "\n"])
    return text_splitter.split_documents(docs)


def generate_embeddings():
    return HuggingFaceEmbeddings(model_name=HUGGINGFACE_MODEL, model_kwargs=MODEL_KWARGS)


def create_chroma_db(texts, embeddings):
    if not os.path.exists(PERSIST_DIRECTORY):
        return Chroma.from_documents(texts, embeddings, persist_directory=PERSIST_DIRECTORY)
    else:
        return Chroma(embedding_function=embeddings, persist_directory=PERSIST_DIRECTORY)


def create_retriever(db):
    return db.as_retriever(search_kwargs={"k": RETRIEVER_K_ARG})


def create_index(llm, retriever):
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


def create_llm():
    return ChatOpenAI(temperature=0, model_name=OPENA_AI_MODEL)


def main():
    st.title("Chatbot Application")

    docs_info = load_documents()
    st.header("Dokumen yang sudah ditambahkan dalam database chatbot:")
    seen_titles = set()  # Set untuk menyimpan judul-judul yang sudah muncul
    for doc_info in docs_info:
        title = doc_info.metadata.get('title')
        if title not in seen_titles:
            st.write(title)
            seen_titles.add(title)  # Menambah judul ke set
            
    st.subheader("Masukkan pertanyaan:")
    query_input = st.text_input("Pertanyaan:", key="query_input")

    if st.button("Submit"):
        docs = load_documents()
        texts = split_documents(docs)
        embeddings = generate_embeddings()
        db = create_chroma_db(texts, embeddings)
        retriever = create_retriever(db)
        llm = create_llm()
        qa = create_index(llm, retriever)

        answer = qa({"query": f"### Instructions. {PRE_PROMPT_INSTRUCTIONS} ###Prompt {query_input}"})
        st.write("Jawaban:", answer['result'])


if __name__ == "__main__":
    main()
