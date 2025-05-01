import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

st.set_page_config(
    page_title="FullstackGPT Home",
    page_icon="🍖"
)


st.title("Code Challenge!!!")
st.markdown("""
Welcome!

Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
""")
with st.sidebar:
    openai_api_key = st.sidebar.text_input(
    "🔑 Enter your OpenAI API Key", type="password"
)
    file = st.file_uploader("Upload a .txt, .pdf, or .docx file", type=["txt", "pdf", "docx"])


if not openai_api_key:
    st.warning("Enter your OpenAI API KEY.")
    st.stop()

llm = ChatOpenAI(temperature=0.1, streaming=True, openai_api_key=openai_api_key)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
@st.cache_data(show_spinner="Embdeeing file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    loader = UnstructuredFileLoader(file_path)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(separator="\n", chunk_size=600, chunk_overlap=100)
    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings()
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    return vectorstore.as_retriever()

condense_prompt = PromptTemplate.from_template(
    """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
    
Chat History:
{chat_history}
Follow-up question: {question}
"""
)


if file:
    retriever = embed_file(file)

    # Conversational RAG 체인 구성
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=condense_prompt,
        return_source_documents=False
    )

    # 대화 상태 초기화
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # 이전 메시지 출력
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 입력 받기
    user_input = st.chat_input("Enter the Question...")
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            response = chain.invoke({"question": user_input})
            st.markdown(response["answer"])
            st.session_state["messages"].append({"role": "assistant", "content": response["answer"]})

else:
    st.info("Please upload a document to begin.")