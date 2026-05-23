from pathlib import Path

import ollama
import streamlit as st


DATA_PATH = Path("cricket_150_lines.txt")
EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
LANGUAGE_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"


def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x**2 for x in a) ** 0.5
    norm_b = sum(y**2 for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


@st.cache_data(show_spinner=False)
def load_dataset(path: str):
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} was not found.")
    return [
        line.strip()
        for line in data_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def get_embedding(text):
    response = ollama.embed(model=EMBEDDING_MODEL, input=text)
    embeddings = response.get("embeddings", [])
    if not embeddings:
        raise ValueError("Ollama returned no embedding. Try a non-empty input.")
    return embeddings[0]


@st.cache_resource(show_spinner=False)
def build_vector_db(chunks):
    vector_db = []
    for chunk in chunks:
        vector_db.append((chunk, get_embedding(chunk)))
    return vector_db


def retrieve(query, vector_db, top_n=3):
    query = query.strip()
    if not query:
        raise ValueError("Please enter a question.")

    query_embedding = get_embedding(query)
    similarities = [
        (chunk, cosine_similarity(query_embedding, embedding))
        for chunk, embedding in vector_db
    ]
    similarities.sort(key=lambda item: item[1], reverse=True)
    return similarities[:top_n]


def stream_answer(query, retrieved_knowledge):
    context = "\n".join(f"- {chunk}" for chunk, _ in retrieved_knowledge)
    instruction_prompt = f"""You are a helpful chatbot.
Use only the following pieces of context to answer the question.
Do not make up any new information.

Context:
{context}
"""

    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {"role": "system", "content": instruction_prompt},
            {"role": "user", "content": query},
        ],
        stream=True,
    )

    for chunk in stream:
        yield chunk["message"]["content"]


st.set_page_config(page_title="Cricket Facts RAG", page_icon="C", layout="wide")

st.title("Cricket Facts RAG")
st.caption("Ask a question and get an answer grounded in your local cricket facts dataset.")

with st.sidebar:
    st.header("Settings")
    top_n = st.slider("Retrieved facts", min_value=1, max_value=8, value=3)
    st.text_input("Embedding model", value=EMBEDDING_MODEL, disabled=True)
    st.text_input("Language model", value=LANGUAGE_MODEL, disabled=True)
    if st.button("Clear cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

try:
    dataset = load_dataset(str(DATA_PATH))
    st.success(f"Loaded {len(dataset)} facts from {DATA_PATH}.")
except Exception as error:
    st.error(f"Could not load dataset: {error}")
    st.stop()

try:
    with st.spinner("Building vector database with Ollama embeddings..."):
        vector_db = build_vector_db(tuple(dataset))
except Exception as error:
    st.error(
        "Could not build the vector database. Make sure the Ollama app/server is "
        "running and the embedding model is available."
    )
    st.exception(error)
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask about cricket...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    try:
        retrieved_knowledge = retrieve(query, vector_db, top_n=top_n)
    except Exception as error:
        with st.chat_message("assistant"):
            st.error(f"Retrieval failed: {error}")
        st.stop()

    with st.expander("Retrieved knowledge", expanded=False):
        for chunk, similarity in retrieved_knowledge:
            st.markdown(f"**{similarity:.2f}** - {chunk}")

    with st.chat_message("assistant"):
        response = st.write_stream(stream_answer(query, retrieved_knowledge))

    st.session_state.messages.append({"role": "assistant", "content": response})
