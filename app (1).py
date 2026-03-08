"""
Autism Information Chatbot - Streamlit Version
RAG system using Qwen2.5-1.5B-Instruct + FAISS + BGE embeddings
"""

import os
import gc
import re
import pickle
import numpy as np
import torch
import faiss
import streamlit as st
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_MODEL_NAME  = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_MODEL_PATH  = "./lora_weights"
FAISS_INDEX_PATH = "./faiss_index.bin"
CHUNKS_PATH      = "./pdf_chunks.pkl"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

REFUSAL_STRING = (
    "I do not understand this question, and the topic you mentioned "
    "is not present in the provided PDF corpus."
)

FORBIDDEN_TOPICS = [
    'capital', 'france', 'paris', 'london', 'berlin', 'tokyo', 'beijing',
    'invented', 'inventor', 'discovered', 'created', 'founded',
    'tire', 'tyre', 'wheel', 'car', 'vehicle', 'automobile',
    'recipe', 'cook', 'bake', 'ingredient', 'kitchen',
    'stock', 'price', 'share', 'market', 'dollar', 'euro',
    'population', 'people live', 'how many people',
    'president', 'prime minister', 'leader',
    'sport', 'game', 'match', 'super bowl', 'world cup',
    'weather', 'temperature', 'forecast', 'rain', 'sunny',
]

AUTISM_KEYWORDS = [
    'autism', 'asd', 'asperger', 'spectrum', 'disorder',
    'developmental', 'behavior', 'child', 'treatment',
    'therapy', 'symptom', 'diagnosis', 'kanner', 'rimland',
    'sensory', 'communication', 'social', 'repetitive',
]

# ─────────────────────────────────────────────
# LOAD MODELS (cached so they only load once)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models, please wait (this may take a few minutes)...")
def load_all():
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    if Path(LORA_MODEL_PATH).exists():
        model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
        model = model.merge_and_unload()
    else:
        model = base_model

    model.eval()
    gc.collect()

    return embedder, index, chunks, tokenizer, model


# ─────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────
def retrieve(question, embedder, index, chunks, k=5):
    instruction = "Represent this sentence for searching relevant passages: "
    q_emb = embedder.encode(
        [instruction + question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    _, indices = index.search(q_emb, k)
    return " ".join(chunks[i] for i in indices[0])


# ─────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────
def ask_qwen(question, context, tokenizer, model):
    question_lower = question.lower()
    has_forbidden = any(t in question_lower for t in FORBIDDEN_TOPICS)
    has_autism    = any(t in question_lower for t in AUTISM_KEYWORDS)

    if has_forbidden and not has_autism and len(context.strip()) < 200:
        return REFUSAL_STRING

    system_prompt = f"""You are a STRICT PDF assistant. Your ONLY source of knowledge is the Context below.

ABSOLUTE RULES:
1. If the Context does NOT contain enough information, respond EXACTLY with:
   "{REFUSAL_STRING}"
2. Never use outside knowledge.
3. Never fabricate information.
4. When in doubt, always use the refusal message.

Context:
{context}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    if has_forbidden and not has_autism and len(response.split()) < 15:
        return REFUSAL_STRING

    return response


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Autism Information Chatbot",
    page_icon="🧩",
    layout="centered",
)

st.title("🧩 Autism Information Chatbot")
st.caption(
    "Ask questions about Autism Spectrum Disorder. "
    "Answers are grounded exclusively in research PDFs."
)
st.warning(
    "⚠️ This chatbot is for **informational purposes only** and is "
    "not a substitute for professional medical advice.",
    icon="⚠️",
)

embedder, index, chunks, tokenizer, model = load_all()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

with st.sidebar:
    st.header("💡 Example Questions")
    examples = [
        "What is autism?",
        "What are the main symptoms of autism?",
        "How is autism diagnosed?",
        "What treatments are available?",
        "What are early signs of autism in children?",
        "What is Asperger's syndrome?",
        "What causes autism?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.prefill = ex

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

prefill = st.session_state.pop("prefill", None)
user_input = st.chat_input("Ask a question about autism...") or prefill

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching PDFs and generating answer..."):
            context  = retrieve(user_input, embedder, index, chunks)
            response = ask_qwen(user_input, context, tokenizer, model)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
