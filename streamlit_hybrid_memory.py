# streamlit_hybrid_memory.py  (GROQ VERSION - ML + MEMORY) — PDF REMOVED

import os
import json
import random
import sqlite3
from datetime import datetime

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ------------------ GROQ CLIENT ------------------
from groq import Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ------------------ CONFIG ------------------
GROQ_MODEL = "llama-3.1-8b-instant"
ML_CONFIDENCE_THRESHOLD = 0.6
MEMORY_CONTEXT_MESSAGES = 6
DB_PATH = "chat_memory.db"
INTENTS_FILE = "intents.json"

# --------------- Load intents + ML training ----------------
with open(INTENTS_FILE, "r", encoding="utf-8") as f:
    intents = json.load(f)

patterns = []
tags = []
for intent in intents["intents"]:
    for p in intent["patterns"]:
        patterns.append(p)
        tags.append(intent["tag"])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)
model = LogisticRegression(max_iter=1000)
model.fit(X, tags)

def ml_response(text):
    X_test = vectorizer.transform([text])
    probs = model.predict_proba(X_test)[0]
    labels = model.classes_
    best_idx = probs.argmax()
    best_tag = labels[best_idx]
    best_prob = probs[best_idx]

    for it in intents["intents"]:
        if it["tag"] == best_tag:
            return random.choice(it["responses"]), best_prob, best_tag

    return None, 0.0, None

# ------------------ GROQ Chat Completion ------------------
def call_groq_chat(system_prompt, messages, model_name=GROQ_MODEL):
    try:
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        response = client.chat.completions.create(
            model=model_name,
            messages=full_messages,
            temperature=0.7,
            max_tokens=500
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Groq API error: {e}"

# ------------------ SQLite Memory ------------------
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            content TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    return conn

DB_CONN = init_db()

def save_message(role, content):
    ts = datetime.utcnow().isoformat()
    c = DB_CONN.cursor()
    c.execute("INSERT INTO messages (role, content, timestamp) VALUES (?, ?, ?)",
              (role, content, ts))
    DB_CONN.commit()

def get_recent_messages(limit=MEMORY_CONTEXT_MESSAGES):
    c = DB_CONN.cursor()
    c.execute("SELECT role, content FROM messages ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    return [{"role": r[0], "content": r[1]} for r in reversed(rows)]


# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Hybrid Groq Chatbot", page_icon="🤖")
st.title("🤖 Hybrid Chatbot — ML Intents + Groq + Memory")
st.write("Ask anything — ML detects intents, Groq handles reasoning, and chat memory persists.")

# -------- Chat UI --------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "loaded_memory" not in st.session_state:
    mem = get_recent_messages(50)
    for m in mem:
        st.session_state.messages.append({"role": m["role"], "content": m["content"]})
    st.session_state.loaded_memory = True

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# -------- Clear History Button --------
def clear_history():
    c = DB_CONN.cursor()
    c.execute("DELETE FROM messages")
    DB_CONN.commit()
    st.session_state.messages = []
    st.success("Chat history cleared!")

st.sidebar.button("🧹 Clear Chat History", on_click=clear_history)

# -------- User input --------
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    save_message("user", user_input)

    # -------- ML then Groq fallback --------
    ml_resp, prob, tag = ml_response(user_input)

    if prob >= ML_CONFIDENCE_THRESHOLD:
        bot_text = f"{ml_resp}  \n*(via ML intent `{tag}` — conf {prob:.2f})*"
    else:
        system_prompt = "You are a polite, helpful assistant."
        recent = get_recent_messages(MEMORY_CONTEXT_MESSAGES)

        chat_messages = []
        for m in recent:
            chat_messages.append({"role": m["role"], "content": m["content"]})
        chat_messages.append({"role": "user", "content": user_input})

        bot_text = call_groq_chat(system_prompt, chat_messages)

    st.session_state.messages.append({"role": "assistant", "content": bot_text})
    st.chat_message("assistant").write(bot_text)
    save_message("assistant", bot_text)
