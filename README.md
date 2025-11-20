📘 Hybrid Groq Chatbot — ML Intents + LLM + Memory (Streamlit App)

A lightweight, fast, and intelligent chatbot built using:

Groq Llama-3.1-8B Instant (super-fast inference)

Machine Learning Intent Classifier (TF-IDF + Logistic Regression)

Persistent Chat Memory (SQLite)

Streamlit Chat UI

This chatbot combines ML intent detection + LLM reasoning to provide fast, accurate responses with conversation memory.

🚀 Features
🧠 1. Machine Learning Intent Detection

Detects user intent using:

TfidfVectorizer

Logistic Regression

Logic:

If ML confidence ≥ 0.6 → returns instant ML-based predefined response.

Else → fallback to Groq LLM.

⚡ 2. Groq LLM (Llama-3.1-8B Instant)

Used for:

Natural replies

Reasoning

Open-ended questions

General conversation

Anything not matched by ML

Groq provides extremely low latency responses.

💾 3. SQLite Persistent Chat Memory

Stores messages in:

chat_memory.db


Memory survives restarts

Last 50 messages loaded on startup

Only last 6 messages used as LLM context

💬 4. Clean Streamlit Chat UI

Chat bubbles (User + Assistant)

Sidebar button to Clear Chat History

Lightweight, modern, fast

📁 Project Structure
chatbot/
│
├── streamlit_hybrid_memory.py       # Main chatbot file
├── intents.json                     # ML intent dataset
├── chat_memory.db                   # Auto-generated DB (ignored in Git)
├── requirements.txt                 # Dependencies
└── venv/                            # Virtual environment (ignored in Git)

🔧 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/chatbot.git
cd chatbot

2️⃣ Create Virtual Environment
python -m venv venv


Activate it:

Windows:
venv\Scripts\activate

Linux/Mac:
source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Add Your Groq API Key
Windows:
setx GROQ_API_KEY "your_api_key"

Mac/Linux:
export GROQ_API_KEY="your_api_key"

▶️ Run the Chatbot
streamlit run streamlit_hybrid_memory.py


Open in browser:

👉 http://localhost:8501

🧹 Clear Chat History

In the sidebar:

🧹 Clear Chat History


This will:

Delete all rows from SQLite

Reset Streamlit's message history

🧪 ML Intents (intents.json)

You can freely add:

New intents

New patterns

New responses

Example:
{
  "tag": "greeting",
  "patterns": ["hi", "hello", "hey"],
  "responses": ["Hello! How can I help you today?"]
}


The ML model re-trains automatically on app start.

🧩 Tech Stack
Component	Technology
UI	Streamlit
LLM	Groq Llama-3.1-8B Instant
ML Model	TF-IDF + Logistic Regression
Memory	SQLite
Language	Python
📌 Future Enhancements (Optional)

Features that can be added:

🔊 Voice input + output

🌙 Dark mode UI

🔄 Groq Realtime streaming

🧠 Vector-based long-term memory

🤖 Multiple bot personalities

If you want any of these, just ask!

⭐ If you like this project, give it a star on GitHub! ⭐
