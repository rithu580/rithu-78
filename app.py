# app.py
import streamlit as st
import os
import requests

st.set_page_config(page_title="100x Voice/Chat (Streamlit)", layout="centered")

st.title("100x Voicebot — Streamlit demo")
st.write("This app sends text to OpenAI via a secure key stored in Streamlit secrets (or environment).")

# get OpenAI key (Streamlit Cloud: st.secrets, local: env)
OPENAI_KEY = None
if "OPENAI_API_KEY" in st.secrets:
    OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
else:
    OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_KEY:
    st.error("OpenAI API key not found. On Streamlit Cloud add it to Secrets or set environment variable locally.")
    st.stop()

# simple chat UI
if "history" not in st.session_state:
    st.session_state.history = []

with st.form("input_form", clear_on_submit=True):
    user_input = st.text_input("Type your message or question", key="input")
    submit = st.form_submit_button("Send")
if submit and user_input:
    st.session_state.history.append({"role": "user", "content": user_input})

# show history
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")

# call OpenAI when there's a new user message without a bot reply
def call_openai(messages):
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": "gpt-4o-mini",  # change if you don't have access
        "messages": messages,
        "max_tokens": 400,
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_KEY}",
        "Content-Type": "application/json"
    }
    r = requests.post(url, json=payload, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    # read assistant message (defensive)
    assistant = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return assistant

# If last message is user and there is no assistant reply yet, call OpenAI
if st.session_state.history:
    last = st.session_state.history[-1]
    # if last is user -> generate bot reply
    if last["role"] == "user":
        with st.spinner("Generating reply..."):
            try:
                # build messages in simple format OpenAI expects
                messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.history]
                assistant_text = call_openai(messages)
            except Exception as e:
                st.error(f"API request failed: {e}")
                assistant_text = "Sorry — API error."
        st.session_state.history.append({"role": "assistant", "content": assistant_text})
        st.experimental_rerun()
