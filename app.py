# app.py
# app.py
import streamlit as st
import os
import requests
import time
import logging

# --- basic logging to Streamlit logs ---
logger = logging.getLogger("100x-voicebot")
logger.setLevel(logging.INFO)

st.set_page_config(page_title="100x Voice/Chat (Streamlit)", layout="centered")

st.title("100x Voicebot — Streamlit demo")
st.write("This app sends text to OpenAI via a secure key stored in Streamlit secrets (or environment).")

# get OpenAI key (Streamlit Cloud: st.secrets, local: env)
OPENAI_KEY = None
try:
    OPENAI_KEY = st.secrets.get("OPENAI_API_KEY") if st.secrets else None
except Exception:
    OPENAI_KEY = None
if not OPENAI_KEY:
    OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_KEY:
    st.error("OpenAI API key not found. On Streamlit Cloud add it to Secrets or set environment variable locally.")
    st.stop()

# initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "awaiting_reply" not in st.session_state:
    st.session_state.awaiting_reply = False

# UI: input form
with st.form("input_form", clear_on_submit=True):
    user_input = st.text_input("Type your message or question", key="input")
    submit = st.form_submit_button("Send")
if submit and user_input:
    # append user message and mark that we need a reply
    st.session_state.history.append({"role": "user", "content": user_input})
    # set flag so we will call OpenAI exactly once for this new message
    st.session_state.awaiting_reply = True

# show history (user + assistant messages)
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")

# --- resilient OpenAI caller with backoff (retries on 429) ---
def call_openai_with_backoff(messages, api_key, model="gpt-4o-mini", max_retries=5, initial_backoff=1.0):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 400,
    }
    backoff = initial_backoff
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as http_err:
            status = getattr(http_err.response, "status_code", None)
            logger.warning(f"OpenAI HTTPError status={status} attempt={attempt}: {http_err}")
            # retry on rate limit (429)
            if status == 429 and attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2
                continue
            # for other statuses, raise to caller so it can show a friendly message
            raise
        except requests.RequestException as req_err:
            # network / timeout — retry a few times
            logger.warning(f"RequestException attempt={attempt}: {req_err}")
            if attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise

# If we have a pending user message awaiting a reply, call OpenAI (once)
if st.session_state.awaiting_reply and st.session_state.history:
    # find last message and ensure it's a user message
    last = st.session_state.history[-1]
    if last["role"] == "user":
        with st.spinner("Generating reply..."):
            try:
                # prepare messages for OpenAI from history
                messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.history]
                # call with backoff
                data = call_openai_with_backoff(messages, OPENAI_KEY)
                assistant_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if not assistant_text:
                    assistant_text = "Sorry — I couldn't generate a response."
                # append assistant reply to history
                st.session_state.history.append({"role": "assistant", "content": assistant_text})
            except requests.HTTPError as http_err:
                # handle HTTP errors gracefully
                status = getattr(http_err.response, "status_code", None)
                if status == 429:
                    st.error("Too many requests to the OpenAI API right now. Please wait a few seconds and try again.")
                else:
                    st.error(f"API request failed (HTTP {status}). Please try again later.")
                logger.exception("OpenAI HTTP error")
                # append an informative assistant message instead of crashing
                st.session_state.history.append({"role": "assistant", "content": "Sorry — API error. Please try again later."})
            except Exception as exc:
                # unexpected error
                st.error("An unexpected error occurred while contacting the OpenAI API. Check logs for details.")
                logger.exception("Unexpected error calling OpenAI")
                st.session_state.history.append({"role": "assistant", "content": "Sorry — unexpected server error."})
            finally:
                # clear the awaiting flag so we don't call again for the same message
                st.session_state.awaiting_reply = False
                # NOTE: do NOT call st.experimental_rerun() here — Streamlit will naturally rerender with new history

       
