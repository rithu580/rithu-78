# app.py
import streamlit as st
import os
import requests
import time
import logging
import tempfile
from gtts import gTTS

# ---------- config ----------
PAGE_TITLE = "100x Voicebot — Streamlit"
COOLDOWN_SECONDS = 5
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_WHISPER_URL = "https://api.openai.com/v1/audio/transcriptions"
ASSESSMENT_PATH = "/mnt/data/100X assessment.docx"  # uploaded file path (will be exposed as download)

# ---------- logging ----------
logger = logging.getLogger("100x-voicebot")
logger.setLevel(logging.INFO)

# ---------- page ----------
st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title("100x Voicebot — Streamlit demo")
st.write("This app accepts text or an audio file, sends text to OpenAI, and returns a voice reply (TTS).")

# ---------- API key ----------
OPENAI_KEY = None
try:
    OPENAI_KEY = st.secrets.get("OPENAI_API_KEY") if st.secrets else None
except Exception:
    OPENAI_KEY = None
if not OPENAI_KEY:
    OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_KEY:
    st.error("OpenAI API key not found. Add it to .streamlit/secrets.toml or set OPENAI_API_KEY as an environment variable.")
    st.stop()

# ---------- session state ----------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {"role":"user"/"assistant", "content": "..."}
if "awaiting_reply" not in st.session_state:
    st.session_state.awaiting_reply = False
if "cooldown_until" not in st.session_state:
    st.session_state.cooldown_until = 0

# helper: check disabled due to cooldown or awaiting reply
def is_disabled():
    return time.time() < st.session_state.cooldown_until or st.session_state.awaiting_reply

# ---------- UI columns ----------
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Send a message (text) or upload an audio file")
    # ---------- Text input form ----------
    with st.form("text_form", clear_on_submit=True):
        disabled = is_disabled()
        user_input = st.text_input("Type your message or question", key="text_input", disabled=disabled)
        submit_text = st.form_submit_button("Send", disabled=disabled)
    if submit_text and user_input:
        st.session_state.history.append({"role": "user", "content": user_input})
        st.session_state.awaiting_reply = True
        st.session_state.cooldown_until = time.time() + COOLDOWN_SECONDS

    # ---------- Audio upload (Whisper) ----------
    st.markdown("---")
    st.write("Or upload a recorded audio file (mp3 / wav / m4a / webm / ogg). We'll transcribe and reply with voice.")
    audio_file = st.file_uploader("Upload audio (optional)", type=["mp3", "wav", "m4a", "webm", "ogg"], disabled=is_disabled())
    if audio_file is not None and not is_disabled():
        # preview
        st.audio(audio_file)
        # enqueue the transcribed text as a user message and mark awaiting_reply
        try:
            st.session_state.awaiting_reply = True
            st.session_state.cooldown_until = time.time() + COOLDOWN_SECONDS
            # We'll transcribe & add to history in the main processing block below
            # store uploaded bytes temporarily in session so processing block can access it
            st.session_state._uploaded_audio = audio_file.getvalue()
        except Exception as e:
            st.error("Failed to queue audio for transcription.")
            logger.exception("Failed to queue audio")

    # ---------- Chat history display ----------
    st.markdown("### Conversation")
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Bot:** {msg['content']}")

with col_right:
    st.subheader("Assessment brief")
    # Provide a download button for the uploaded assessment file
    if os.path.exists(ASSESSMENT_PATH):
        with open(ASSESSMENT_PATH, "rb") as f:
            st.download_button("Download assessment brief", f, file_name=os.path.basename(ASSESSMENT_PATH))
    else:
        st.info("Assessment file not found at expected path.")

# ---------- resilient OpenAI caller with backoff ----------
def call_openai_with_backoff(messages, api_key, model="gpt-4o-mini", max_retries=5, initial_backoff=1.0):
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
            resp = requests.post(OPENAI_CHAT_URL, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as http_err:
            status = getattr(http_err.response, "status_code", None)
            logger.warning(f"OpenAI HTTPError status={status} attempt={attempt}: {http_err}")
            if status == 429 and attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise
        except requests.RequestException as req_err:
            logger.warning(f"RequestException attempt={attempt}: {req_err}")
            if attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise

# ---------- helper: transcribe audio with Whisper ----------
def transcribe_audio_bytes(audio_bytes, api_key, filename_hint="audio"):
    files = {
        "file": (filename_hint, audio_bytes),
        "model": (None, "whisper-1")
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.post(OPENAI_WHISPER_URL, headers=headers, files=files, timeout=60)
    resp.raise_for_status()
    return resp.json().get("text", "")

# ---------- helper: make TTS and return temporary filename ----------
def synthesize_tts_to_file(text, lang="en"):
    if not text:
        return None
    tts = gTTS(text, lang=lang)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

# ---------- main processing: handle awaiting reply (text or uploaded audio) ----------
if st.session_state.awaiting_reply and st.session_state.history:
    # Prevent re-entry while processing
    st.session_state.awaiting_reply = True
    last = st.session_state.history[-1]
    # Determine if we have uploaded audio queued
    uploaded_audio = st.session_state.pop("_uploaded_audio", None) if "_uploaded_audio" in st.session_state else None

    with st.spinner("Generating reply..."):
        try:
            # If uploaded audio present, transcribe first and append as user message
            if uploaded_audio:
                try:
                    transcribed = transcribe_audio_bytes(uploaded_audio, OPENAI_KEY, filename_hint="upload.wav")
                    if transcribed:
                        # append as user message (replace last if last was a placeholder)
                        st.session_state.history.append({"role": "user", "content": transcribed})
                        last = st.session_state.history[-1]
                        st.success("Transcription complete.")
                    else:
                        st.warning("Transcription returned empty text.")
                except requests.HTTPError as httpe:
                    status = getattr(httpe.response, "status_code", None)
                    logger.exception("Whisper HTTP error")
                    st.error(f"Transcription failed (HTTP {status}).")
                    st.session_state.history.append({"role": "assistant", "content": "Sorry — transcription failed."})
                    st.session_state.awaiting_reply = False
                except Exception:
                    logger.exception("Whisper unexpected error")
                    st.error("Transcription failed unexpectedly.")
                    st.session_state.history.append({"role": "assistant", "content": "Sorry — transcription failed."})
                    st.session_state.awaiting_reply = False

            # If last is user, call OpenAI Chat
            if st.session_state.history and st.session_state.history[-1]["role"] == "user":
                messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.history]
                # call chat with backoff
                data = call_openai_with_backoff(messages, OPENAI_KEY)
                assistant_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if not assistant_text:
                    assistant_text = "Sorry — I couldn't generate a response."
                st.session_state.history.append({"role": "assistant", "content": assistant_text})

                # create TTS and play it
                try:
                    tts_file = synthesize_tts_to_file(assistant_text, lang="en")
                    if tts_file:
                        st.audio(tts_file)
                except Exception:
                    logger.exception("TTS generation failed")
                    # still show the assistant text even if TTS fails
            else:
                # nothing to do
                pass

        except requests.HTTPError as http_err:
            status = getattr(http_err.response, "status_code", None)
            if status == 429:
                st.error("Too many requests to the OpenAI API right now. Please wait a few seconds and try again.")
            else:
                st.error(f"API request failed (HTTP {status}). Please try again later.")
            logger.exception("OpenAI HTTP error")
            st.session_state.history.append({"role": "assistant", "content": "Sorry — API error. Please try again later."})

        except Exception:
            logger.exception("Unexpected error calling OpenAI")
            st.error("An unexpected error occurred while contacting the OpenAI API. Check logs for details.")
            st.session_state.history.append({"role": "assistant", "content": "Sorry — unexpected server error."})

        finally:
            # clear awaiting flag and set cooldown to prevent immediate resubmits
            st.session_state.awaiting_reply = False
            st.session_state.cooldown_until = time.time() + COOLDOWN_SECONDS
