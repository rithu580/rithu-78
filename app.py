import streamlit as st
import requests
import os
import base64

st.set_page_config(page_title="100x Voicebot — Streamlit", layout="centered")

# --- Load OpenAI Key ---
OPENAI_KEY = None
try:
    OPENAI_KEY = st.secrets.get("OPENAI_API_KEY") if st.secrets else None
except:
    OPENAI_KEY = None

if not OPENAI_KEY:
    OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_KEY:
    st.error("Missing API Key. Add OPENAI_API_KEY in Streamlit Secrets.")
    st.stop()

headers_json = {
    "Authorization": f"Bearer {OPENAI_KEY}",
    "Content-Type": "application/json"
}
headers_auth = {"Authorization": f"Bearer {OPENAI_KEY}"}

st.title("🎤 100x Voicebot — Text + Voice Mode")
st.write("Chat with text or upload audio. The bot replies with voice & text.")

# ------------------------------------------------------------------------------------
# TEXT CHAT SECTION
# ------------------------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

with st.form("chat_form", clear_on_submit=True):
    txt = st.text_input("Type your message")
    send = st.form_submit_button("Send")

if send and txt:
    st.session_state.history.append({"role": "user", "content": txt})

# Display chat history
for m in st.session_state.history:
    speaker = "You" if m["role"] == "user" else "Bot"
    st.markdown(f"**{speaker}:** {m['content']}")

# Process last user message
if st.session_state.history and st.session_state.history[-1]["role"] == "user":
    with st.spinner("Thinking..."):
        payload = {
            "model": "gpt-4o-mini",
            "messages": st.session_state.history,
            "max_tokens": 400
        }
        r = requests.post("https://api.openai.com/v1/chat/completions",
                          json=payload, headers=headers_json)
        reply = r.json()["choices"][0]["message"]["content"]

        st.session_state.history.append({"role": "assistant", "content": reply})

        # Generate voice using OpenAI Speech API
        tts_payload = {
            "model": "gpt-4o-mini-tts",
            "input": reply,
            "voice": "alloy"
        }

        audio_response = requests.post(
            "https://api.openai.com/v1/audio/speech",
            json=tts_payload,
            headers=headers_json
        )

        # Save audio to temporary file
        audio_bytes = audio_response.content
        audio_path = "bot_reply.mp3"
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        st.audio(audio_path)

# ------------------------------------------------------------------------------------
# VOICE UPLOAD SECTION
# ------------------------------------------------------------------------------------

st.subheader("🎧 Upload Audio for Voice Conversation")

audio_file = st.file_uploader("Upload voice message", type=["mp3", "wav", "m4a"])

if audio_file:
    st.audio(audio_file)

    with st.spinner("Transcribing speech..."):
        files = {
            "file": (audio_file.name, audio_file.getvalue(), "audio/mpeg"),
            "model": (None, "whisper-1")
        }

        trans = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers=headers_auth,
            files=files
        ).json()["text"]

        st.markdown(f"**You said:** {trans}")

        # Send transcription to chat model
        chat_payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": trans}]
        }

        r2 = requests.post("https://api.openai.com/v1/chat/completions",
                           json=chat_payload, headers=headers_json)

        bot_reply = r2.json()["choices"][0]["message"]["content"]
        st.markdown(f"**Bot:** {bot_reply}")

        # Convert bot reply to speech using OpenAI TTS
        tts_payload2 = {
            "model": "gpt-4o-mini-tts",
            "input": bot_reply,
            "voice": "alloy"
        }

        audio_response2 = requests.post(
            "https://api.openai.com/v1/audio/speech",
            json=tts_payload2,
            headers=headers_json
        )

        bot_audio_path = "bot_voice_reply.mp3"
        with open(bot_audio_path, "wb") as f:
            f.write(audio_response2.content)

        st.audio(bot_audio_path)
