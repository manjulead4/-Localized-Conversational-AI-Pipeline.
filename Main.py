import streamlit as st
import io
import struct
from google import genai
import re
from typing import Optional

try:
    from google.genai.errors import APIError
except ImportError:
    try:
        from google.genai import APIError
    except ImportError:
        class APIError(Exception):
            pass


API_KEY = "AIzaSyD_aklgpblibEKUE-Rdge4Ugj4IQ_Z78Hs"
LLM_MODEL = 'gemini-2.5-flash-preview-05-20'
TTS_MODEL = 'gemini-2.5-flash-preview-tts'
TELUGU_VOICE = "Kore"

try:
    client = genai.Client(api_key=API_KEY)
except Exception:
    st.error("Error initializing Gemini client. Please check your API Key and ensure the `google-genai` library is installed.")
    st.stop()


def pcm_to_wav_bytes(pcm_data: bytes, sample_rate: int = 24000, num_channels: int = 1, bits_per_sample: int = 16) -> bytes:
    byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
    block_align = num_channels * (bits_per_sample // 8)

    wav_io = io.BytesIO()

    wav_io.write(b'RIFF')
    wav_io.write(struct.pack('<I', 36 + len(pcm_data)))
    wav_io.write(b'WAVE')

    wav_io.write(b'fmt ')
    wav_io.write(struct.pack('<I', 16))
    wav_io.write(struct.pack('<H', 1))
    wav_io.write(struct.pack('<H', num_channels))
    wav_io.write(struct.pack('<I', sample_rate))
    wav_io.write(struct.pack('<I', byte_rate))
    wav_io.write(struct.pack('<H', block_align))
    wav_io.write(struct.pack('<H', bits_per_sample))

    wav_io.write(b'data')
    wav_io.write(struct.pack('<I', len(pcm_data)))
    wav_io.write(pcm_data)

    return wav_io.getvalue()

@st.cache_data(show_spinner=False)
def get_llm_response(query: str) -> str:
    system_instruction = "Act as a helpful and friendly assistant. Respond concisely and entirely in natural, conversational Telugu (India) using the input provided."

    try:
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=query,
            config={
                "system_instruction": system_instruction
            }
        )
        return response.text
    except APIError as e:
        return f"LLM API Error: {e}"
    except Exception as e:
        return f"An unexpected error occurred during LLM call: {e}"

def get_tts_audio_data(text: str) -> tuple[Optional[bytes], int]:
    try:
        response = client.models.generate_content(
            model=TTS_MODEL,
            contents=[
                genai.types.Content(
                    parts=[
                        genai.types.Part.from_text(text=text)
                    ],
                )
            ],
            config={
                "response_modalities": ["AUDIO"],
                "speech_config": {
                    "voice_config": {
                        "prebuilt_voice_config": {
                            "voice_name": TELUGU_VOICE
                        }
                    }
                }
            }
        )

        audio_part = response.candidates[0].content.parts[0]
        pcm_data = audio_part.inline_data.data
        mime_type = audio_part.inline_data.mime_type

        rate_match = re.search(r'rate=(\d+)', mime_type)
        sample_rate = int(rate_match.group(1)) if rate_match else 24000

        return pcm_data, sample_rate

    except APIError as e:
        st.error(f"TTS API Error: {e}")
        return None, 0
    except Exception as e:
        st.error(f"An unexpected error occurred during TTS call: {e}")
        return None, 0


def main():
    st.set_page_config(page_title="Telugu Conversational AI", layout="centered")

    st.title("🗣️ తెలుగు సంభాషణ AI పైప్‌లైన్")
    st.markdown("ASR (Simulation) → LLM → TTS Pipeline using Gemini")
    st.markdown("---")

    user_input = st.text_area(
        "తెలుగులో మీ ప్రశ్నను ఇక్కడ టైప్ చేయండి (ASR అవుట్‌పుట్ అనుకరణ):",
        placeholder="ఉదాహరణకు: ఈరోజు మార్కెట్లో బంగారం ధర ఎంత? (What is the price of gold in the market today?)",
        height=100
    )

    if st.button("LLM ద్వారా ప్రాసెస్ చేయండి & ఆడియోను రూపొందించండి", use_container_width=True, type="primary"):
        if not user_input.strip():
            st.warning("దయచేసి మీ ప్రశ్నను తెలుగులో టైప్ చేయండి.")
            return

        with st.spinner("LLM ప్రాసెసింగ్ & TTS జనరేషన్ జరుగుతోంది..."):

            llm_response_telugu = get_llm_response(user_input)

            st.success("✅ LLM ప్రతిస్పందన సిద్ధంగా ఉంది:")
            st.markdown(f"**LLM జవాబు:** *{llm_response_telugu}*")

            if "Error" in llm_response_telugu or "లోపం" in llm_response_telugu:
                st.error("LLM ప్రాసెసింగ్‌లో లోపం జరిగింది.")
                return

            pcm_data, sample_rate = get_tts_audio_data(llm_response_telugu)

            if not pcm_data:
                return

            wav_bytes = pcm_to_wav_bytes(pcm_data, sample_rate=sample_rate)

            st.success("🔊 ఆడియో ప్లేబ్యాక్:")
            st.audio(wav_bytes, format='audio/wav', start_time=0)

if __name__ == '__main__':
    main()
