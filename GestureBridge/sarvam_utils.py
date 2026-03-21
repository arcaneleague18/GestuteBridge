"""Sarvam AI translation & speech-to-text helpers."""

import os
import requests
import tempfile
from dotenv import load_dotenv

load_dotenv()
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")

LANGUAGES = {
    "English": "en-IN",
    "Hindi": "hi-IN",
    "Bengali": "bn-IN",
    "Tamil": "ta-IN",
    "Telugu": "te-IN",
    "Kannada": "kn-IN",
    "Malayalam": "ml-IN",
    "Marathi": "mr-IN",
    "Gujarati": "gu-IN",
    "Punjabi": "pa-IN",
    "Odia": "od-IN",
    "Urdu": "ur-IN",
    "Assamese": "as-IN",
    "Nepali": "ne-IN",
}

HEADERS = {
    "Content-Type": "application/json",
}


def _get_headers_json():
    return {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json",
    }


def _get_headers_form():
    return {
        "api-subscription-key": SARVAM_API_KEY,
    }


def translate_text(text: str, source_lang_code: str, target_lang_code: str) -> str:
    """Translate text between languages using Sarvam AI.

    Returns translated text, or original text on failure.
    """
    if not SARVAM_API_KEY:
        return f"[API key missing] {text}"
    if source_lang_code == target_lang_code:
        return text
    if not text.strip():
        return ""

    try:
        resp = requests.post(
            "https://api.sarvam.ai/translate",
            json={
                "input": text,
                "source_language_code": source_lang_code,
                "target_language_code": target_lang_code,
            },
            headers=_get_headers_json(),
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("translated_text", text)
    except Exception as e:
        return f"[Translation error: {e}] {text}"


def speech_to_text(audio_bytes: bytes, language_code: str = "unknown") -> str:
    """Transcribe audio using Sarvam AI speech-to-text.

    Returns transcript string, or error message on failure.
    """
    if not SARVAM_API_KEY:
        return "[API key missing]"

    try:
        # Write audio bytes to a temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            resp = requests.post(
                "https://api.sarvam.ai/speech-to-text",
                files={"file": ("audio.wav", f, "audio/wav")},
                data={
                    "model": "saaras:v3",
                    "language_code": language_code,
                },
                headers=_get_headers_form(),
                timeout=30,
            )
        os.unlink(tmp_path)
        resp.raise_for_status()
        return resp.json().get("transcript", "")
    except Exception as e:
        return f"[STT error: {e}]"


def speech_to_english(audio_bytes: bytes, language_code: str = "unknown") -> str:
    """Transcribe audio and translate to English using Sarvam AI saaras:v3 translate mode."""
    if not SARVAM_API_KEY:
        return "[API key missing]"

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            resp = requests.post(
                "https://api.sarvam.ai/speech-to-text",
                files={"file": ("audio.wav", f, "audio/wav")},
                data={
                    "model": "saaras:v3",
                    "mode": "translate",
                    "language_code": language_code,
                },
                headers=_get_headers_form(),
                timeout=30,
            )
        os.unlink(tmp_path)
        resp.raise_for_status()
        return resp.json().get("transcript", "")
    except Exception as e:
        return f"[STT error: {e}]"
