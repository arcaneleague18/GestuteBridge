import os
os.environ["KERAS_BACKEND"] = "torch"

import streamlit as st
import numpy as np
import cv2
import pathlib
import time
import threading
from collections import Counter

from hand_tracking import HandTracker
from model import load_action_model
from actions import load_actions
from text_to_sign import expand_units, get_sequence_frames, draw_landmarks
from data_collection import get_next_sequence_index
from sarvam_utils import translate_text, speech_to_text, LANGUAGES

# Try importing webrtc at top level so errors are visible
WEBRTC_AVAILABLE = False
WEBRTC_ERROR = ""
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
    import av
    WEBRTC_AVAILABLE = True
except Exception as e:
    WEBRTC_ERROR = str(e)

ROOT_DIR = pathlib.Path(__file__).resolve().parent
DATA_PATH = str(ROOT_DIR / "MP_Data")

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(page_title="Gesture Bridge", page_icon="🤟", layout="wide")

# ── Load Resources ──────────────────────────────────────────
@st.cache_resource
def load_resources():
    acts = np.array(load_actions(data_path=DATA_PATH))
    try:
        if len(acts) == 0:
            return None, acts, "No signs found – record some in the Add Signs tab."
        mdl = load_action_model(actions_shape=acts.shape[0])
        output_classes = int(mdl.output_shape[-1])
        if output_classes != int(acts.shape[0]):
            return None, acts, (
                f"Model output classes ({output_classes}) do not match actions ({acts.shape[0]}). "
                "Retrain the model or sync actions.txt with model labels."
            )
        return mdl, acts, None
    except Exception as e:
        return None, acts, str(e)

model, actions, model_error = load_resources()

# ── Header ──────────────────────────────────────────────────
st.title("Gesture Bridge 🤟")

# ── Tabs ────────────────────────────────────────────────────
tab_s2t, tab_t2s, tab_add = st.tabs(["Sign → Text", "Text → Sign", "Add Signs"])

# ═════════════════════════════════════════════════════════════
#  TAB 1: Sign → Text
# ═════════════════════════════════════════════════════════════
with tab_s2t:
    st.subheader("Sign → Text")

    output_lang = st.selectbox("Output language", list(LANGUAGES.keys()), index=0, key="s2t_lang")

    if model_error:
        st.error(model_error)
    elif not WEBRTC_AVAILABLE:
        st.error(f"streamlit-webrtc failed to load: `{WEBRTC_ERROR}`")
        st.code("pip install streamlit-webrtc av", language="bash")
    else:

        class SignProcessor(VideoProcessorBase):
            def __init__(self):
                self.tracker = HandTracker(max_hands=2)
                self.sequence = []
                self.words = []
                self.predictions = []
                self.current_label = ""
                self.current_confidence = 0.0
                self.stable_label = ""
                self.threshold = 0.35
                self.stability_window = 6
                self.stability_hits = 4
                self.n = 0
                self.prev_kp = None
                self.motion = 0.0
                self.motion_history = []
                self.last_error = ""

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                image = self.tracker.find_hands(img)
                kp = self.tracker.find_position(img)
                kp_arr = np.array(kp, dtype=np.float32)
                
                if self.prev_kp is not None:
                    self.motion = float(np.mean(np.abs(kp_arr - self.prev_kp)))
                self.prev_kp = kp_arr
                self.motion_history.append(self.motion)
                self.motion_history = self.motion_history[-12:]
                
                self.sequence.append(kp)
                self.sequence = self.sequence[-30:]
                self.n += 1
                h, w = image.shape[:2]

                if len(self.sequence) == 30 and self.n % 2 == 0:
                    try:
                        inp = np.expand_dims(self.sequence, axis=0).astype(np.float32)
                        res = np.array(model(inp, training=False)[0])
                        if res.shape[0] != len(actions):
                            raise ValueError(
                                f"Prediction size {res.shape[0]} does not match actions {len(actions)}"
                            )
                        idx = int(np.argmax(res))
                        self.predictions.append(idx)
                        self.predictions = self.predictions[-self.stability_window:]
                        conf = float(res[idx])
                        self.current_label = str(actions[idx])
                        self.current_confidence = conf
                        self.last_error = ""

                        if len(self.predictions) >= self.stability_hits:
                            stable_idx, stable_count = Counter(self.predictions).most_common(1)[0]
                            stable_conf = float(res[stable_idx])
                            if stable_count >= self.stability_hits and stable_conf > self.threshold:
                                act = str(actions[stable_idx])
                                self.stable_label = act
                                if len(self.words) > 0:
                                    if act != self.words[-1]:
                                        self.words.append(act)
                                else:
                                    self.words.append(act)

                        if len(self.words) > 5:
                            self.words = self.words[-5:]

                        cv2.rectangle(image, (0, 0), (w, 40), (40, 40, 40), -1)
                        cv2.putText(image, " ".join(self.words), (10, 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        clr = (0, 255, 0) if conf > 0.7 else (0, 200, 255) if conf > 0.5 else (0, 0, 255)
                        motion_score = self.motion * 1000.0
                        cv2.putText(image, f"{self.current_label}: {conf:.0%} | motion {motion_score:.2f}", (10, h - 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 2)
                    except Exception as e:
                        self.last_error = str(e)

                return av.VideoFrame.from_ndarray(image, format="bgr24")

        col1, col2 = st.columns([2, 1])

        with col1:
            ctx = webrtc_streamer(
                key="sign-translate",
                video_processor_factory=SignProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

        with col2:
            st.markdown("#### Live Translation")
            if "word_translation_cache" not in st.session_state:
                st.session_state.word_translation_cache = {}
            if "s2t_camera_seen" not in st.session_state:
                st.session_state.s2t_camera_seen = False
            if "s2t_last_words" not in st.session_state:
                st.session_state.s2t_last_words = []
            if "s2t_last_label" not in st.session_state:
                st.session_state.s2t_last_label = ""
            if "s2t_last_confidence" not in st.session_state:
                st.session_state.s2t_last_confidence = 0.0
            if "s2t_last_stable" not in st.session_state:
                st.session_state.s2t_last_stable = ""
            if "s2t_last_motion" not in st.session_state:
                st.session_state.s2t_last_motion = 0.0
            if "s2t_last_error" not in st.session_state:
                st.session_state.s2t_last_error = ""
            if "s2t_translated_text" not in st.session_state:
                st.session_state.s2t_translated_text = ""
            if "s2t_translated_source" not in st.session_state:
                st.session_state.s2t_translated_source = ""

            @st.fragment(run_every=1)
            def live_translation_view():
                vp = ctx.video_processor
                if vp is not None:
                    st.session_state.s2t_camera_seen = True
                    st.session_state.s2t_last_words = list(vp.words)
                    st.session_state.s2t_last_label = str(vp.current_label)
                    st.session_state.s2t_last_confidence = float(vp.current_confidence)
                    st.session_state.s2t_last_stable = str(vp.stable_label)
                    st.session_state.s2t_last_motion = float(vp.motion)
                    st.session_state.s2t_last_error = str(vp.last_error)

                current_words = st.session_state.s2t_last_words
                camera_running = bool(ctx.state.playing or st.session_state.s2t_camera_seen)

                if current_words:
                    detected_words = " ".join(current_words)
                    st.write(f"**Detected (English):** {detected_words}")
                elif camera_running:
                    st.info("Camera is running. Hold a sign steady to capture it.")
                else:
                    st.info("Start the camera to see live translation.")

                if st.session_state.s2t_last_label:
                    st.caption(
                        f"Current prediction: {st.session_state.s2t_last_label} "
                        f"({st.session_state.s2t_last_confidence:.0%})"
                    )
                if st.session_state.s2t_last_stable:
                    st.caption(f"Last accepted word: {st.session_state.s2t_last_stable}")
                if camera_running:
                    st.caption(f"Motion level: {st.session_state.s2t_last_motion * 1000.0:.2f}")
                if camera_running and not current_words and not st.session_state.s2t_last_label:
                    st.caption("No stable signs detected yet. Move your hand fully into frame and hold the sign steady.")
                if st.session_state.s2t_last_error:
                    st.error(f"Recognition error: {st.session_state.s2t_last_error}")

            live_translation_view()

            current_words = st.session_state.s2t_last_words
            can_translate = bool(current_words)

            if st.button("Translate", type="primary", key="s2t_translate_button", disabled=not can_translate):
                if not current_words:
                    st.warning("No detected English words to translate yet.")
                else:
                    detected_words = " ".join(current_words)
                    st.session_state.s2t_translated_source = detected_words
                    target_code = LANGUAGES[output_lang]
                    if target_code == "en-IN":
                        st.session_state.s2t_translated_text = detected_words
                    else:
                        cache = st.session_state.word_translation_cache.setdefault(target_code, {})
                        translated_words = []
                        for word in current_words:
                            if word not in cache:
                                cache[word] = translate_text(word, "en-IN", target_code)
                            translated_words.append(cache[word])
                        st.session_state.s2t_translated_text = " ".join(translated_words)

            if st.session_state.s2t_translated_text:
                st.write(f"**{output_lang}:** {st.session_state.s2t_translated_text}")

# ═════════════════════════════════════════════════════════════
#  TAB 2: Text → Sign
# ═════════════════════════════════════════════════════════════
with tab_t2s:
    st.subheader("Text → Sign")

    input_lang = st.selectbox("Your language", list(LANGUAGES.keys()), index=0, key="t2s_lang")

    available_signs = sorted(load_actions(data_path=DATA_PATH))
    with st.expander(f"Available signs ({len(available_signs)})"):
        st.write(", ".join(available_signs) if available_signs else "None – add signs first.")

    text_input = st.text_input("Type your text", placeholder="Type in your language...", key="t2s_text")
    audio_val = st.audio_input("Or speak", key="t2s_audio")

    speed = st.select_slider("Playback speed", ["Slow", "Normal", "Fast"], value="Normal", key="t2s_speed")
    delay = {"Slow": 0.08, "Normal": 0.04, "Fast": 0.02}[speed]

    if st.button("Translate to Sign", type="primary", key="t2s_go"):
        english_text = ""

        if audio_val is not None:
            audio_bytes = audio_val.read()
            lang_code = LANGUAGES[input_lang]

            with st.spinner("Transcribing audio..."):
                if lang_code == "en-IN":
                    english_text = speech_to_text(audio_bytes, "en-IN")
                else:
                    transcript = speech_to_text(audio_bytes, lang_code)
                    st.write(f"**Heard ({input_lang}):** {transcript}")
                    with st.spinner("Translating to English..."):
                        english_text = translate_text(transcript, lang_code, "en-IN")

            st.write(f"**English:** {english_text}")

        elif text_input.strip():
            lang_code = LANGUAGES[input_lang]
            if lang_code != "en-IN":
                with st.spinner("Translating to English..."):
                    english_text = translate_text(text_input, lang_code, "en-IN")
                st.write(f"**Original ({input_lang}):** {text_input}")
                st.write(f"**English:** {english_text}")
            else:
                english_text = text_input
                st.write(f"**English:** {english_text}")
        else:
            st.warning("Enter text or record audio first.")

        if english_text and not english_text.startswith("["):
            units = expand_units(english_text)
            if not units:
                st.warning("No matching signs found for this text.")
            else:
                st.write(f"**Signs:** {' → '.join(u for u in units if u != ' ')}")
                progress = st.progress(0)
                canvas = st.empty()

                for i, unit in enumerate(units):
                    progress.progress((i + 1) / len(units))

                    if unit == " ":
                        blank = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(blank, "SPACE", (220, 250),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 100, 100), 3)
                        canvas.image(cv2.cvtColor(blank, cv2.COLOR_BGR2RGB),
                                     channels="RGB", use_container_width=True)
                        time.sleep(0.5)
                        continue

                    paths = get_sequence_frames(unit)
                    if not paths:
                        st.caption(f"No data for '{unit}' – skipping")
                        continue

                    for npy_path in paths:
                        lm = np.load(npy_path)
                        img = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(img, unit, (260, 280),
                                    cv2.FONT_HERSHEY_SIMPLEX, 3, (150, 150, 255), 4)
                        draw_landmarks(img, lm)
                        canvas.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                     channels="RGB", use_container_width=True)
                        time.sleep(delay)
                    time.sleep(0.1)

                progress.progress(1.0, text="Done")

# ═════════════════════════════════════════════════════════════
#  TAB 3: Add Signs
# ═════════════════════════════════════════════════════════════
with tab_add:
    st.subheader("Add Signs")

    existing = sorted(
        d for d in os.listdir(DATA_PATH)
        if os.path.isdir(os.path.join(DATA_PATH, d))
    ) if os.path.exists(DATA_PATH) else []

    with st.expander(f"Existing signs ({len(existing)})"):
        st.write(", ".join(existing) if existing else "None")

    if not WEBRTC_AVAILABLE:
        st.error(f"streamlit-webrtc failed to load: `{WEBRTC_ERROR}`")
        st.code("pip install streamlit-webrtc av", language="bash")
    else:

        class RecordProcessor(VideoProcessorBase):
            def __init__(self):
                self.tracker = HandTracker(max_hands=2)
                self._lock = threading.Lock()
                self._recording = False
                self._current_frames = []
                self._completed_seqs = []
                self._sign_name = ""
                self._target_seqs = 5
                self._seq_length = 30
                self._done = False

            def start_recording(self, name, seqs, length):
                with self._lock:
                    self._sign_name = name
                    self._target_seqs = seqs
                    self._seq_length = length
                    self._current_frames = []
                    self._completed_seqs = []
                    self._done = False
                    self._recording = True

            def stop_recording(self):
                with self._lock:
                    self._recording = False

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                image = self.tracker.find_hands(img)
                h, w = image.shape[:2]

                with self._lock:
                    if self._recording and not self._done:
                        kp = self.tracker.find_position(img)
                        self._current_frames.append(kp)
                        fn = len(self._current_frames)
                        sn = len(self._completed_seqs) + 1

                        cv2.circle(image, (w - 20, 20), 8, (0, 0, 255), -1)
                        cv2.putText(image,
                            f"REC {self._sign_name} | Seq {sn}/{self._target_seqs} | Frame {fn}/{self._seq_length}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        prog = fn / self._seq_length
                        cv2.rectangle(image, (10, h - 15), (int(10 + (w - 20) * prog), h - 5), (0, 200, 0), -1)

                        if fn >= self._seq_length:
                            self._completed_seqs.append(list(self._current_frames))
                            self._current_frames = []
                            if len(self._completed_seqs) >= self._target_seqs:
                                self._done = True
                                self._recording = False

                    elif self._done:
                        cv2.putText(image, "Recording complete!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

                return av.VideoFrame.from_ndarray(image, format="bgr24")

            @property
            def is_done(self):
                with self._lock:
                    return self._done

            @property
            def completed_sequences(self):
                with self._lock:
                    return list(self._completed_seqs)

            @property
            def progress_info(self):
                with self._lock:
                    return len(self._completed_seqs), len(self._current_frames), self._recording

        c1, c2 = st.columns([1, 1])
        with c1:
            sign_name = st.text_input("Sign name", placeholder="e.g. HELLO", key="rec_sign")
        with c2:
            num_seqs = st.number_input("Sequences", min_value=1, max_value=100, value=5, key="rec_seqs")

        ctx = webrtc_streamer(
            key="sign-recorder",
            video_processor_factory=RecordProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if ctx.video_processor is not None:
            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("🔴 Record", type="primary", use_container_width=True):
                    name = sign_name.strip().upper()
                    if not name:
                        st.warning("Enter a sign name.")
                    else:
                        ctx.video_processor.start_recording(name, num_seqs, 30)
                        st.toast(f"Recording {name}...", icon="🔴")
            with b2:
                if st.button("⏹ Stop", use_container_width=True):
                    ctx.video_processor.stop_recording()
            with b3:
                if st.button("💾 Save", use_container_width=True):
                    if ctx.video_processor.is_done:
                        name = sign_name.strip().upper()
                        seqs = ctx.video_processor.completed_sequences
                        sign_dir = os.path.join(DATA_PATH, name)
                        start = get_next_sequence_index(name)
                        for i, seq_frames in enumerate(seqs):
                            seq_dir = os.path.join(sign_dir, str(start + i))
                            os.makedirs(seq_dir, exist_ok=True)
                            for fi, kp in enumerate(seq_frames):
                                np.save(os.path.join(seq_dir, f"{fi}.npy"), np.array(kp))
                        st.success(f"Saved {len(seqs)} sequences for '{name}'")
                        st.info("Run `python train.py` to retrain the model.")
                    else:
                        st.warning("Recording not finished yet.")

            done, cur, is_rec = ctx.video_processor.progress_info
            if is_rec:
                st.progress(done / max(num_seqs, 1), text=f"Seq {done+1}/{num_seqs} – frame {cur}/30")
            elif ctx.video_processor.is_done:
                st.success(f"Done! {done} sequences ready. Click Save.")
        else:
            st.info("Click START on the webcam above to begin.")
