"""
BrowSync — Microphone prosody processor.

Runs continuously in a background thread using sounddevice.
Produces a ProsodyFrame every ~100ms with normalised pitch, energy,
speech rate, and voice activity detection.

These map directly to the PROSODY_FEATURES in schema.py and are
injected into every input frame regardless of whether VRCFT is
sending face/eye data — this is what makes the mic-only fallback work.
"""

import threading
import time
import logging
import queue
from dataclasses import dataclass
from collections import deque
from typing import Optional

import numpy as np

log = logging.getLogger("browsync.mic")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SAMPLE_RATE     = 16000     # Hz — enough for pitch/energy, low CPU cost
BLOCK_SIZE      = 512       # samples per callback (~32ms at 16kHz)
ANALYSIS_WINDOW = 0.10      # seconds of audio per prosody update
PITCH_MIN_HZ    = 80.0      # below this = not voiced
PITCH_MAX_HZ    = 400.0     # above this = not voiced (for speech)

# Normalisation ranges
ENERGY_FLOOR_DB = -60.0
ENERGY_CEIL_DB  = -20.0
PITCH_FLOOR_HZ  = 80.0
PITCH_CEIL_HZ   = 300.0


@dataclass
class ProsodyFrame:
    pitch_norm:   float = 0.0   # 0-1, normalised fundamental frequency
    pitch_delta:  float = 0.0   # first derivative (rising=+, falling=-)
    energy_norm:  float = 0.0   # 0-1, normalised RMS energy
    energy_delta: float = 0.0   # first derivative
    speech_rate:  float = 0.0   # 0-1, estimated syllable rate
    is_speaking:  float = 0.0   # 0 or 1, voice activity

    def to_feature_dict(self) -> dict:
        return {
            "PitchNorm":   self.pitch_norm,
            "PitchDelta":  self.pitch_delta,
            "EnergyNorm":  self.energy_norm,
            "EnergyDelta": self.energy_delta,
            "SpeechRate":  self.speech_rate,
            "IsSpeaking":  self.is_speaking,
        }

    @staticmethod
    def zero() -> "ProsodyFrame":
        return ProsodyFrame()


class MicrophoneProcessor:
    """
    Captures microphone audio and extracts prosody features continuously.
    Thread-safe: call .latest from any thread.
    """

    def __init__(self, device: Optional[int] = None, sample_rate: int = SAMPLE_RATE):
        self._device      = device
        self._sample_rate = sample_rate
        self._latest      = ProsodyFrame.zero()
        self._lock        = threading.Lock()
        self._audio_q: queue.Queue = queue.Queue(maxsize=50)
        self._stop        = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._available   = False

        # Rolling history for delta computation
        self._pitch_history:  deque = deque(maxlen=10)
        self._energy_history: deque = deque(maxlen=10)

        # Speech rate estimation: track energy onset events
        self._onset_times: deque = deque(maxlen=8)
        self._prev_energy_norm = 0.0
        self._energy_onset_threshold = 0.3

    @property
    def latest(self) -> ProsodyFrame:
        with self._lock:
            return self._latest

    @property
    def is_available(self) -> bool:
        return self._available

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="BrowSync-Mic", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    # -- Background thread ---------------------------------------------------

    def _run(self):
        try:
            import sounddevice as sd
        except ImportError:
            log.warning("[Mic] sounddevice not installed. pip install sounddevice")
            return

        try:
            samples_per_update = int(self._sample_rate * ANALYSIS_WINDOW)
            buffer = np.zeros(samples_per_update, dtype=np.float32)
            buf_pos = 0

            def callback(indata, frames, time_info, status):
                mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
                self._audio_q.put(mono.copy())

            with sd.InputStream(
                device=self._device,
                channels=1,
                samplerate=self._sample_rate,
                blocksize=BLOCK_SIZE,
                dtype=np.float32,
                callback=callback,
            ):
                self._available = True
                log.info("[Mic] Microphone capture started.")

                while not self._stop.is_set():
                    try:
                        chunk = self._audio_q.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    # Fill rolling buffer
                    n = len(chunk)
                    space = samples_per_update - buf_pos
                    if n >= space:
                        buffer[buf_pos:] = chunk[:space]
                        frame = self._analyse(buffer.copy())
                        with self._lock:
                            self._latest = frame
                        # Carry remainder
                        remainder = chunk[space:]
                        buffer[:len(remainder)] = remainder
                        buf_pos = len(remainder)
                    else:
                        buffer[buf_pos:buf_pos + n] = chunk
                        buf_pos += n

        except Exception as e:
            log.warning(f"[Mic] Could not open microphone: {e}")
        finally:
            self._available = False

    def _analyse(self, audio: np.ndarray) -> ProsodyFrame:
        """Extract prosody features from one analysis window."""
        # -- RMS energy ------------------------------------------------------
        rms = float(np.sqrt(np.mean(audio ** 2)) + 1e-9)
        rms_db = 20 * np.log10(rms)
        energy_norm = float(np.clip(
            (rms_db - ENERGY_FLOOR_DB) / (ENERGY_CEIL_DB - ENERGY_FLOOR_DB),
            0.0, 1.0
        ))

        # -- Voice activity detection (simple energy threshold) --------------
        is_speaking = 1.0 if energy_norm > 0.15 else 0.0

        # -- Pitch estimation (autocorrelation) ------------------------------
        pitch_norm = 0.0
        if is_speaking:
            pitch_hz = self._estimate_pitch(audio)
            if pitch_hz is not None:
                pitch_norm = float(np.clip(
                    (pitch_hz - PITCH_FLOOR_HZ) / (PITCH_CEIL_HZ - PITCH_FLOOR_HZ),
                    0.0, 1.0
                ))

        # -- Deltas ----------------------------------------------------------
        self._pitch_history.append(pitch_norm)
        self._energy_history.append(energy_norm)

        pitch_delta  = self._delta(self._pitch_history)
        energy_delta = self._delta(self._energy_history)

        # -- Speech rate (onset counting) ------------------------------------
        now = time.monotonic()
        if (energy_norm > self._energy_onset_threshold and
                self._prev_energy_norm <= self._energy_onset_threshold):
            self._onset_times.append(now)
        self._prev_energy_norm = energy_norm

        # Count onsets in last 3 seconds, normalise to ~0-1 range (max ~6 syl/s)
        recent_onsets = sum(1 for t in self._onset_times if now - t < 3.0)
        speech_rate = float(np.clip(recent_onsets / 18.0, 0.0, 1.0))

        return ProsodyFrame(
            pitch_norm=pitch_norm,
            pitch_delta=float(np.clip(pitch_delta * 3.0, -1.0, 1.0)),
            energy_norm=energy_norm,
            energy_delta=float(np.clip(energy_delta * 3.0, -1.0, 1.0)),
            speech_rate=speech_rate,
            is_speaking=is_speaking,
        )

    def _estimate_pitch(self, audio: np.ndarray) -> Optional[float]:
        """
        Autocorrelation-based pitch detection.
        Returns fundamental frequency in Hz, or None if unvoiced.
        """
        n = len(audio)
        min_lag = int(self._sample_rate / PITCH_MAX_HZ)
        max_lag = int(self._sample_rate / PITCH_MIN_HZ)

        if max_lag >= n:
            return None

        # Normalised autocorrelation
        audio = audio - audio.mean()
        corr = np.correlate(audio, audio, mode='full')[n-1:]
        if corr[0] < 1e-9:
            return None
        corr = corr / corr[0]

        # Find the highest peak in the valid lag range
        search = corr[min_lag:max_lag]
        if len(search) == 0:
            return None

        peak_idx = int(np.argmax(search))
        peak_val = search[peak_idx]

        # Threshold: must be clearly periodic
        if peak_val < 0.45:
            return None

        lag = peak_idx + min_lag
        return float(self._sample_rate / lag)

    @staticmethod
    def _delta(history: deque) -> float:
        if len(history) < 3:
            return 0.0
        vals = list(history)
        return vals[-1] - vals[0]


# ---------------------------------------------------------------------------
# SpeechBrain emotion recognition — optional upgrade layer
# ---------------------------------------------------------------------------

class SpeechBrainSER:
    """
    Wraps SpeechBrain's emotion recognition classifier to produce
    valence/arousal estimates from short audio windows (~1-2s).

    Runs in the same background thread as MicrophoneProcessor.
    Output feeds directly into EmotionValence / EmotionArousal / EmotionConfidence
    features in the input vector.

    Model used: speechbrain/emotion-recognition-wav2vec2-IEMOCAP
    Maps 4-class output (neutral/happy/sad/angry) to continuous valence/arousal.
    """

    # IEMOCAP 4-class → (valence, arousal) mapping
    # Based on the Russell circumplex model of affect
    EMOTION_MAP = {
        "neu": (0.0,   0.1),   # neutral   — centred, low arousal
        "hap": (0.8,   0.7),   # happy     — positive, high arousal
        "sad": (-0.6,  -0.4),  # sad       — negative, low arousal
        "ang": (-0.5,  0.8),   # angry     — negative, high arousal
    }

    def __init__(self):
        self._classifier = None
        self._available  = False
        self._valence    = 0.0
        self._arousal    = 0.0
        self._confidence = 0.0
        self._lock       = threading.Lock()

    def load(self, sample_rate: int = 16000):
        """Load model — call once at startup. Can take a few seconds."""
        try:
            from speechbrain.inference.classifiers import EncoderClassifier
            self._classifier = EncoderClassifier.from_hparams(
                source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                savedir="models/speechbrain_ser",
                run_opts={"device": "cpu"},
            )
            self._sample_rate = sample_rate
            self._available   = True
            log.info("[Mic] SpeechBrain SER model loaded.")
        except (ImportError, RuntimeError) as e:
            error_msg = str(e).lower()
            if "k2" in error_msg or "lazymodule" in error_msg:
                log.warning(
                    "[Mic] SpeechBrain SER not available: k2_fsa dependency missing. "
                    "Install with: pip install k2. SER will be disabled."
                )
            else:
                log.warning(f"[Mic] SpeechBrain SER not available: {e}")
        except Exception as e:
            log.warning(f"[Mic] SpeechBrain SER not available: {e}")

    @property
    def is_available(self) -> bool:
        return self._available

    def infer(self, audio: np.ndarray, sample_rate: int) -> tuple[float, float, float]:
        """
        Run inference on an audio segment.
        Returns (valence, arousal, confidence).
        Non-blocking — returns last known values if model is busy.
        """
        if not self._available or self._classifier is None:
            return 0.0, 0.0, 0.0

        try:
            import torch
            wav = torch.tensor(audio).unsqueeze(0).float()
            out_prob, score, index, label = self._classifier.classify_batch(wav)

            emotion_key = label[0].strip().lower()[:3]   # "neu", "hap", "sad", "ang"
            valence, arousal = self.EMOTION_MAP.get(emotion_key, (0.0, 0.1))
            confidence = float(score[0])

            with self._lock:
                self._valence    = valence
                self._arousal    = arousal
                self._confidence = confidence

            return valence, arousal, confidence

        except Exception as e:
            log.debug(f"[Mic] SER inference error: {e}")
            with self._lock:
                return self._valence, self._arousal, self._confidence

    def latest(self) -> tuple[float, float, float]:
        with self._lock:
            return self._valence, self._arousal, self._confidence


class MicrophoneProcessorWithSER(MicrophoneProcessor):
    """
    MicrophoneProcessor extended with SpeechBrain emotion recognition.

    SER runs on a slower cadence (~500ms window, ~2x per second) in the
    same background thread, interleaved with the fast prosody updates.
    EmotionValence, EmotionArousal, EmotionConfidence are added to the
    ProsodyFrame's feature dict when SER is active.
    """

    SER_WINDOW_SECONDS = 1.5     # audio window fed to SER
    SER_INTERVAL       = 0.5     # seconds between SER inference calls

    def __init__(self, device=None, sample_rate: int = SAMPLE_RATE):
        super().__init__(device=device, sample_rate=sample_rate)
        self._ser             = SpeechBrainSER()
        self._ser_buffer      = np.zeros(
            int(sample_rate * self.SER_WINDOW_SECONDS), dtype=np.float32
        )
        self._ser_buf_pos     = 0
        self._last_ser_time   = 0.0
        self._ser_valence     = 0.0
        self._ser_arousal     = 0.0
        self._ser_confidence  = 0.0

    def start(self):
        # Load SER model before starting capture thread
        self._ser.load(self._sample_rate)
        super().start()

    def _analyse(self, audio: np.ndarray) -> "ProsodyFrame":
        frame = super()._analyse(audio)

        # Feed audio into SER rolling buffer
        n = len(audio)
        buf_len = len(self._ser_buffer)
        if n >= buf_len:
            self._ser_buffer[:] = audio[-buf_len:]
            self._ser_buf_pos   = buf_len
        else:
            space = buf_len - self._ser_buf_pos
            if n <= space:
                self._ser_buffer[self._ser_buf_pos:self._ser_buf_pos + n] = audio
                self._ser_buf_pos += n
            else:
                # Wrap around
                self._ser_buffer[self._ser_buf_pos:] = audio[:space]
                rem = audio[space:]
                self._ser_buffer[:len(rem)] = rem
                self._ser_buf_pos = len(rem)

        # Run SER on interval
        now = time.monotonic()
        if (self._ser.is_available
                and frame.is_speaking > 0
                and now - self._last_ser_time >= self.SER_INTERVAL):
            self._last_ser_time = now
            v, a, c = self._ser.infer(self._ser_buffer.copy(), self._sample_rate)
            self._ser_valence    = v
            self._ser_arousal    = a
            self._ser_confidence = c

        # Decay emotion confidence toward zero when not speaking
        if frame.is_speaking < 0.5:
            self._ser_confidence *= 0.95   # slow decay

        # Attach emotion features to the prosody frame's dict via monkeypatching
        # the to_feature_dict output — cleaner than subclassing ProsodyFrame
        base_dict = frame.to_feature_dict()
        base_dict["EmotionValence"]    = float(self._ser_valence)
        base_dict["EmotionArousal"]    = float(self._ser_arousal)
        base_dict["EmotionConfidence"] = float(self._ser_confidence)

        # Wrap in extended frame
        frame._extra = base_dict
        frame.to_feature_dict = lambda: base_dict   # type: ignore

        return frame
