#!/usr/bin/env python3
# Copyright 2026 The Marusya-Gemini Authors.
# SPDX-License-Identifier: Apache-2.0
"""Wake word spotter for marusya-server.

marusya-server opens one WebSocket per speaker, sends the keywords as JSON
({"keywords": [...], "sample_rate": 16000}) and then the speaker's idle
microphone audio (16-bit mono PCM, binary frames). On a detection the
spotter answers {"keyword": "...", "engine": "...", "score": ...}.

Keywords are routed by script:

* Cyrillic -> Vosk (small Russian model). A keyword-only grammar spots
  candidates quickly (~0.3 s), then the full vocabulary re-decodes the last
  seconds of audio to confirm them. The grammar alone is not enough: it
  forces any speech into the keyword, so «сегодня хорошая погода» was heard
  as «маруся» with confidence 1.0. The words must be in the model's
  vocabulary; unknown ones are reported back as unsupported.
* Hebrew and everything else -> faster-whisper on short speech segments,
  fuzzy matched against the keywords. Slower (the utterance has to end), but
  any word in any of Whisper's languages works.

The models are loaded on the first connection. When the last speaker
disconnects (marusya-server has no speakers), the spotter falls asleep after
WAKE_SLEEP_AFTER seconds: it unloads the models to free memory and loads them
again when a speaker comes back. 0 keeps them loaded.

A third engine is optional: OWW_MODELS maps a keyword to an openWakeWord model
("джарвис=hey_jarvis"), and that phrase is then spotted by a network trained for
it instead of by Vosk. Trained models are far better at their own phrase, but
exist only for the phrases they were trained on.

Tuning (environment): VOSK_MIN_CONF, VOSK_HOLD, WHISPER_MODEL,
WHISPER_COMPUTE_TYPE, WHISPER_DEVICE, WHISPER_THREADS, WHISPER_MATCH, WAKE_COOLDOWN,
WAKE_SLEEP_AFTER, OWW_MODELS, OWW_THRESHOLD.
"""

import asyncio
import collections
import concurrent.futures
import difflib
import gc
import importlib.util
import json
import logging
import os
import pathlib
import re
import threading
import time
import unicodedata

import numpy as np
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

log = logging.getLogger("wake")

VOSK_MODEL = os.environ.get("VOSK_MODEL", "/models/vosk-model-small-ru-0.22")
VOSK_MIN_CONF = float(os.environ.get("VOSK_MIN_CONF", "0.6"))
# How close a free decode of the same audio must be to a keyword before it is
# worth a line: tells a rushed «маруса» apart from a cough.
VOSK_NEAR_RATIO = float(os.environ.get("VOSK_NEAR_RATIO", "0.6"))
# A second wake-word model that only scores what was heard and writes it to the
# journal: behaviour stays exactly as it is, the numbers decide later.
SHADOW_MODEL = os.environ.get("SHADOW_MODEL", str(pathlib.Path.home() / "models/oww/shadow.onnx"))
# When Vosk free-decodes a candidate and does not confirm it, the shadow model
# gets the last word: measured on live speech, it kept saying 1.00 on genuine
# «Маруся» that the free decode heard as «дуся». Zero disables the rescue.
SHADOW_ACCEPT = float(os.environ.get("SHADOW_ACCEPT", "0") or 0)
SHADOW_WORD = os.environ.get("SHADOW_WORD", "маруся")
# A keyword must stay in Vosk's partial result this long before it counts.
VOSK_HOLD = float(os.environ.get("VOSK_HOLD", "0.25"))
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_THREADS = int(os.environ.get("WHISPER_THREADS", "0")) or os.cpu_count() or 4
WHISPER_MATCH = float(os.environ.get("WHISPER_MATCH", "0.8"))
COOLDOWN = float(os.environ.get("WAKE_COOLDOWN", "1.5"))
# Quiet time after a wake. Vosk re-proposes the same word about every 0.7 s
# after its Reset, so anything above that is enough to stop a second wake, and
# everything below it is a window in which the owner can repeat the word.
WAKE_MUTE = float(os.environ.get("WAKE_MUTE", "1.3"))
SLEEP_AFTER = float(os.environ.get("WAKE_SLEEP_AFTER", "60"))
# Keywords spotted by openWakeWord instead of Vosk: "keyword=model" pairs, e.g.
# "джарвис=hey_jarvis". The models are neural detectors of one phrase each and
# are far more reliable than a general ASR model, but they only exist for the
# phrases they were trained on; the wheel ships hey_jarvis, alexa, hey_mycroft,
# hey_marvin, weather and timer.
OWW_MODELS = os.environ.get("OWW_MODELS", "")
OWW_THRESHOLD = float(os.environ.get("OWW_THRESHOLD", "0.5"))

# One Whisper inference at a time for all speakers: it is CPU heavy.
whisper_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
# Candidate verification is short but must not block the event loop.
vosk_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def script_of(word: str) -> str:
    if any("֐" <= c <= "׿" for c in word):
        return "he"
    if any("Ѐ" <= c <= "ӿ" for c in word):
        return "cyrillic"
    return "other"


# ---------- models (loaded once, lazily) ----------

_vosk_model = None
_whisper_model = None
_whisper_error = None
_generation = 0  # bumped by unload_models, so a late Whisper load is dropped
_lock = threading.Lock()


def vosk_model():
    global _vosk_model
    with _lock:
        if _vosk_model is None:
            from vosk import Model, SetLogLevel

            SetLogLevel(-1)
            log.info("loading Vosk model %s", VOSK_MODEL)
            _vosk_model = Model(VOSK_MODEL)
        return _vosk_model


def whisper_model():
    """Returns the Whisper model, or None while it is being loaded."""
    global _whisper_model, _whisper_error
    with _lock:
        if _whisper_model is not None or _whisper_error is not None:
            return _whisper_model
        if getattr(whisper_model, "loading", False):
            return None
        whisper_model.loading = True
        generation = _generation

    def load():
        global _whisper_model, _whisper_error
        try:
            from faster_whisper import WhisperModel

            log.info("loading Whisper model %s (%s, %s); the first start downloads it",
                     WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE)
            m = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE,
                             cpu_threads=WHISPER_THREADS)
            with _lock:
                if generation == _generation:
                    _whisper_model = m
            log.info("Whisper model ready")
        except Exception as e:  # noqa: BLE001 - reported to the server
            log.exception("loading Whisper failed")
            with _lock:
                if generation == _generation:
                    _whisper_error = e
        finally:
            with _lock:
                if generation == _generation:
                    whisper_model.loading = False

    threading.Thread(target=load, daemon=True).start()
    return None


def whisper_installed() -> bool:
    """Reports whether faster-whisper is installed at all."""
    return importlib.util.find_spec("faster_whisper") is not None


def unload_models():
    """Drops the models; the next connection loads them again."""
    global _vosk_model, _whisper_model, _whisper_error, _generation
    with _lock:
        loaded = _vosk_model is not None or _whisper_model is not None
        _vosk_model = _whisper_model = _whisper_error = None
        _generation += 1
        whisper_model.loading = False
    gc.collect()
    return loaded


class Sleeper:
    """Unloads the models a while after the last connection closes."""

    def __init__(self, delay):
        self.delay = delay
        self.clients = 0
        self.timer = None

    def connected(self):
        self.clients += 1
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None

    def disconnected(self):
        self.clients -= 1
        if self.clients == 0 and self.delay > 0:
            self.timer = asyncio.get_running_loop().call_later(self.delay, self.sleep)

    def sleep(self):
        self.timer = None
        if self.clients == 0 and unload_models():
            log.info("no speakers: models unloaded, sleeping until one connects")


sleeper = Sleeper(SLEEP_AFTER)


# ---------- engines ----------


class VoskSpotter:
    """Spots keywords with a small grammar, then verifies them.

    The grammar is fast but forces any speech into the keywords, so a
    candidate is re-decoded with the full Vosk vocabulary: unrelated phrases
    come out as themselves and are dropped. Verification runs only on
    candidates, in a worker thread, so idle audio stays cheap and the event
    loop keeps accepting connections.
    """

    engine = "vosk"
    BUFFER = 2.0  # seconds of audio kept for verification

    def __init__(self, keywords, rate):
        from vosk import KaldiRecognizer

        self.keywords = keywords
        self.norm = [normalize(k) for k in keywords]
        self.rate = rate
        rec = KaldiRecognizer(vosk_model(), rate, json.dumps(keywords + ["[unk]"], ensure_ascii=False))
        rec.SetWords(True)
        self.rec = rec
        self.since = None
        self.buf = collections.deque()
        self.buf_bytes = 0
        self.max_bytes = int(rate * self.BUFFER) * 2
        self.buf_lock = threading.Lock()
        self.verify_lock = threading.Lock()
        self.verifying = False
        self.mute_until = 0.0  # audio time; suppresses repeats of one phrase
        self.near_at = 0.0  # audio time of the last near-miss line
        self.near_text = ""

    @staticmethod
    def supports(word: str):
        """Returns the spelling of word known to the model, or None."""
        m = vosk_model()
        for w in (word, word.replace("ё", "е")):
            if all(m.vosk_model_find_word(part) >= 0 for part in w.split()):
                return w
        return None

    def _hit(self, text):
        padded = f" {text} "
        return next((k for k in self.keywords if f" {k} " in padded), None)

    def _shadow(self, text, kw, delay=0.0, emit=None, at=None):
        """Hands the utterance to the shadow model, off the audio path.

        delay lets the rest of the word reach the buffer: Vosk proposes a
        candidate from the first syllables, and scoring right then cuts the
        word in half (a genuine «Маруся» scored 0.00 that way)."""
        # Vosk reports an endpoint every few seconds in a silent room; scoring
        # those would put twelve meaningless lines a minute into the journal.
        if not kw and not text.strip():
            return
        sh = shadow()
        if sh is None:
            return
        _shadow_pool.submit(self._shadow_score, sh, self, text, kw, delay, emit, at)

    @staticmethod
    def _shadow_score(sh, spotter, text, kw, delay, emit=None, at=None):
        try:
            if delay:
                time.sleep(delay)
            with spotter.buf_lock:
                pcm = b"".join(spotter.buf)
            score = sh.score(pcm)
            if score is None:
                return
            log.info("shadow %s: %.2f | vosk %s | heard %r", sh.name, score,
                     f"{kw!r}" if kw else "no keyword", text[:60])
            # A sure model is the verdict: the decoders are slow and only add
            # a second of delay to a word it already knows.
            if (emit is not None and at is not None and kw == SHADOW_WORD
                    and SHADOW_ACCEPT and score >= SHADOW_ACCEPT):
                with spotter.verify_lock:
                    fire = at >= spotter.mute_until
                    if fire:
                        # The grammar re-proposes the same word right after a
                        # Reset; one wake per phrase is enough.
                        spotter.mute_until = at + WAKE_MUTE
                if fire:
                    log.info("shadow %s: %.2f, taking the word without waiting for Vosk",
                             sh.name, score)
                    emit(kw, "shadow", round(score, 2), at)
        except Exception:  # noqa: BLE001 - a diagnostic must never break the stream
            log.exception("shadow score")

    def _near_check(self, at):
        """Queues a free decode of the utterance that had no keyword in it.

        The grammar pass turns everything unknown into [unk], so it cannot say
        what was heard; a free decode of the same three seconds can."""
        if at < self.mute_until or at - self.near_at < 2.0:
            return
        self.near_at = at
        with self.buf_lock:
            pcm = b"".join(self.buf)
        vosk_pool.submit(self._near_decode, pcm[-int(self.rate * 3.0) * 2:], at)

    def _near_decode(self, pcm, at):
        try:
            from vosk import KaldiRecognizer

            rec = KaldiRecognizer(vosk_model(), self.rate)
            rec.AcceptWaveform(pcm)
            text = json.loads(rec.FinalResult()).get("text", "")
            tokens = [t for w in text.split() for t in normalize(w)]
            if not tokens:
                return
            best, word = 0.0, None
            for kw, toks in zip(self.keywords, self.norm):
                target = "".join(toks)
                for n in range(max(1, len(toks) - 1), len(toks) + 2):
                    for i in range(len(tokens) - n + 1):
                        r = difflib.SequenceMatcher(None, target, "".join(tokens[i:i + n])).ratio()
                        if r > best:
                            best, word = r, kw
            if best >= VOSK_NEAR_RATIO:
                log.info("near miss: heard %r, close to %r (%.2f)", text, word, round(best, 2))
        except Exception:  # noqa: BLE001 - never kill the stream
            log.exception("near miss decode")

    def _remember(self, pcm):
        with self.buf_lock:
            self.buf.append(pcm)
            self.buf_bytes += len(pcm)
            while self.buf_bytes > self.max_bytes and self.buf:
                self.buf_bytes -= len(self.buf.popleft())

    def _verify(self, kw, at, emit):
        """The slow opinion: a free decode of the same audio.

        Only runs when the wake-word model was not sure. When it is sure, the
        shadow worker has already fired and set the cooldown, so this candidate
        is suppressed."""
        try:
            from vosk import KaldiRecognizer

            with self.buf_lock:
                pcm = b"".join(self.buf)
            rec = KaldiRecognizer(vosk_model(), self.rate)
            rec.SetWords(True)
            rec.AcceptWaveform(pcm)
            res = json.loads(rec.FinalResult()).get("result", [])
            tokens, confs = [], []
            for w in res:
                for t in normalize(w["word"]):
                    tokens.append(t)
                    confs.append(w.get("conf", 0.0))
            toks = self.norm[self.keywords.index(kw)]
            n = len(toks)
            best = 0.0
            near = 0.0
            for size in (n - 1, n, n + 1):
                if size < 1:
                    continue
                for i in range(len(tokens) - size + 1):
                    window = tokens[i:i + size]
                    if window == toks:
                        best = max(best, min(confs[i:i + size]))
                    near = max(near, difflib.SequenceMatcher(None, "".join(toks), "".join(window)).ratio())
            log.debug("verified %r as %r (%.2f)", kw, " ".join(tokens), best)
            if best >= VOSK_MIN_CONF:
                # The grammar keeps forcing the rest of the phrase into the
                # keyword after Reset, so ignore its candidates for a while.
                # The model may also have taken this phrase already: one wake.
                with self.verify_lock:
                    if at < self.mute_until:
                        return
                    self.mute_until = at + WAKE_MUTE
                emit(kw, self.engine, round(best, 2), at)
            elif tokens:
                # The grammar pass proposed the keyword, the free decode of the
                # same audio did not confirm it: the word was heard badly. This
                # is what a real miss looks like from the inside.
                log.info("near miss: %r not confirmed, heard %r (closeness %.2f, conf %s)",
                         kw, " ".join(tokens)[:60], round(near, 2),
                         "/".join(f"{c:.2f}" for c in confs))
        except Exception:  # noqa: BLE001 - never kill the stream
            log.exception("vosk verify")
        finally:
            with self.verify_lock:
                self.verifying = False

    def _submit(self, kw, at, emit, text=""):
        with self.verify_lock:
            busy = self.verifying
            muted = at < self.mute_until
            if not busy and not muted:
                self.verifying = True
        # Every candidate that is not suppressed by the post-wake cooldown gets
        # scored, also the ones that wait for a verification in flight: the
        # point is the statistics, not the pipeline.
        if muted:
            return
        # The model decides on its own, in its own thread pool: no waiting on
        # Kaldi, and a candidate that arrives while a decode runs still counts.
        # 0.35 s is enough for the tail of «маруся» to land in the buffer.
        self._shadow(text, kw, delay=0.35, emit=emit, at=at)
        if busy:
            return
        vosk_pool.submit(self._verify, kw, at, emit)

    def feed(self, pcm, now, emit):
        self._remember(pcm)
        if self.rec.AcceptWaveform(pcm):
            self.since = None
            res = json.loads(self.rec.Result()).get("result", [])
            words = [w for w in res if w.get("conf", 0) >= VOSK_MIN_CONF]
            text = " ".join(w["word"] for w in words)
            kw = self._hit(text)
            if kw:
                self._submit(kw, now, emit, text)
            else:
                self._near_check(now)
                self._shadow(text, None)  # an utterance with no keyword in it
            return
        kw = self._hit(json.loads(self.rec.PartialResult()).get("partial", ""))
        if not kw:
            self.since = None
        elif self.since is None:
            self.since = now
        elif now - self.since >= VOSK_HOLD:
            self.rec.Reset()
            self.since = None
            self._submit(kw, now, emit)


class Shadow:
    """Scores every utterance with a second model, logging only.

    The point is to collect real numbers — hits on the owner's speech and false
    accepts on everything else — before deciding whether to switch detectors.
    """

    def __init__(self, path):
        from openwakeword.model import Model as OWWModel

        try:
            self.model = OWWModel(wakeword_models=[str(path)], inference_framework="onnx")
        except TypeError:
            # openWakeWord 0.4 (the version that installs on a Pi) has another
            # argument name and is ONNX-only.
            self.model = OWWModel(wakeword_model_paths=[str(path)])
        self.name = pathlib.Path(path).name
        self.lock = threading.Lock()

    def score(self, pcm):
        audio = np.frombuffer(pcm, dtype=np.int16)
        if len(audio) < 16000:  # the model needs about a second of context
            return None
        best = 0.0
        with self.lock:
            self.model.reset()
            for i in range(0, len(audio) - 1280, 1280):
                best = max(best, float(max(self.model.predict(audio[i:i + 1280]).values())))
        return best


_shadow = None
# Scoring two seconds of audio costs about half a second on the Pi, and the
# verification waits on it: with one worker a second candidate queued behind the
# first and woke up three seconds late.
_shadow_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="shadow")


def shadow():
    """Loads the shadow model once; None when the file is not there."""
    global _shadow
    if _shadow is None and pathlib.Path(SHADOW_MODEL).exists():
        try:
            _shadow = Shadow(SHADOW_MODEL)
            log.info("shadow model %s: scores utterances into the journal only", _shadow.name)
        except Exception:  # noqa: BLE001 - a diagnostic must never break the assistant
            log.exception("shadow model did not load; continuing without it")
            _shadow = False
    return _shadow or None


HEBREW_MARKS = re.compile("[֑-ׇ]")  # niqqud and cantillation
# Final forms, and letters that sound alike in modern Hebrew (Whisper often
# swaps them in a lone name).
FINAL_LETTERS = str.maketrans("ךםןףץעתק", "כמנפצאטכ")


def normalize(text: str) -> list[str]:
    text = HEBREW_MARKS.sub("", unicodedata.normalize("NFKC", text.lower()))
    text = text.translate(FINAL_LETTERS).replace("ё", "е")
    return re.findall(r"[^\W\d_]+(?:['׳״-][^\W\d_]+)*", text)


class WhisperSpotter:
    """Transcribes short speech segments and fuzzy matches the keywords."""

    engine = "whisper"
    FRAME = 0.02  # seconds per energy frame
    MAX_SEGMENT = 2.5
    MIN_SPEECH = 0.25
    END_SILENCE = 0.35

    def __init__(self, keywords, rate, language):
        self.keywords = keywords
        self.norm = [normalize(k) for k in keywords]
        self.rate = rate
        self.language = language
        self.floor = -60.0
        self.buf = []  # int16 arrays of the current segment
        self.speech = 0.0
        self.silence = 0.0
        self.skip = False  # the rest of a long utterance
        self.started = 0.0  # when the current segment began
        self.busy = False  # a segment is being transcribed
        self.queue = collections.deque(maxlen=2)
        self.lock = threading.Lock()
        self.pending = np.zeros(0, dtype=np.int16)

    def feed(self, pcm, now, emit):
        samples = np.concatenate([self.pending, np.frombuffer(pcm, dtype=np.int16)])
        n = int(self.rate * self.FRAME)
        whole = len(samples) // n * n
        self.pending = samples[whole:]
        for frame in samples[:whole].reshape(-1, n):
            self._frame(frame, now, emit)

    def _frame(self, frame, now, emit):
        rms = np.sqrt(np.mean(frame.astype(np.float32) ** 2)) + 1e-9
        db = 20 * np.log10(rms / 32768)
        # Noise floor: follows quiet frames at once, rises slowly.
        self.floor = db if db < self.floor else self.floor + 0.02
        voiced = db > max(self.floor + 10, -55)

        if self.skip:
            if voiced:
                self.silence = 0.0
            else:
                self.silence += self.FRAME
                if self.silence >= self.END_SILENCE:
                    self.skip = False
            return
        if voiced:
            self.speech += self.FRAME
            self.silence = 0.0
        elif self.buf:
            self.silence += self.FRAME
        if voiced and not self.buf:
            self.started = now
        if voiced or self.buf:
            self.buf.append(frame)
        total = len(self.buf) * self.FRAME
        if self.buf and (self.silence >= self.END_SILENCE or total >= self.MAX_SEGMENT):
            if total >= self.MAX_SEGMENT and self.silence < self.END_SILENCE:
                self.skip = True
            if self.speech >= self.MIN_SPEECH:
                self._submit(np.concatenate(self.buf), self.started, emit)
            self.buf, self.speech, self.silence = [], 0.0, 0.0

    def _submit(self, audio, started, emit):
        if whisper_model() is None:
            return
        with self.lock:
            # Transcription can be slower than speech: keep the newest segments
            # rather than the oldest, so a keyword after a long phrase is heard.
            self.queue.append((audio, started))
            if self.busy:
                return
            self.busy = True
        self._next(emit)

    def _next(self, emit):
        with self.lock:
            if not self.queue:
                self.busy = False
                return
            audio, started = self.queue.popleft()
        fut = whisper_pool.submit(self._transcribe, whisper_model(), audio)

        def done(f):
            try:
                hit = f.result()
            except Exception:  # noqa: BLE001
                log.exception("whisper")
                hit = None
            if hit:
                with self.lock:
                    self.queue.clear()  # the rest predates the wake
                emit(*hit, started)
            self._next(emit)

        fut.add_done_callback(done)

    def _transcribe(self, model, audio):
        segments, _ = model.transcribe(
            audio.astype(np.float32) / 32768.0,
            language=self.language,
            beam_size=1,
            without_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=False,
            # No temperature fallback: retries on unclear audio take seconds.
            temperature=0.0,
            max_new_tokens=16,  # the keyword is at the start
        )
        text = ""
        for s in segments:
            if s.no_speech_prob > 0.6 or s.avg_logprob < -1.2:
                continue
            text += " " + s.text
        words = normalize(text)
        log.debug("whisper heard %r", text.strip())
        best = (0.0, None)
        for kw, toks in zip(self.keywords, self.norm):
            n = len(toks)
            for i in range(max(0, len(words) - n) + 1):
                ratio = difflib.SequenceMatcher(None, " ".join(words[i:i + n]), " ".join(toks)).ratio()
                best = max(best, (ratio, kw), key=lambda x: x[0])
        if best[1] and best[0] >= WHISPER_MATCH:
            return best[1], self.engine, round(best[0], 2)
        return None


def oww_keywords(keywords):
    """Returns {keyword: model path} for the keywords listed in OWW_MODELS.

    Only phrases the operator mapped to a model are taken over; everything else
    keeps its old engine, so a missing model never removes a keyword.
    """
    if not OWW_MODELS:
        return {}
    import openwakeword

    models_dir = pathlib.Path(openwakeword.__file__ or "").parent / "resources" / "models"
    out = {}
    # Pairs are separated by commas: a keyword may contain spaces
    # ("hey jarvis=hey_jarvis"), so splitting on spaces would cut it in half.
    for pair in OWW_MODELS.split(","):
        if "=" not in pair:
            continue
        kw, model = (p.strip().lower() for p in pair.split("=", 1))
        if kw not in keywords:
            continue
        for path in sorted(models_dir.glob(model + "*.onnx")):
            out[kw] = path
            break
        else:
            log.warning("no openWakeWord model %r for keyword %r", model, kw)
    return out


class OpenWakeWordSpotter:
    """Spots a phrase with a model trained for that phrase.

    openWakeWord scores 80 ms frames with a small ONNX network: far more
    reliable than a general ASR model that merely contains the word in its
    vocabulary, and cheap enough for a Pi 4. It needs 16 kHz audio, which is
    what the server sends.
    """

    engine = "oww"
    FRAME = 1280  # samples: 80 ms at 16 kHz, the chunk size the models expect
    HOLD = 1.5  # seconds to ignore after a detection

    def __init__(self, pairs, rate):
        if rate != 16000:
            raise ValueError(f"openWakeWord needs 16 kHz audio, got {rate}")
        from openwakeword.model import Model

        self.by_model = {p.stem: kw for kw, p in pairs.items()}
        self.model = Model(wakeword_model_paths=[str(p) for p in pairs.values()])
        self.pending = np.zeros(0, dtype=np.int16)
        self.mute_until = 0.0

    def feed(self, pcm, now, emit):
        self.pending = np.concatenate([self.pending, np.frombuffer(pcm, dtype=np.int16)])
        while len(self.pending) >= self.FRAME:
            frame = self.pending[:self.FRAME]
            self.pending = self.pending[self.FRAME:]
            try:
                scores = self.model.predict(frame)
            except Exception:  # noqa: BLE001 - a model hiccup must not kill the stream
                log.exception("openWakeWord predict")
                continue
            for name, score in scores.items():
                if score < OWW_THRESHOLD:
                    continue
                kw = self.by_model.get(name)
                if kw is None or now < self.mute_until:
                    continue
                self.mute_until = now + self.HOLD
                self.model.reset()
                emit(kw, self.engine, round(float(score), 2), now)


def build(keywords, rate):
    """Creates spotters for keywords; returns (spotters, engines, unsupported)."""
    spotters, engines, unsupported = [], {}, []
    try:
        oww = oww_keywords(keywords)
    except Exception:  # noqa: BLE001 - a broken install must not stop the others
        log.exception("openWakeWord is not usable; every keyword goes to Vosk")
        oww = {}
    if oww:
        try:
            spotters.append(OpenWakeWordSpotter(oww, rate))
            for kw in oww:
                engines[kw] = OpenWakeWordSpotter.engine
            keywords = [kw for kw in keywords if kw not in oww]
        except Exception:  # noqa: BLE001 - same: fall back, do not fail
            log.exception("openWakeWord spotter failed; falling back to Vosk")
            spotters, engines = [], {}

    groups = {}
    for kw in keywords:
        groups.setdefault(script_of(kw), []).append(kw)

    cyr = groups.pop("cyrillic", [])
    if cyr:
        known = []
        for kw in cyr:
            spelling = VoskSpotter.supports(kw)
            if spelling:
                known.append(spelling)
                engines[kw] = "vosk"
            else:
                groups.setdefault("ru-whisper", []).append(kw)  # out of vocabulary
        if known:
            spotters.append(VoskSpotter(known, rate))
    for group, kws in groups.items():
        language = {"he": "he", "ru-whisper": "ru"}.get(group)  # None: detect
        spotters.append(WhisperSpotter(kws, rate, language))
        for kw in kws:
            engines[kw] = "whisper"
    if any(isinstance(s, WhisperSpotter) for s in spotters):
        if not whisper_installed():
            # Reported instead of silently routing to an engine that will
            # never fire: faster-whisper is an optional extra.
            unsupported = [k for k, e in engines.items() if e == "whisper"]
            log.warning("faster-whisper is not installed; these keywords cannot fire: %s",
                        unsupported)
        else:
            whisper_model()  # start loading in the background
            if _whisper_error is not None:
                unsupported = [k for k, e in engines.items() if e == "whisper"]
    return spotters, engines, unsupported


def feed_all(spotters, pcm, now, emit):
    """Feeds one audio frame to every spotter.

    Runs on a worker thread: a Vosk decode of a busy box can take tens of
    milliseconds, and blocking the event loop stalls the keepalive ping/pong
    until the client drops the stream.
    """
    for spotter in spotters:
        spotter.feed(pcm, now, emit)


async def handler(ws):
    sleeper.connected()
    try:
        await serve_speaker(ws)
    except (ConnectionClosed, EOFError) as exc:
        # The column drops its stream on every reboot and Wi-Fi hiccup; a full
        # traceback a few dozen times a day buried the wake lines in the journal.
        log.info("%s: stream ended without a close frame (%s)",
                 ws.remote_address, type(exc).__name__)
    finally:
        sleeper.disconnected()


async def serve_speaker(ws):
    loop = asyncio.get_running_loop()
    peer = ws.remote_address
    spotters = []
    last = -COOLDOWN
    # One worker per connection: frames stay in order, and a slow verify in
    # vosk_pool cannot delay the audio path.
    feed_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="wake-feed")

    def emit(keyword, engine, score, at):
        """Reports a keyword heard in the audio received at time at."""

        def send():
            # Engines report the same word at different delays (Whisper can
            # lag by seconds, and also hears a Russian «Антон» as «אנטון»):
            # compare when the audio was heard, not when the result came.
            nonlocal last
            if at < last + COOLDOWN:
                return
            last = at
            log.info("%s: heard %r (%s, %s), %.2f s after the audio",
                     peer, keyword, engine, score, time.monotonic() - at)
            msg = {"keyword": keyword, "engine": engine}
            if score is not None:
                msg["score"] = score
            asyncio.ensure_future(ws.send(json.dumps(msg, ensure_ascii=False)))

        loop.call_soon_threadsafe(send)

    try:
        async for msg in ws:
            if isinstance(msg, str):
                try:
                    cfg = json.loads(msg)
                    keywords = [str(k).strip().lower() for k in cfg.get("keywords", []) if str(k).strip()]
                    rate = int(cfg.get("sample_rate", 16000))
                    spotters, engines, unsupported = await loop.run_in_executor(None, build, keywords, rate)
                except Exception as e:  # noqa: BLE001 - reported to the server
                    log.exception("configure")
                    await ws.send(json.dumps({"error": str(e)}))
                    return
                log.info("%s: keywords %s", peer, engines)
                await ws.send(json.dumps({"ready": True, "engines": engines, "unsupported": unsupported},
                                         ensure_ascii=False))
                continue
            now = time.monotonic()
            await loop.run_in_executor(feed_pool, feed_all, spotters, msg, now, emit)
    finally:
        feed_pool.shutdown(wait=False, cancel_futures=True)


async def main():
    host, _, port = os.environ.get("WAKE_LISTEN", "0.0.0.0:2700").rpartition(":")
    if not os.path.isdir(VOSK_MODEL):  # fail fast on a broken image
        raise SystemExit(f"no Vosk model at {VOSK_MODEL}")
    import vosk  # noqa: F401 - the model itself is loaded on the first connection
    async with serve(handler, host or "0.0.0.0", int(port), max_size=1 << 20,
                      ping_interval=20, ping_timeout=60):
        log.info("listening on %s:%s", host, port)
        await asyncio.get_running_loop().create_future()


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("WAKE_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main())
