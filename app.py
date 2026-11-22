"""AccentLive Pro: type a line or press a button, a clear American voice goes into a virtual microphone.
Buttons are voiced once at start-up and kept in memory, so pressing one plays it straight away.
Every call opens with a spoken disclosure (your name, company, AI voice, recording notice)."""
import collections, json, os, queue, subprocess, sys, threading, time, webbrowser
from pathlib import Path

FROZEN = getattr(sys, "frozen", False)
BASE = Path(sys.executable).parent if FROZEN else Path(__file__).parent
if FROZEN:  # no console window: keep a log file instead
    sys.stdout = sys.stderr = open(BASE / "log.txt", "w", buffering=1, encoding="utf-8")
if (BASE / "models" / "whisper").exists():  # speech model shipped inside the package (works offline)
    os.environ.setdefault("WHISPER_MODEL", str(BASE / "models" / "whisper"))

import numpy as np
import sounddevice as sd
import uvicorn
from fastapi import Body, FastAPI
from fastapi.responses import FileResponse

import core  # voice + speech engine (Kokoro / Whisper)

HERE = BASE
core.MODELS = BASE / "models"
CFG_FILE = HERE / "settings.json"
PORT = 8766
DEFAULT = {
    "name": "", "company": "", "voice": "am_adam", "speed": 1.0, "silence": 300,
    "mic": None, "cable": None, "monitor": None, "live": False,
    "buttons": [
        {"label": "Yes", "text": "Yes."},
        {"label": "One moment", "text": "One moment please."},
        {"label": "Repeat", "text": "Could you please repeat that?"},
        {"label": "Thanks", "text": "Thank you, have a great day."},
    ],
}
cfg = dict(DEFAULT)
if CFG_FILE.exists():
    try:
        cfg.update(json.loads(CFG_FILE.read_text("utf-8")))
    except Exception:
        pass

S = {"call": False, "ready": False, "gen": 0, "heard": [], "prepared": 0, "total": 0, "err": ""}
engine = None
text_q, audio_q = queue.Queue(), queue.Queue()
out_streams, mic = {}, {"s": None}
clips, pending, busy = {}, set(), {}  # clips: finished audio per button text; pending: lines queued or playing
seg = core.Segmenter(cfg["silence"])


def disclosure():
    return ("Hi, this is %s from %s. I'm using an AI voice so I'm easy to understand, "
            "and this call may be recorded." % (cfg["name"], cfg["company"]))


# ---------- audio devices ----------
def wasapi():
    for i, h in enumerate(sd.query_hostapis()):
        if "WASAPI" in h["name"]:
            return i
    return None


def devices():
    ha, ins, outs = wasapi(), [], []
    for i, d in enumerate(sd.query_devices()):
        if ha is not None and d["hostapi"] != ha:
            continue
        if d["max_input_channels"] > 0:
            ins.append({"id": i, "name": d["name"]})
        if d["max_output_channels"] > 0:
            outs.append({"id": i, "name": d["name"]})
    return ins, outs


def extra(dev):
    if dev is not None and sd.query_devices(dev)["hostapi"] == wasapi():
        return sd.WasapiSettings(auto_convert=True)
    return None


class Out:
    """An output stream that stays open and plays whatever is put in its buffer.
    Starting a clip just adds it to the buffer, so there is no waiting and no stutter."""
    def __init__(self, dev, sr):
        self.buf, self.cur, self.pos = collections.deque(), None, 0
        self.stream = sd.OutputStream(samplerate=sr, device=dev, channels=1, dtype="float32",
                                      latency="low", extra_settings=extra(dev), callback=self.cb)
        self.stream.start()

    def cb(self, out, frames, t, status):
        i = 0
        try:
            while i < frames:
                if self.cur is None:
                    self.cur, self.pos = self.buf.popleft(), 0
                n = min(frames - i, len(self.cur) - self.pos)
                out[i:i + n, 0] = self.cur[self.pos:self.pos + n]
                i += n
                self.pos += n
                if self.pos >= len(self.cur):
                    self.cur = None
        except IndexError:
            pass
        out[i:] = 0

    def clear(self):
        self.buf.clear()
        self.cur = None


def play(samples, sr, gen):
    targets = [d for d in (cfg["cable"], cfg["monitor"]) if d is not None]
    if not targets:
        S["err"] = "Choose the virtual cable in settings first."
        return
    if gen != S["gen"]:
        return
    x = np.ascontiguousarray(samples, dtype=np.float32)
    try:
        for d in targets:
            o = out_streams.get((d, sr))
            if o is None:
                o = out_streams[(d, sr)] = Out(d, sr)
            o.buf.append(x)  # starts at once, does not wait for the audio to finish
    except Exception as e:
        out_streams.clear()
        S["err"] = "Audio output: %s" % e


def busy_check(text, secs):
    """True if this same line was started a moment ago (one press = one spoken line)."""
    now = time.time()
    if now < busy.get(text, 0):
        return True
    busy[text] = now + secs
    return False


# ---------- speaking ----------
def pieces(text):
    return [p for p in core.split_pieces(text) if any(c.isalnum() for c in p)]


def clip_of(text):
    c = clips.get((text, cfg["voice"], cfg["speed"]))
    if c is None:
        parts = [engine.speak(p, cfg["voice"], cfg["speed"]) for p in pieces(text)]
        if parts:
            c = (np.concatenate([a for a, _ in parts]), parts[0][1])
            clips[(text, cfg["voice"], cfg["speed"])] = c
            if len(clips) > 400:
                clips.pop(next(iter(clips)))
    return c


def say_text(text, gen):
    c = clips.get((text, cfg["voice"], cfg["speed"]))
    if c is not None:  # prepared button: one finished clip, plays at once
        play(c[0], c[1], gen)
        return
    for p in pieces(text):
        if gen != S["gen"]:
            return
        try:
            samples, sr = engine.speak(p, cfg["voice"], cfg["speed"])  # instant when prepared
        except Exception as e:
            S["err"] = str(e)
            continue
        play(samples, sr, gen)


def tts_worker():
    while True:
        text, gen = text_q.get()
        try:
            if gen == S["gen"] and S["call"]:
                say_text(text, gen)
        finally:
            pending.discard(text)


def prepare():
    """Voice every button now, so nothing has to be made while the customer waits."""
    def run():
        bs = list(cfg["buttons"])
        texts = [b["text"] for b in bs]
        if cfg["name"] and cfg["company"]:
            texts.append(disclosure())
        S["total"], S["prepared"] = len(texts), 0
        for t in texts:
            try:
                clip_of(t)
            except Exception as e:
                S["err"] = str(e)
            S["prepared"] += 1
    threading.Thread(target=run, daemon=True).start()


# ---------- optional live voice (speak instead of type) ----------
def mic_cb(indata, frames, t, status):
    if S["call"] and cfg["live"]:
        for sg in seg.push(indata[:, 0]):
            audio_q.put(sg)


def sync_mic():
    if mic["s"]:
        try:
            mic["s"].close()
        except Exception:
            pass
        mic["s"] = None
    if not (S["call"] and cfg["live"]):
        return
    try:
        s = sd.InputStream(samplerate=16000, device=cfg["mic"], channels=1, dtype="int16",
                           blocksize=320, callback=mic_cb, extra_settings=extra(cfg["mic"]))
        s.start()
        mic["s"] = s
    except Exception as e:
        S["err"] = "Microphone: %s" % e


def stt_worker():
    while True:
        a = audio_q.get()
        if not (S["call"] and cfg["live"]):
            continue
        try:
            t = engine.transcribe(a)
        except Exception as e:
            S["err"] = "Speech recognition: %s" % e
            continue
        if t:
            S["heard"] = (S["heard"] + [t])[-6:]
            enqueue(t)


def enqueue(text):
    c = clips.get((text, cfg["voice"], cfg["speed"]))
    if busy_check(text, (len(c[0]) / c[1] + 0.3) if c else 2.0):
        return
    text_q.put((text, S["gen"]))


def stop():
    S["gen"] += 1
    busy.clear()
    for o in list(out_streams.values()):
        o.clear()
    while not text_q.empty():
        try:
            text_q.get_nowait()
        except queue.Empty:
            break


# ---------- local web interface ----------
app = FastAPI()


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/api/state")
def state():
    ins, outs = devices()
    return {"cfg": cfg, "ins": ins, "outs": outs, "voices": engine.voices if engine else [],
            "cable_found": any("CABLE" in d["name"] for d in outs), "call": S["call"], "ready": S["ready"], "heard": S["heard"], "err": S["err"],
            "prepared": S["prepared"], "total": S["total"], "disclosure": disclosure()}


@app.post("/api/config")
def config(body: dict = Body(...)):
    old = (cfg["voice"], cfg["speed"], json.dumps(cfg["buttons"]), cfg["name"], cfg["company"])
    for k in DEFAULT:
        if k in body:
            cfg[k] = body[k]
    cfg["buttons"] = [b for b in cfg["buttons"] if b.get("text")][:30]
    CFG_FILE.write_text(json.dumps(cfg, indent=1), "utf-8")
    seg.set_silence(int(cfg["silence"]))
    S["err"] = ""
    if engine and old != (cfg["voice"], cfg["speed"], json.dumps(cfg["buttons"]), cfg["name"], cfg["company"]):
        prepare()
    sync_mic()
    return {"ok": True}


@app.post("/api/call")
def call(body: dict = Body(...)):
    if body.get("on"):
        if not (cfg["name"].strip() and cfg["company"].strip()):
            return {"error": "Enter your name and company first. They are spoken at the start of every call."}
        if cfg["cable"] is None:
            return {"error": "Choose the virtual cable in settings first."}
        S["call"] = True
        enqueue(disclosure())  # the call always opens with the disclosure
    else:
        stop()
        S["call"] = False
    sync_mic()
    return {"ok": True}


@app.post("/api/say")
def say(body: dict = Body(...)):
    t = str(body.get("text", "")).strip()[:2000]
    if S["call"] and t:
        enqueue(t)
    return {"ok": S["call"]}


@app.post("/api/button")
def button(body: dict = Body(...)):
    i = int(body.get("i", -1))
    if S["call"] and 0 <= i < len(cfg["buttons"]):
        t = cfg["buttons"][i]["text"]
        c = clips.get((t, cfg["voice"], cfg["speed"]))
        if c is None:
            enqueue(t)
        elif not busy_check(t, len(c[0]) / c[1] + 0.3):
            play(c[0], c[1], S["gen"])  # already made: plays immediately
    return {"ok": S["call"]}


prep_lock = threading.Lock()


@app.post("/api/prepare")
def prepare_line(body: dict = Body(...)):
    """Voice a line while it is being typed, so Enter plays it at once."""
    t = str(body.get("text", "")).strip()[:300]
    if engine and t:
        S["want"] = t

        def run():
            with prep_lock:
                if S.get("want") != t:  # a newer line is being typed; skip this one
                    return
                try:
                    clip_of(t)
                except Exception:
                    pass
        threading.Thread(target=run, daemon=True).start()
    return {"ok": True}


@app.post("/api/install_cable")
def install_cable():
    """One-click install of the free VB-Cable virtual microphone driver (Windows asks for admin permission)."""
    def run():
        try:
            import ctypes, tempfile, urllib.request, zipfile
            d = Path(tempfile.mkdtemp())
            z = d / "vbcable.zip"
            urllib.request.urlretrieve("https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack43.zip", z)
            zipfile.ZipFile(z).extractall(d)
            ctypes.windll.shell32.ShellExecuteW(None, "runas", str(d / "VBCABLE_Setup_x64.exe"), None, str(d), 1)
        except Exception:
            webbrowser.open("https://vb-audio.com/Cable/")
    threading.Thread(target=run, daemon=True).start()
    return {"ok": True}


@app.post("/api/stop")
def stop_api():
    stop()
    return {"ok": True}


def autoselect():
    _, outs = devices()
    if cfg["cable"] is None:
        for d in outs:
            if "CABLE Input" in d["name"]:
                cfg["cable"] = d["id"]


def boot():
    global engine
    try:
        autoselect()
        engine = core.Engine()
        if cfg["voice"] not in engine.voices:
            cfg["voice"] = "am_adam" if "am_adam" in engine.voices else engine.voices[0]
        engine.speak("Ready.", cfg["voice"], 1.0)
        prepare()
        S["ready"] = True
        if cfg["live"]:
            threading.Thread(target=engine.preload, daemon=True).start()
    except SystemExit:
        S["err"] = "Voice model files are missing. See README.txt."
    except Exception as e:
        S["err"] = "Engine could not start: %s" % e


def open_window():
    url = "http://127.0.0.1:%d" % PORT
    if sys.platform == "win32":
        subprocess.Popen('start "" msedge --app=%s --window-size=1040,780' % url, shell=True)
    else:
        webbrowser.open(url)


if __name__ == "__main__":
    core.set_high_priority()
    for f in (tts_worker, stt_worker, boot):
        threading.Thread(target=f, daemon=True).start()
    threading.Timer(2.0, open_window).start()
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_config=None)
