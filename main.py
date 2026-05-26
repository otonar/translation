"""
リアルタイム英語音声 → 日本語翻訳アプリ
  - 音声入力: sounddevice (ステレオミキサー対応)
  - 文字起こし: faster-whisper (完全ローカル・無料)
  - 翻訳: Google Cloud Translation API (要APIキー・要インターネット)
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import queue
import numpy as np
import sounddevice as sd
import time
import logging
import urllib.request
import json
import os

logging.basicConfig(level=logging.WARNING)


def _load_dotenv() -> None:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            val = value.strip()
            if val and val[0] in ('"', "'"):
                quote = val[0]
                end = val.find(quote, 1)
                val = val[1:end] if end != -1 else val[1:]
            else:
                val = val.split("#")[0].strip()
            os.environ.setdefault(key.strip(), val)


_load_dotenv()


def _get_api_key() -> str:
    return os.environ.get("GOOGLE_TRANSLATE_API_KEY", "")


_MAX_RESPONSE_BYTES = 1_048_576  # 1MB


def _google_translate(text: str) -> str:
    url = "https://translation.googleapis.com/language/translate/v2"
    body = json.dumps({"q": text, "source": "en", "target": "ja", "format": "text"}).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        "X-Goog-Api-Key": _get_api_key(),
    })
    with urllib.request.urlopen(req, timeout=5) as r:
        raw = r.read(_MAX_RESPONSE_BYTES)
        if len(raw) >= _MAX_RESPONSE_BYTES:
            raise RuntimeError("Google API レスポンスが大きすぎます")
        data = json.loads(raw)
    if "error" in data:
        err = data["error"]
        raise RuntimeError(f"Google API エラー {err.get('code', '?')}: {err.get('message', err)}")
    return data["data"]["translations"][0]["translatedText"]


# ─── 設定 ───────────────────────────────────────────────────────────────────
SAMPLE_RATE        = 16000
CHUNK_DURATION     = 0.3    # 秒: 音声バッファの処理単位
CHUNK_SIZE         = int(SAMPLE_RATE * CHUNK_DURATION)
SILENCE_THRESHOLD  = 0.005  # RMSエネルギーの無音判定閾値
SILENCE_DURATION   = 1.5    # 秒: この長さ無音が続いたら発話終了とみなす
MAX_UTTERANCE_SEC  = 30     # 秒: 強制処理の上限

# IT系専門用語の認識を高めるプロンプト
IT_PROMPT = (
    "Technical discussion: software engineering, programming, API, REST, GraphQL, "
    "microservices, Kubernetes, Docker, CI/CD, DevOps, Git, GitHub, cloud computing, "
    "AWS, Azure, GCP, database, SQL, NoSQL, machine learning, AI, LLM, neural network, "
    "cybersecurity, networking, TCP/IP, HTTP, TLS, authentication, OAuth, JWT."
)

COLORS = {
    "bg":       "#1e1e2e",
    "surface":  "#2a2a3d",
    "surface2": "#313147",
    "text":     "#cdd6f4",
    "subtext":  "#a6adc8",
    "green":    "#a6e3a1",
    "red":      "#f38ba8",
    "yellow":   "#f9e2af",
    "blue":     "#89b4fa",
    "mauve":    "#cba6f7",
}


# ─── 音声バッファ・VAD ───────────────────────────────────────────────────────
class AudioProcessor:
    """エネルギーベースのVAD付き音声バッファ。発話単位で音声を切り出す。"""

    def __init__(self):
        self._buf: list[np.ndarray] = []
        self._silence = 0
        self.is_speaking = False
        self.ready: queue.Queue = queue.Queue(maxsize=5)
        self._silence_need = int(SILENCE_DURATION / CHUNK_DURATION)
        self._max_chunks   = int(MAX_UTTERANCE_SEC / CHUNK_DURATION)
        self._dropped = threading.Event()

    def feed(self, chunk: np.ndarray):
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms > SILENCE_THRESHOLD:
            self._buf.append(chunk)
            self._silence = 0
            self.is_speaking = True
            if len(self._buf) >= self._max_chunks:
                self._flush()
        elif self.is_speaking:
            self._buf.append(chunk)
            self._silence += 1
            if self._silence >= self._silence_need or len(self._buf) >= self._max_chunks:
                self._flush()

    def _flush(self):
        if self._buf:
            try:
                self.ready.put_nowait(np.concatenate(self._buf))
            except queue.Full:
                self._dropped.set()
        self._buf = []
        self._silence = 0
        self.is_speaking = False

    def pop_dropped(self) -> bool:
        """発話がドロップされていた場合 True を返し、フラグをリセットする。"""
        if self._dropped.is_set():
            self._dropped.clear()
            return True
        return False

    def reset(self):
        self._buf = []
        self._silence = 0
        self.is_speaking = False
        while not self.ready.empty():
            try:
                self.ready.get_nowait()
            except queue.Empty:
                break


# ─── メインアプリ ────────────────────────────────────────────────────────────
class TranslatorApp:
    def __init__(self):
        self.model = None
        self._current_model_size = ""
        self.stream = None
        self._running = threading.Event()
        self._log_file = None
        self._log_path = None
        self._audio_q: queue.Queue = queue.Queue(maxsize=50)  # ~15秒分、上限付きで無制限増加を防ぐ
        self._result_q: queue.Queue = queue.Queue()
        self._vad = AudioProcessor()
        self._device_indices: list[int] = []
        self._feed_thread: threading.Thread | None = None
        self._transcribe_thread: threading.Thread | None = None

        self._build_gui()

    # ── GUI ──────────────────────────────────────────────────────────────────

    def _build_gui(self):
        self.root = tk.Tk()
        self.root.title("リアルタイム英語 → 日本語翻訳")
        self.root.geometry("1020x740")
        self.root.configure(bg=COLORS["bg"])
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._apply_ttk_style()
        self._build_controls()
        self._build_statusbar()
        self._build_text_panes()

        self._refresh_devices()
        self.root.after(100, self._poll_results)

    def _apply_ttk_style(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TCombobox",
                    fieldbackground=COLORS["surface2"],
                    background=COLORS["surface2"],
                    foreground=COLORS["text"],
                    selectbackground=COLORS["mauve"],
                    selectforeground=COLORS["bg"])
        s.map("TCombobox", fieldbackground=[("readonly", COLORS["surface2"])])

    def _build_controls(self):
        bar = tk.Frame(self.root, bg=COLORS["bg"], pady=12)
        bar.pack(fill=tk.X, padx=20)

        # デバイス
        lbl = lambda text: tk.Label(bar, text=text, bg=COLORS["bg"],
                                    fg=COLORS["subtext"], font=("Yu Gothic UI", 10))
        lbl("入力デバイス:").pack(side=tk.LEFT)
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(bar, textvariable=self.device_var,
                                         width=44, state="readonly",
                                         font=("Yu Gothic UI", 10))
        self.device_combo.pack(side=tk.LEFT, padx=(6, 4))

        def _btn(parent, text, cmd, **kw):
            b = tk.Button(parent, text=text, command=cmd, relief="flat",
                          cursor="hand2", **kw)
            return b

        _btn(bar, "↺", self._refresh_devices,
             bg=COLORS["surface2"], fg=COLORS["subtext"],
             font=("Arial", 12), padx=6, pady=0).pack(side=tk.LEFT, padx=(0, 18))

        # モデル
        lbl("モデル:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value="small")
        ttk.Combobox(bar, textvariable=self.model_var, width=10,
                     values=["tiny", "base", "small", "medium"],
                     state="readonly", font=("Yu Gothic UI", 10)
                     ).pack(side=tk.LEFT, padx=(6, 0))
        tk.Label(bar, text="← Zoom中はbase推奨", bg=COLORS["bg"],
                 fg=COLORS["subtext"], font=("Yu Gothic UI", 9)
                 ).pack(side=tk.LEFT, padx=(4, 0))

        # 右側ボタン群
        _btn(bar, "クリア", self._clear,
             bg=COLORS["surface2"], fg=COLORS["subtext"],
             font=("Yu Gothic UI", 10), padx=12, pady=4
             ).pack(side=tk.RIGHT, padx=(8, 0))

        self.toggle_btn = _btn(bar, "  開始  ", self._toggle,
                               bg=COLORS["green"], fg=COLORS["bg"],
                               font=("Yu Gothic UI", 11, "bold"), padx=16, pady=5)
        self.toggle_btn.pack(side=tk.RIGHT)

    def _build_statusbar(self):
        bar = tk.Frame(self.root, bg=COLORS["surface"], pady=5)
        bar.pack(fill=tk.X, padx=20, pady=(0, 8))

        self._led = tk.Label(bar, text="●", bg=COLORS["surface"],
                             fg=COLORS["subtext"], font=("Arial", 10))
        self._led.pack(side=tk.LEFT, padx=(10, 4))

        self._status_var = tk.StringVar(value="準備完了 — デバイスを選択して「開始」を押してください")
        self._status_label = tk.Label(bar, textvariable=self._status_var, bg=COLORS["surface"],
                 fg=COLORS["subtext"], font=("Yu Gothic UI", 10),
                 anchor="w")
        self._status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._speaking_var = tk.StringVar(value="")
        tk.Label(bar, textvariable=self._speaking_var, bg=COLORS["surface"],
                 fg=COLORS["yellow"], font=("Yu Gothic UI", 10)
                 ).pack(side=tk.RIGHT, padx=12)

    def _build_text_panes(self):
        outer = tk.Frame(self.root, bg=COLORS["bg"])
        outer.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 16))

        pw = tk.PanedWindow(outer, orient=tk.HORIZONTAL, bg=COLORS["bg"],
                            sashwidth=6, sashrelief="flat", sashpad=2)
        pw.pack(fill=tk.BOTH, expand=True)

        def _pane(parent, label, color):
            f = tk.Frame(parent, bg=COLORS["bg"])
            tk.Label(f, text=label, bg=COLORS["bg"], fg=color,
                     font=("Yu Gothic UI", 10, "bold"), pady=4).pack(anchor="w")
            t = scrolledtext.ScrolledText(
                f, wrap=tk.WORD, bg=COLORS["surface"], fg=COLORS["text"],
                relief="flat", padx=12, pady=10,
                insertbackground=COLORS["text"],
                selectbackground=COLORS["mauve"])
            t.pack(fill=tk.BOTH, expand=True)
            return f, t

        en_f, self.en_text = _pane(pw, "英語（原文）", COLORS["blue"])
        self.en_text.configure(font=("Consolas", 11))
        pw.add(en_f, minsize=250)

        ja_f, self.ja_text = _pane(pw, "日本語（翻訳）", COLORS["green"])
        self.ja_text.configure(font=("Yu Gothic UI", 11))
        pw.add(ja_f, minsize=250)

    # ── デバイス管理 ──────────────────────────────────────────────────────────

    def _refresh_devices(self):
        try:
            self._device_indices = []
            names = []
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] > 0:
                    self._device_indices.append(i)
                    names.append(d["name"])
            self.device_combo["values"] = names
            # ステレオミキサーを自動選択
            keywords = ["stereo", "ステレオ", "mix", "loopback", "what u hear",
                        "wave out", "sum", "システム"]
            for j, name in enumerate(names):
                if any(kw in name.lower() for kw in keywords):
                    self.device_combo.current(j)
                    return
            if names:
                self.device_combo.current(0)
        except Exception as e:
            self._set_status(f"デバイス取得エラー: {e}", "red")

    # ── 開始 / 停止 ───────────────────────────────────────────────────────────

    def _toggle(self):
        if not self._running.is_set():
            self._start()
        else:
            self._stop()

    def _start(self):
        model_size = self.model_var.get()
        device_sel = self.device_combo.current()
        self.toggle_btn.config(state="disabled")
        if self.model is None or self._current_model_size != model_size:
            self._set_status("モデルを読み込み中... (初回は数百MBのダウンロードが発生します)", "yellow")
        else:
            self._set_status("再開準備中...", "yellow")
        threading.Thread(target=self._load_and_start, args=(model_size, device_sel), daemon=True).start()

    def _load_and_start(self, model_size: str, device_sel: int):
        try:
            if not _get_api_key():
                raise ValueError(
                    "GOOGLE_TRANSLATE_API_KEY が未設定です。\n"
                    "環境変数を設定してからアプリを起動してください。"
                )

            from faster_whisper import WhisperModel

            if self.model is None or self._current_model_size != model_size:
                self._set_status(f"Whisper [{model_size}] を初期化中...", "yellow")
                self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
                self._current_model_size = model_size

            if device_sel < 0 or device_sel >= len(self._device_indices):
                raise ValueError("入力デバイスが選択されていません")
            device_idx = self._device_indices[device_sel]

            self._running.set()
            self._vad.reset()
            # 停止→再開時に残った古い音声チャンクを破棄
            while not self._audio_q.empty():
                try:
                    self._audio_q.get_nowait()
                except queue.Empty:
                    break

            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_name = time.strftime("translation_%Y%m%d_%H%M%S.txt")
            self._log_path = os.path.join(log_dir, log_name)
            self._log_file = open(self._log_path, "w", encoding="utf-8")
            self._log_file.write(
                f"# 翻訳セッション: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"# {'─' * 40}\n\n")
            self._log_file.flush()

            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                blocksize=CHUNK_SIZE, device=device_idx,
                callback=self._audio_cb)
            self.stream.start()

            self._feed_thread = threading.Thread(target=self._feed_loop, daemon=True)
            self._transcribe_thread = threading.Thread(target=self._transcribe_loop, daemon=True)
            self._feed_thread.start()
            self._transcribe_thread.start()

            log_name_cap = log_name
            self.root.after(0, lambda n=log_name_cap: (
                self.toggle_btn.config(text="  停止  ", bg=COLORS["red"],
                                       fg=COLORS["bg"], state="normal"),
                self._set_status(f"録音中 — {n} に保存中", "green"),
                self._led.config(fg=COLORS["green"]),
            ))
        except Exception as exc:
            if self._log_file:
                try:
                    self._log_file.close()
                except Exception:
                    pass
                self._log_file = None
            if self.stream:
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None
            self._running.clear()
            self.root.after(0, lambda e=exc: (
                self.toggle_btn.config(state="normal"),
                self._set_status(f"起動エラー: {e}", "red"),
                self._led.config(fg=COLORS["red"]),
            ))

    def _stop(self):
        self._running.clear()
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        for t in (self._feed_thread, self._transcribe_thread):
            if t and t.is_alive():
                t.join(timeout=2.0)
        while True:
            try:
                en, ja = self._result_q.get_nowait()
                ts = time.strftime("%H:%M:%S")
                self.en_text.insert(tk.END, f"[{ts}]  {en}\n\n")
                self.en_text.see(tk.END)
                self.ja_text.insert(tk.END, f"[{ts}]  {ja}\n\n")
                self.ja_text.see(tk.END)
                if self._log_file:
                    self._log_file.write(f"[{ts}]\nEN: {en}\nJA: {ja}\n\n")
            except queue.Empty:
                break
        saved_name = None
        if self._log_file:
            self._log_file.close()
            self._log_file = None
            saved_name = os.path.basename(self._log_path) if self._log_path else None
        self.toggle_btn.config(text="  開始  ", bg=COLORS["green"], fg=COLORS["bg"])
        msg = f"停止 — {saved_name} に保存済み" if saved_name else "停止しました"
        self._set_status(msg)
        self._led.config(fg=COLORS["subtext"])
        self._speaking_var.set("")

    def _clear(self):
        self.en_text.delete("1.0", tk.END)
        self.ja_text.delete("1.0", tk.END)

    # ── 音声パイプライン ───────────────────────────────────────────────────────

    def _audio_cb(self, indata, frames, time_info, status):
        if self._running.is_set():
            try:
                self._audio_q.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass  # バックプレッシャー時はフレームを捨てる（コールバックをブロックしない）

    def _feed_loop(self):
        """音声チャンクをVADプロセッサに流し込む。"""
        while self._running.is_set():
            try:
                chunk = self._audio_q.get(timeout=0.5)
                was_speaking = self._vad.is_speaking
                self._vad.feed(chunk)
                now_speaking = self._vad.is_speaking
                if now_speaking and not was_speaking:
                    self.root.after(0, lambda: self._speaking_var.set("● 音声検出中..."))
                elif not now_speaking and was_speaking:
                    self.root.after(0, lambda: self._speaking_var.set(""))
                if self._vad.pop_dropped():
                    self._set_status("処理が追いつかず発話をスキップしました（モデルを small → base に変更を推奨）", "yellow")
            except queue.Empty:
                continue

    def _transcribe_loop(self):
        """発話バッファを受け取り、文字起こし→翻訳する。"""
        while self._running.is_set():
            try:
                audio = self._vad.ready.get(timeout=0.5)
                self._set_status("文字起こし中...", "yellow")
                self._run(audio)
                if self._running.is_set():
                    self._set_status("録音中", "green")
            except queue.Empty:
                continue

    def _run(self, audio: np.ndarray):
        model = self.model
        if model is None:
            return
        try:
            segments, _ = model.transcribe(
                audio,
                language="en",
                vad_filter=True,
                initial_prompt=IT_PROMPT,
                word_timestamps=False,
            )
            text = " ".join(s.text.strip() for s in segments).strip()
            if not text:
                return

            try:
                translation = _google_translate(text)
            except Exception as te:
                logging.warning("translation error: %s", te)
                translation = "[翻訳エラー（ログを確認してください）]"

            self._result_q.put((text, translation))
        except Exception as e:
            logging.error("transcribe error: %s", e)

    # ── 結果表示 ──────────────────────────────────────────────────────────────

    def _poll_results(self):
        try:
            while True:
                en, ja = self._result_q.get_nowait()
                ts = time.strftime("%H:%M:%S")
                self.en_text.insert(tk.END, f"[{ts}]  {en}\n\n")
                self.en_text.see(tk.END)
                self.ja_text.insert(tk.END, f"[{ts}]  {ja}\n\n")
                self.ja_text.see(tk.END)
                if self._log_file:
                    self._log_file.write(f"[{ts}]\nEN: {en}\nJA: {ja}\n\n")
                    self._log_file.flush()
        except queue.Empty:
            pass
        self.root.after(100, self._poll_results)

    # ── ユーティリティ ────────────────────────────────────────────────────────

    def _set_status(self, msg: str, color_key: str = "subtext"):
        color = COLORS.get(color_key, COLORS["subtext"])
        self.root.after(0, lambda m=msg, c=color: (
            self._status_var.set(m),
            self._status_label.config(fg=c),
        ))

    def _on_close(self):
        if self._running.is_set():
            self._stop()
        self.model = None
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = TranslatorApp()
    app.run()
