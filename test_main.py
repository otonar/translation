"""
main.py のユニットテスト
実行: python -m pytest test_main.py -v
対象: 純粋ロジック (_load_dotenv, _google_translate, AudioProcessor)
      ※ tkinter/sounddevice/WhisperModel は import 前にモック化
"""

import io
import json
import os
import queue
import sys
import threading
import time
import unittest
from unittest import mock

import numpy as np

# ─── tkinter / sounddevice を import 前にモック化 ────────────────────────────
_tk_mock = mock.MagicMock()
sys.modules.setdefault("tkinter", _tk_mock)
sys.modules.setdefault("tkinter.ttk", _tk_mock.ttk)
sys.modules.setdefault("tkinter.scrolledtext", _tk_mock.scrolledtext)
sys.modules.setdefault("sounddevice", mock.MagicMock())

import main  # noqa: E402  (モック後に import)


# ═══════════════════════════════════════════════════════════════════════════════
# _load_dotenv
# ═══════════════════════════════════════════════════════════════════════════════
class TestLoadDotenv(unittest.TestCase):

    def _run(self, content: str, env: dict | None = None) -> dict:
        """content を .env として読み込み、設定された環境変数を返す。"""
        env = env or {}
        with mock.patch("os.path.exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data=content)), \
             mock.patch.dict(os.environ, env, clear=True):
            main._load_dotenv()
            return dict(os.environ)

    # ── 正常系 ──────────────────────────────────────────────────────────────

    def test_basic_key_value(self):
        result = self._run("FOO=bar\n")
        self.assertEqual(result["FOO"], "bar")

    def test_multiple_pairs(self):
        result = self._run("A=1\nB=2\nC=3\n")
        self.assertEqual(result["A"], "1")
        self.assertEqual(result["B"], "2")
        self.assertEqual(result["C"], "3")

    def test_double_quoted_value(self):
        result = self._run('KEY="hello world"\n')
        self.assertEqual(result["KEY"], "hello world")

    def test_single_quoted_value(self):
        result = self._run("KEY='hello world'\n")
        self.assertEqual(result["KEY"], "hello world")

    def test_value_with_equals_sign(self):
        """= を含む値は最初の = で分割され残りが値になる。"""
        result = self._run("KEY=abc=def==\n")
        self.assertEqual(result["KEY"], "abc=def==")

    def test_whitespace_stripped_from_key_and_value(self):
        result = self._run("  KEY  =  value  \n")
        self.assertEqual(result["KEY"], "value")

    def test_empty_value(self):
        result = self._run("KEY=\n")
        self.assertEqual(result["KEY"], "")

    # ── スキップ行 ───────────────────────────────────────────────────────────

    def test_comment_lines_ignored(self):
        result = self._run("# comment\nKEY=val\n")
        self.assertNotIn("# comment", result)
        self.assertEqual(result["KEY"], "val")

    def test_empty_lines_ignored(self):
        result = self._run("\n\nKEY=val\n\n")
        self.assertEqual(result["KEY"], "val")

    def test_line_without_equals_ignored(self):
        result = self._run("INVALID\nKEY=val\n")
        self.assertNotIn("INVALID", result)

    # ── setdefault: 既存環境変数は上書きしない ──────────────────────────────

    def test_existing_env_not_overwritten(self):
        result = self._run("KEY=new\n", env={"KEY": "original"})
        self.assertEqual(result["KEY"], "original")

    def test_new_key_added_alongside_existing(self):
        result = self._run("NEW=added\n", env={"EXISTING": "kept"})
        self.assertEqual(result["EXISTING"], "kept")
        self.assertEqual(result["NEW"], "added")

    # ── ファイルなし ─────────────────────────────────────────────────────────

    def test_no_env_file_no_error(self):
        with mock.patch("os.path.exists", return_value=False):
            main._load_dotenv()  # 例外が出ないこと

    # ── PermissionError は伝播する ───────────────────────────────────────────

    def test_permission_error_propagates(self):
        with mock.patch("os.path.exists", return_value=True), \
             mock.patch("builtins.open", side_effect=PermissionError("denied")):
            with self.assertRaises(PermissionError):
                main._load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════════
# _google_translate
# ═══════════════════════════════════════════════════════════════════════════════
class TestGoogleTranslate(unittest.TestCase):

    def _mock_urlopen(self, translated_text: str):
        """正常なAPIレスポンスを返すモックコンテキストマネージャを作る。"""
        resp_bytes = json.dumps({
            "data": {"translations": [{"translatedText": translated_text}]}
        }).encode()
        resp = mock.MagicMock()
        resp.read.side_effect = lambda n=None: resp_bytes[:n] if n else resp_bytes
        resp.__enter__ = mock.Mock(return_value=resp)
        resp.__exit__ = mock.Mock(return_value=False)
        return resp

    # ── 正常系 ──────────────────────────────────────────────────────────────

    def test_basic_translation(self):
        with mock.patch("urllib.request.urlopen", return_value=self._mock_urlopen("こんにちは")), \
             mock.patch("main._get_api_key", return_value="fake_key"):
            self.assertEqual(main._google_translate("Hello"), "こんにちは")

    def test_empty_string(self):
        with mock.patch("urllib.request.urlopen", return_value=self._mock_urlopen("")), \
             mock.patch("main._get_api_key", return_value="fake_key"):
            self.assertEqual(main._google_translate(""), "")

    def test_special_characters(self):
        with mock.patch("urllib.request.urlopen", return_value=self._mock_urlopen("特殊")), \
             mock.patch("main._get_api_key", return_value="fake_key"):
            result = main._google_translate("!@#$%^&*()<>")
            self.assertEqual(result, "特殊")

    # ── リクエスト内容の検証 ────────────────────────────────────────────────

    def test_api_key_in_header_not_url(self):
        """APIキーはヘッダーに含まれ、URLには含まれない（ログ漏洩防止）。"""
        captured = []
        resp = self._mock_urlopen("テスト")
        def capture(req, timeout=None):
            captured.append(req)
            return resp
        with mock.patch("urllib.request.urlopen", side_effect=capture), \
             mock.patch("main._get_api_key", return_value="secret_key_abc"):
            main._google_translate("test")
        req = captured[0]
        self.assertNotIn("secret_key_abc", req.full_url)
        self.assertIn("secret_key_abc", req.headers.get("X-goog-api-key", ""))

    def test_request_body_json(self):
        """リクエストボディに入力テキスト・source・target が含まれる。"""
        captured = []
        resp = self._mock_urlopen("テスト")
        def capture(req, timeout=None):
            captured.append(req)
            return resp
        with mock.patch("urllib.request.urlopen", side_effect=capture), \
             mock.patch("main._get_api_key", return_value="fake"):
            main._google_translate("hello world")
        body = json.loads(captured[0].data)
        self.assertEqual(body["q"], "hello world")
        self.assertEqual(body["source"], "en")
        self.assertEqual(body["target"], "ja")
        self.assertEqual(body["format"], "text")

    def test_timeout_is_set(self):
        """urlopen に timeout 引数が渡される。"""
        captured = []
        resp = self._mock_urlopen("テスト")
        def capture(req, timeout=None):
            captured.append(timeout)
            return resp
        with mock.patch("urllib.request.urlopen", side_effect=capture), \
             mock.patch("main._get_api_key", return_value="fake"):
            main._google_translate("test")
        self.assertIsNotNone(captured[0])
        self.assertGreater(captured[0], 0)

    # ── エラー系 ────────────────────────────────────────────────────────────

    def test_timeout_exception_propagates(self):
        with mock.patch("urllib.request.urlopen", side_effect=TimeoutError("timeout")), \
             mock.patch("main._get_api_key", return_value="fake"):
            with self.assertRaises(TimeoutError):
                main._google_translate("test")

    def test_api_error_response_raises_runtime_error(self):
        """APIがエラーJSONを返した場合、コードとメッセージ付きの RuntimeError が発生する。"""
        error_bytes = json.dumps({"error": {"code": 403, "message": "Forbidden"}}).encode()
        resp = mock.MagicMock()
        resp.read.return_value = error_bytes
        resp.__enter__ = mock.Mock(return_value=resp)
        resp.__exit__ = mock.Mock(return_value=False)
        with mock.patch("urllib.request.urlopen", return_value=resp), \
             mock.patch("main._get_api_key", return_value="fake"):
            with self.assertRaises(RuntimeError) as ctx:
                main._google_translate("test")
        self.assertIn("403", str(ctx.exception))
        self.assertIn("Forbidden", str(ctx.exception))

    def test_large_response_works_without_truncation(self):
        """65536バイトを超えるレスポンスも正しく読み取れる（read() 上限なし）。"""
        huge = "あ" * 40000   # UTF-8 で 120,000 バイト
        full_bytes = json.dumps({
            "data": {"translations": [{"translatedText": huge}]}
        }).encode()
        self.assertGreater(len(full_bytes), 65536, "テスト前提条件: 65536バイト超")

        resp = mock.MagicMock()
        resp.read.return_value = full_bytes
        resp.__enter__ = mock.Mock(return_value=resp)
        resp.__exit__ = mock.Mock(return_value=False)
        with mock.patch("urllib.request.urlopen", return_value=resp), \
             mock.patch("main._get_api_key", return_value="fake"):
            result = main._google_translate("large")
        self.assertEqual(result, huge)

    def test_unicode_emoji_in_translation(self):
        """翻訳結果に絵文字が含まれても正しく返る。"""
        with mock.patch("urllib.request.urlopen", return_value=self._mock_urlopen("テスト🎉")), \
             mock.patch("main._get_api_key", return_value="fake"):
            self.assertEqual(main._google_translate("Test"), "テスト🎉")


# ═══════════════════════════════════════════════════════════════════════════════
# AudioProcessor (VAD)
# ═══════════════════════════════════════════════════════════════════════════════
class TestAudioProcessor(unittest.TestCase):

    def _loud(self, size=main.CHUNK_SIZE) -> np.ndarray:
        t = np.linspace(0, 2 * np.pi, size)
        return (np.sin(t) * 0.1).astype(np.float32)   # RMS ≈ 0.071 >> 0.005

    def _silent(self, size=main.CHUNK_SIZE) -> np.ndarray:
        return np.zeros(size, dtype=np.float32)

    # ── 初期状態 ────────────────────────────────────────────────────────────

    def test_initial_not_speaking(self):
        vad = main.AudioProcessor()
        self.assertFalse(vad.is_speaking)
        self.assertTrue(vad.ready.empty())
        self.assertEqual(len(vad._buf), 0)
        self.assertEqual(vad._silence, 0)

    # ── 発話検出 ────────────────────────────────────────────────────────────

    def test_loud_chunk_starts_speaking(self):
        vad = main.AudioProcessor()
        vad.feed(self._loud())
        self.assertTrue(vad.is_speaking)

    def test_silent_chunk_alone_no_speech(self):
        vad = main.AudioProcessor()
        vad.feed(self._silent())
        self.assertFalse(vad.is_speaking)

    def test_speaking_resets_silence_counter(self):
        vad = main.AudioProcessor()
        vad.feed(self._loud())
        half = vad._silence_need // 2
        for _ in range(half):
            vad.feed(self._silent())
        self.assertEqual(vad._silence, half)
        vad.feed(self._loud())   # 再び発話 → カウンタがリセット
        self.assertEqual(vad._silence, 0)

    # ── フラッシュ（無音タイムアウト） ──────────────────────────────────────

    def test_silence_timeout_flushes(self):
        vad = main.AudioProcessor()
        vad.feed(self._loud())
        for _ in range(vad._silence_need):
            vad.feed(self._silent())
        self.assertFalse(vad.ready.empty())
        self.assertFalse(vad.is_speaking)
        self.assertEqual(len(vad._buf), 0)

    def test_partial_silence_does_not_flush(self):
        """_silence_need - 1 回の無音ではフラッシュしない。"""
        vad = main.AudioProcessor()
        vad.feed(self._loud())
        for _ in range(vad._silence_need - 1):
            vad.feed(self._silent())
        self.assertTrue(vad.ready.empty())

    def test_flushed_audio_is_numpy_array(self):
        vad = main.AudioProcessor()
        vad.feed(self._loud())
        for _ in range(vad._silence_need):
            vad.feed(self._silent())
        audio = vad.ready.get_nowait()
        self.assertIsInstance(audio, np.ndarray)

    def test_flushed_audio_contains_all_chunks(self):
        """フラッシュされた配列は speech + trailing silence を含む。"""
        vad = main.AudioProcessor()
        vad.feed(self._loud())
        for _ in range(vad._silence_need):
            vad.feed(self._silent())
        audio = vad.ready.get_nowait()
        expected_min = main.CHUNK_SIZE * (1 + vad._silence_need)
        self.assertEqual(len(audio), expected_min)

    # ── 強制フラッシュ（最大発話長） ────────────────────────────────────────

    def test_max_utterance_forces_flush(self):
        vad = main.AudioProcessor()
        for _ in range(vad._max_chunks + 2):
            vad.feed(self._loud())
        self.assertFalse(vad.ready.empty())

    def test_max_utterance_flush_resets_state(self):
        """_max_chunks チャンク目でフラッシュ → バッファ空・is_speaking=False になる。"""
        vad = main.AudioProcessor()
        for _ in range(vad._max_chunks):
            vad.feed(self._loud())
        # ちょうど _max_chunks 個でフラッシュ済み → 状態がリセットされている
        self.assertFalse(vad.ready.empty())
        self.assertEqual(len(vad._buf), 0)
        self.assertFalse(vad.is_speaking)

    # ── キュー満杯 → 無音ドロップ ──────────────────────────────────────────

    def test_queue_full_drops_utterance_silently(self):
        """ready キューが満杯のときフラッシュ失敗が無音で起きる。"""
        vad = main.AudioProcessor()
        for _ in range(vad.ready.maxsize):
            vad.ready.put(np.zeros(1))
        vad.feed(self._loud())
        for _ in range(vad._silence_need):
            vad.feed(self._silent())
        # 追加されていない（maxsize のまま）
        self.assertEqual(vad.ready.qsize(), vad.ready.maxsize)

    def test_pop_dropped_returns_true_after_drop(self):
        """キュー満杯でドロップが起きると pop_dropped() が True を返す。"""
        vad = main.AudioProcessor()
        for _ in range(vad.ready.maxsize):
            vad.ready.put(np.zeros(1))
        vad.feed(self._loud())
        for _ in range(vad._silence_need):
            vad.feed(self._silent())
        self.assertTrue(vad.pop_dropped())

    def test_pop_dropped_resets_after_call(self):
        """pop_dropped() は一度 True を返したら次は False になる。"""
        vad = main.AudioProcessor()
        for _ in range(vad.ready.maxsize):
            vad.ready.put(np.zeros(1))
        vad.feed(self._loud())
        for _ in range(vad._silence_need):
            vad.feed(self._silent())
        vad.pop_dropped()           # 1回目: True
        self.assertFalse(vad.pop_dropped())  # 2回目: False

    def test_pop_dropped_false_when_no_drop(self):
        """ドロップなしの状態では pop_dropped() は False。"""
        vad = main.AudioProcessor()
        vad.feed(self._loud())
        for _ in range(vad._silence_need):
            vad.feed(self._silent())
        self.assertFalse(vad.pop_dropped())

    # ── 複数発話 ────────────────────────────────────────────────────────────

    def test_three_sequential_utterances(self):
        vad = main.AudioProcessor()
        for _ in range(3):
            vad.feed(self._loud())
            for _ in range(vad._silence_need):
                vad.feed(self._silent())
        count = 0
        while not vad.ready.empty():
            vad.ready.get_nowait()
            count += 1
        self.assertEqual(count, 3)

    # ── reset ───────────────────────────────────────────────────────────────

    def test_reset_clears_everything(self):
        vad = main.AudioProcessor()
        vad.feed(self._loud())
        vad.ready.put(np.zeros(1))
        vad.reset()
        self.assertFalse(vad.is_speaking)
        self.assertEqual(len(vad._buf), 0)
        self.assertEqual(vad._silence, 0)
        self.assertTrue(vad.ready.empty())

    def test_reset_then_new_speech_works(self):
        vad = main.AudioProcessor()
        vad.feed(self._loud())
        vad.reset()
        vad.feed(self._loud())
        self.assertTrue(vad.is_speaking)

    # ── 閾値境界値 ──────────────────────────────────────────────────────────

    def test_exactly_at_threshold_is_silent(self):
        """RMS == SILENCE_THRESHOLD は「無音」（> でないため）。"""
        vad = main.AudioProcessor()
        chunk = np.full(main.CHUNK_SIZE, main.SILENCE_THRESHOLD, dtype=np.float32)
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        self.assertAlmostEqual(rms, main.SILENCE_THRESHOLD, places=6)
        vad.feed(chunk)
        self.assertFalse(vad.is_speaking)

    def test_just_above_threshold_is_speech(self):
        vad = main.AudioProcessor()
        val = main.SILENCE_THRESHOLD + 1e-4
        chunk = np.full(main.CHUNK_SIZE, val, dtype=np.float32)
        vad.feed(chunk)
        self.assertTrue(vad.is_speaking)

    # ── エッジケース ────────────────────────────────────────────────────────

    def test_single_sample_chunk_no_crash(self):
        vad = main.AudioProcessor()
        vad.feed(np.array([0.1], dtype=np.float32))

    def test_all_zeros_chunk_is_silent(self):
        vad = main.AudioProcessor()
        vad.feed(np.zeros(main.CHUNK_SIZE, dtype=np.float32))
        self.assertFalse(vad.is_speaking)

    def test_silence_before_any_speech_never_flushes(self):
        """発話前の無音が何チャンクあっても ready に積まれない。"""
        vad = main.AudioProcessor()
        for _ in range(vad._silence_need * 3):
            vad.feed(self._silent())
        self.assertTrue(vad.ready.empty())

    def test_empty_flush_no_queue_item(self):
        """バッファが空の状態で _flush() しても ready に積まれない。"""
        vad = main.AudioProcessor()
        vad._flush()
        self.assertTrue(vad.ready.empty())

    def test_int16_array_accepted(self):
        """int16 配列でも RMS 計算が破綻しない。"""
        vad = main.AudioProcessor()
        chunk = np.ones(main.CHUNK_SIZE, dtype=np.int16)
        vad.feed(chunk)   # RMS=1.0 >> threshold → speaking
        self.assertTrue(vad.is_speaking)


# ═══════════════════════════════════════════════════════════════════════════════
# 設定定数のサニティチェック
# ═══════════════════════════════════════════════════════════════════════════════
class TestConfigConstants(unittest.TestCase):

    def test_sample_rate_16k(self):
        self.assertEqual(main.SAMPLE_RATE, 16000)

    def test_chunk_size_derived_correctly(self):
        self.assertEqual(main.CHUNK_SIZE, int(main.SAMPLE_RATE * main.CHUNK_DURATION))

    def test_silence_need_derived_correctly(self):
        vad = main.AudioProcessor()
        self.assertEqual(vad._silence_need, int(main.SILENCE_DURATION / main.CHUNK_DURATION))

    def test_max_chunks_derived_correctly(self):
        vad = main.AudioProcessor()
        self.assertEqual(vad._max_chunks, int(main.MAX_UTTERANCE_SEC / main.CHUNK_DURATION))

    def test_silence_threshold_positive(self):
        self.assertGreater(main.SILENCE_THRESHOLD, 0)

    def test_max_utterance_sec_positive(self):
        self.assertGreater(main.MAX_UTTERANCE_SEC, 0)

    def test_ready_queue_is_bounded(self):
        vad = main.AudioProcessor()
        self.assertGreater(vad.ready.maxsize, 0)

    def test_silence_need_at_least_one(self):
        """_silence_need が 0 になるとフラッシュが無限に起きるため最低 1 必要。"""
        vad = main.AudioProcessor()
        self.assertGreaterEqual(vad._silence_need, 1)

    def test_it_prompt_not_empty(self):
        self.assertTrue(len(main.IT_PROMPT) > 0)

    def test_colors_dict_has_required_keys(self):
        required = {"bg", "surface", "surface2", "text", "subtext",
                    "green", "red", "yellow", "blue", "mauve"}
        self.assertTrue(required.issubset(main.COLORS.keys()))


# ═══════════════════════════════════════════════════════════════════════════════
# スレッド安全性の簡易検証
# ═══════════════════════════════════════════════════════════════════════════════
class TestAudioProcessorThreadSafety(unittest.TestCase):
    """
    AudioProcessor は複数スレッドから feed/reset が呼ばれる想定。
    CPython GIL のもとで基本的な操作が破綻しないことを確認する。
    """

    def test_concurrent_feed_does_not_crash(self):
        vad = main.AudioProcessor()
        loud = np.ones(main.CHUNK_SIZE, dtype=np.float32) * 0.1
        errors = []

        def feeder():
            try:
                for _ in range(50):
                    vad.feed(loud)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=feeder) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        self.assertEqual(errors, [], f"スレッドエラー: {errors}")

    def test_reset_during_feed_does_not_crash(self):
        vad = main.AudioProcessor()
        loud = np.ones(main.CHUNK_SIZE, dtype=np.float32) * 0.1
        errors = []

        def feeder():
            try:
                for _ in range(100):
                    vad.feed(loud)
            except Exception as e:
                errors.append(e)

        def resetter():
            try:
                for _ in range(10):
                    time.sleep(0.002)
                    vad.reset()
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=feeder)
        t2 = threading.Thread(target=resetter)
        t1.start(); t2.start()
        t1.join(timeout=5); t2.join(timeout=5)
        self.assertEqual(errors, [], f"スレッドエラー: {errors}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
