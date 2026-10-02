"""Streaming-TTS arm gate regression (#84046, macOS): ``_chat_setup_turn_audio`` gates
streaming TTS on ``sounddevice`` import, which raises on macOS (PortAudio/CoreAudio init
triggers a kTCCServiceMediaLibrary prompt). The swallow-``except`` then disables streaming
TTS even though the speaker path deliberately avoids sounddevice on Darwin (it routes
output through tempfile -> afplay; tts_tool_speaker._device_usable returns False there).

The gate must not probe sounddevice: enablement is ``check_tts_requirements()`` alone, and
on Darwin (or anywhere sounddevice is absent) a working provider still arms streaming TTS.

Behavior contracts, mocked per test; no real audio, no config on disk.
"""

import sys
import threading
import time
import types
from pathlib import Path

import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cli import HermesCLI, _ChatTurn  # noqa: E402


def _speaker_patch_targets():
    """The mixins import ``stream_tts_to_speaker`` inside the function body, so patching
    the speaker module attribute intercepts the late import."""
    import tools.tts_tool_speaker as speaker_mod
    return speaker_mod


class _StreamingTTSArming:
    """Drive ``HermesCLI._chat_setup_turn_audio`` without a real CLI instance."""

    def make_cli(self):
        cli = HermesCLI.__new__(HermesCLI)
        cli._voice_mode = False
        cli._voice_continuous = False
        cli._voice_tts = True
        cli._voice_tts_done = threading.Event()
        cli._voice_last_tts_text = ""
        cli.show_timestamps = False
        cli.streaming_enabled = False
        cli.timestamp_format = "%H:%M"
        return cli

    def make_turn(self):
        turn = _ChatTurn()
        turn.mute_notification_reply = False
        return turn


def _probe_darwin_streaming_tts(monkeypatch, sounddevice_raises: bool):
    """Return (armed, tts_requirements_called) after arming a voice-tts turn.

    ``sounddevice`` import is made to raise (as it does on macOS) when
    *sounddevice_raises*; the tts requirements probe is stubbed to True (a working
    provider — the issue's Edge TTS setup); platform reports Darwin.
    """
    cli = _StreamingTTSArming().make_cli()
    turn = _StreamingTTSArming().make_turn()

    def _boom():
        raise OSError("PaAlsa [ CoreAudio TCC media-library prompt ]")

    import tools.tts_tool as tts_tool
    monkeypatch.setattr(tts_tool, "_import_sounddevice", _boom)
    monkeypatch.setattr(tts_tool, "check_tts_requirements", lambda: True)

    speaker_calls = []
    speaker_mod = _speaker_patch_targets()
    started = threading.Event()

    def _fake_stream(text_queue, stop_event, done_event, *a, **k):
        speaker_calls.append(True)
        done_event.set()
        started.set()

    monkeypatch.setattr(speaker_mod, "stream_tts_to_speaker", _fake_stream)

    monkeypatch.setattr("platform.system", lambda: "Darwin")

    cli._chat_setup_turn_audio(turn, "hello", False)
    # The worker thread sets stop_event / text_queue synchronously when armed; the
    # speaker thread itself is only a side effect we wait on briefly.
    if turn.use_streaming_tts:
        assert turn.text_queue is not None
        assert turn.stop_event is not None
        turn.text_queue.put(None)  # drain sentinel so the fake speaker exits
        started.wait(timeout=2)
    return turn.use_streaming_tts, bool(speaker_calls)


def test_streaming_tts_arms_when_sounddevice_raises_on_darwin(monkeypatch):
    """#84046: on macOS sounddevice import raises, but the speaker path needs no
    sounddevice (tempfile -> afplay). Streaming TTS must still arm."""
    armed, spoke = _probe_darwin_streaming_tts(monkeypatch, sounddevice_raises=True)
    assert armed is True, "streaming TTS must arm without sounddevice (macOS path)"
    assert spoke is True, "the TTS speaker thread must have been started"


def test_streaming_tts_arms_with_no_sounddevice_anywhere(monkeypatch):
    """The arm gate must not import sounddevice at all: even where the import never
    raises, wiring must come solely from check_tts_requirements() + speaker policy."""
    import tools.tts_tool as tts_tool

    probe = []
    monkeypatch.setattr(tts_tool, "_import_sounddevice",
                        lambda: probe.append(1) or types.ModuleType("sounddevice"))
    monkeypatch.setattr(tts_tool, "check_tts_requirements", lambda: True)
    speaker_mod = _speaker_patch_targets()
    started = threading.Event()

    def _fake_stream(text_queue, stop_event, done_event, *a, **k):
        done_event.set()
        started.set()

    monkeypatch.setattr(speaker_mod, "stream_tts_to_speaker", _fake_stream)

    cli = _StreamingTTSArming().make_cli()
    turn = _StreamingTTSArming().make_turn()
    monkeypatch.setattr("platform.system", lambda: "Linux")

    cli._chat_setup_turn_audio(turn, "hello", False)
    assert turn.use_streaming_tts is True
    assert probe == [], "the arm gate must not probe sounddevice"
    turn.text_queue.put(None)
    started.wait(timeout=2)


def test_streaming_tts_stays_off_when_provider_unavailable(monkeypatch):
    """Arming remains honest: with the provider unavailable the turn stays unarmed
    (fallback to whole-response TTS), whether or not sounddevice imports."""
    import tools.tts_tool as tts_tool

    monkeypatch.setattr(tts_tool, "check_tts_requirements", lambda: False)
    probe = []
    monkeypatch.setattr(tts_tool, "_import_sounddevice",
                        lambda: probe.append(1) or types.ModuleType("sounddevice"))

    cli = _StreamingTTSArming().make_cli()
    turn = _StreamingTTSArming().make_turn()
    monkeypatch.setattr("platform.system", lambda: "Darwin")

    cli._chat_setup_turn_audio(turn, "hello", False)
    assert turn.use_streaming_tts is False
    assert turn.text_queue is None
    assert probe == [], "no sounddevice probe on the unarmed path either"


def test_streaming_tts_not_armed_without_voice_tts(monkeypatch):
    """Non-voice-tts turns are untouched: no probe, no queue, no thread."""
    import tools.tts_tool as tts_tool

    probe = []
    monkeypatch.setattr(tts_tool, "_import_sounddevice",
                        lambda: probe.append(1) or types.ModuleType("sounddevice"))

    cli = _StreamingTTSArming().make_cli()
    cli._voice_tts = False
    turn = _StreamingTTSArming().make_turn()
    monkeypatch.setattr("platform.system", lambda: "Darwin")

    cli._chat_setup_turn_audio(turn, "hello", False)
    assert turn.use_streaming_tts is False
    assert turn.text_queue is None
    assert probe == []
