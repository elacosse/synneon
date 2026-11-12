"""Tests for apps/run_example.py — parsing utilities and config helpers."""

from __future__ import annotations

import csv
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from synneon.apps.run_example import (
    _resolve_path,
    extract_all_frames_as_ndarray,
    extract_video_frames,
    load_agent_settings,
    load_memory_config,
    load_neons_configs,
    load_stt_config,
    load_tts_config,
    parse_conversation_file,
    parse_csv_file,
)

# ---------------------------------------------------------------------------
# parse_conversation_file
# ---------------------------------------------------------------------------


class TestParseConversationFile:
    def test_basic_parsing(self, tmp_path):
        f = tmp_path / "conversation.txt"
        f.write_text("other: Hello there\nuser: G S Y T\nother: Sure thing.\n")
        entries = parse_conversation_file(f, user_id="u1", session_id="s1")
        assert len(entries) == 3
        assert entries[0].role == "other"
        assert entries[0].text == "Hello there"
        assert entries[1].role == "user"
        assert entries[1].text == "G S Y T"
        assert entries[2].role == "other"
        assert entries[2].text == "Sure thing."

    def test_blank_lines_skipped(self, tmp_path):
        f = tmp_path / "conversation.txt"
        f.write_text("\nother: Hi\n\nuser: Bye\n\n")
        entries = parse_conversation_file(f, user_id="u1", session_id="s1")
        assert len(entries) == 2

    def test_unknown_prefix_defaults_to_other(self, tmp_path):
        f = tmp_path / "conversation.txt"
        f.write_text("speaker3: Some text\n")
        entries = parse_conversation_file(f, user_id="u1", session_id="s1")
        assert entries[0].role == "other"
        assert entries[0].text == "speaker3: Some text"

    def test_user_id_and_session_id_attached(self, tmp_path):
        f = tmp_path / "conversation.txt"
        f.write_text("other: Hi\n")
        entries = parse_conversation_file(f, user_id="alice", session_id="sess-42")
        assert entries[0].user_id == "alice"
        assert entries[0].session_id == "sess-42"

    def test_empty_file_returns_empty_list(self, tmp_path):
        f = tmp_path / "conversation.txt"
        f.write_text("")
        entries = parse_conversation_file(f, user_id="u1", session_id="s1")
        assert entries == []


# ---------------------------------------------------------------------------
# parse_csv_file
# ---------------------------------------------------------------------------


def _make_csv(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "data.csv"
    if not rows:
        path.write_text("timestamp,gaze_x,gaze_y,gaze_worn,pupil_left,pupil_right,audio_rms\n")
        return path
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


class TestParseCsvFile:
    def _sample_rows(self, n: int = 10) -> list[dict]:
        return [
            {
                "timestamp": str(i * 0.033),
                "gaze_x": str(100 + i),
                "gaze_y": str(200 + i),
                "gaze_worn": "1",
                "pupil_left": str(3.5 + i * 0.01),
                "pupil_right": str(3.6 + i * 0.01),
                "audio_rms": "0.01",
            }
            for i in range(n)
        ]

    def test_gaze_and_pupil_extracted(self, tmp_path):
        path = _make_csv(tmp_path, self._sample_rows(5))
        gaze, pupil = parse_csv_file(path)
        assert len(gaze) == 5
        assert len(pupil) == 5
        ts, x, y = gaze[0]
        assert ts == pytest.approx(0.0)
        assert x == pytest.approx(100.0)
        assert y == pytest.approx(200.0)

    def test_rows_downsampled_to_max(self, tmp_path):
        path = _make_csv(tmp_path, self._sample_rows(1000))
        gaze, pupil = parse_csv_file(path, max_rows=200)
        assert len(gaze) <= 200
        assert len(pupil) <= 200

    def test_no_downsampling_when_max_rows_none(self, tmp_path):
        path = _make_csv(tmp_path, self._sample_rows(500))
        gaze, pupil = parse_csv_file(path, max_rows=None)
        assert len(gaze) == 500
        assert len(pupil) == 500

    def test_missing_column_raises(self, tmp_path):
        rows = [{"timestamp": "0.0", "gaze_x": "1.0", "gaze_y": "2.0"}]  # missing pupil cols
        path = _make_csv(tmp_path, rows)
        with pytest.raises(ValueError, match="missing required columns"):
            parse_csv_file(path)

    def test_empty_csv_returns_empty_lists(self, tmp_path):
        path = _make_csv(tmp_path, [])
        gaze, pupil = parse_csv_file(path)
        assert gaze == []
        assert pupil == []

    def test_pupil_values_correct(self, tmp_path):
        path = _make_csv(tmp_path, self._sample_rows(3))
        _, pupil = parse_csv_file(path)
        ts, left, right = pupil[0]
        assert left == pytest.approx(3.5)
        assert right == pytest.approx(3.6)


# ---------------------------------------------------------------------------
# extract_video_frames
# ---------------------------------------------------------------------------


class TestExtractVideoFrames:
    def _mock_cap(self, total_frames: int, fps: float, mock_cv2: MagicMock) -> MagicMock:
        import numpy as np

        cap = MagicMock()
        mock_cv2.CAP_PROP_FRAME_COUNT = 7
        cap.get.side_effect = lambda p: total_frames if p == 7 else fps
        cap.read.return_value = (True, np.zeros((10, 10, 3), dtype="uint8"))
        mock_cv2.VideoCapture.return_value = cap
        _, jpeg_buf = __import__("cv2").imencode(".jpg", np.zeros((10, 10, 3), dtype="uint8"))
        mock_cv2.imencode.return_value = (True, jpeg_buf)
        return cap

    def _fake_jpeg(self) -> bytes:
        import cv2
        import numpy as np

        _, buf = cv2.imencode(".jpg", np.zeros((10, 10, 3), dtype="uint8"))
        return buf.tobytes()

    def test_returns_list_of_tuples(self, tmp_path):
        import numpy as np

        video_path = tmp_path / "dummy.mp4"
        video_path.write_bytes(b"")
        jpeg = self._fake_jpeg()

        import cv2

        frame_count_prop = cv2.CAP_PROP_FRAME_COUNT

        with patch("cv2.VideoCapture") as mock_vc, patch("cv2.imencode") as mock_enc:
            cap = MagicMock()
            cap.get.side_effect = lambda p: 8 if p == frame_count_prop else 30.0
            cap.read.return_value = (True, np.zeros((10, 10, 3), dtype="uint8"))
            mock_vc.return_value = cap
            mock_enc.return_value = (True, MagicMock(tobytes=lambda: jpeg))

            frames = extract_video_frames(video_path, max_frames=4)

        assert len(frames) <= 4
        for ts, data in frames:
            assert isinstance(ts, float)
            assert isinstance(data, bytes)

    def test_at_most_max_frames_extracted(self, tmp_path):
        import numpy as np

        video_path = tmp_path / "dummy.mp4"
        video_path.write_bytes(b"")
        jpeg = self._fake_jpeg()

        import cv2

        frame_count_prop = cv2.CAP_PROP_FRAME_COUNT

        with patch("cv2.VideoCapture") as mock_vc, patch("cv2.imencode") as mock_enc:
            cap = MagicMock()
            cap.get.side_effect = lambda p: 100 if p == frame_count_prop else 30.0
            cap.read.return_value = (True, np.zeros((10, 10, 3), dtype="uint8"))
            mock_vc.return_value = cap
            mock_enc.return_value = (True, MagicMock(tobytes=lambda: jpeg))

            frames = extract_video_frames(video_path, max_frames=4)

        assert len(frames) == 4


# ---------------------------------------------------------------------------
# extract_all_frames_as_ndarray
# ---------------------------------------------------------------------------


class TestExtractAllFramesAsNdarray:
    def test_returns_list_of_ndarrays(self, tmp_path):
        import numpy as np

        video_path = tmp_path / "dummy.mp4"
        video_path.write_bytes(b"")
        fake_frame = np.zeros((10, 10, 3), dtype="uint8")

        call_count = 0

        def fake_read():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return (True, fake_frame)
            return (False, None)

        with patch("cv2.VideoCapture") as mock_vc:
            cap = MagicMock()
            cap.read.side_effect = fake_read
            mock_vc.return_value = cap

            frames = extract_all_frames_as_ndarray(video_path)

        assert len(frames) == 3
        for f in frames:
            assert isinstance(f, np.ndarray)

    def test_empty_video_returns_empty_list(self, tmp_path):
        video_path = tmp_path / "empty.mp4"
        video_path.write_bytes(b"")

        with patch("cv2.VideoCapture") as mock_vc:
            cap = MagicMock()
            cap.read.return_value = (False, None)
            mock_vc.return_value = cap

            frames = extract_all_frames_as_ndarray(video_path)

        assert frames == []


# ---------------------------------------------------------------------------
# _resolve_path
# ---------------------------------------------------------------------------


class TestResolvePath:
    def test_absolute_existing_path_returned_unchanged(self, tmp_path):
        f = tmp_path / "myconfig.toml"
        f.write_text("")
        result = _resolve_path(f)
        assert result == f

    def test_relative_path_existing_in_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = tmp_path / "relative.toml"
        f.write_text("")
        result = _resolve_path(Path("relative.toml"))
        assert result == Path("relative.toml")

    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _resolve_path(tmp_path / "nonexistent.toml")


# ---------------------------------------------------------------------------
# Config loaders
# ---------------------------------------------------------------------------


class TestLoadAgentSettings:
    def test_loads_profile(self, tmp_path):
        toml = tmp_path / "agent.toml"
        toml.write_text(
            textwrap.dedent("""\
            [agent.default]
            model = "google:gemini-3.8-flash"
            system_prompt = "You are helpful"
            prompt_template = "{tokens}"
            """)
        )
        settings = load_agent_settings(toml, "default")
        assert settings.model == "google:gemini-3.8-flash"
        assert settings.system_prompt == "You are helpful"

    def test_missing_profile_raises(self, tmp_path):
        toml = tmp_path / "agent.toml"
        toml.write_text("[agent.other]\nmodel = 'x'\nsystem_prompt = 'y'\nprompt_template = 'z'\n")
        with pytest.raises(ValueError, match="not found"):
            load_agent_settings(toml, "default")


class TestLoadMemoryConfig:
    def test_loads_profile(self, tmp_path):
        toml = tmp_path / "memory.toml"
        toml.write_text(
            textwrap.dedent("""\
            [memory.default]
            redis_url = "redis://myhost:6380/1"
            conversation_window = 30
            memory_window = 5
            conversation_ttl_days = 7
            user_id = "alice"
            """)
        )
        cfg = load_memory_config(toml, "default")
        assert cfg.redis_url == "redis://myhost:6380/1"
        assert cfg.user_id == "alice"

    def test_missing_profile_returns_defaults(self, tmp_path):
        toml = tmp_path / "memory.toml"
        toml.write_text("[memory.other]\nuser_id = 'x'\n")
        cfg = load_memory_config(toml, "default")
        assert cfg.redis_url == "redis://localhost:6379/0"


class TestLoadNeonConfigs:
    def test_loads_blink_fixation_saliency_recording(self, tmp_path):
        toml = tmp_path / "neons.toml"
        toml.write_text(
            textwrap.dedent("""\
            [blink]
            double_blink_delta = 0.5
            circle_annotation_radius = 80

            [fixation]
            algorithm = "I-DT"
            min_duration_ms = 150.0
            max_dispersion_pixels = 25.0

            [saliency]
            max_peaks = 3

            [recording]
            max_duration_s = 60.0
            """)
        )
        blink, fixation, saliency, recording = load_neons_configs(toml)
        assert blink.double_blink_delta == pytest.approx(0.5)
        assert fixation.min_duration_ms == pytest.approx(150.0)
        assert saliency.max_peaks == 3
        assert recording.max_duration_s == pytest.approx(60.0)


class TestLoadSttConfig:
    def test_loads_stt_section(self, tmp_path):
        toml = tmp_path / "stt.toml"
        toml.write_text(
            textwrap.dedent("""\
            [stt.default]
            provider = "google"
            model = "gemini-3.8-flash"
            prompt = "Transcribe verbatim"
            thinking = "low"
            """)
        )
        cfg = load_stt_config(toml, "default")
        assert cfg.provider == "google"
        assert cfg.model == "gemini-3.8-flash"
        assert cfg.prompt == "Transcribe verbatim"
        assert cfg.thinking == "low"

    def test_missing_profile_raises(self, tmp_path):
        toml = tmp_path / "stt.toml"
        toml.write_text("[stt.other]\nmodel = 'x'\n")
        with pytest.raises(ValueError, match="not found"):
            load_stt_config(toml, "default")


class TestLoadTtsConfig:
    def test_loads_tts_section(self, tmp_path):
        toml = tmp_path / "tts.toml"
        toml.write_text(
            textwrap.dedent("""\
            [tts.default]
            provider = "google"
            model = "gemini-3.8-flash-tts"
            voice = "Kore"
            """)
        )
        cfg = load_tts_config(toml, "default")
        assert cfg.provider == "google"
        assert cfg.model == "gemini-3.8-flash-tts"
        assert cfg.voice == "Kore"

    def test_missing_profile_raises(self, tmp_path):
        toml = tmp_path / "tts.toml"
        toml.write_text("[tts.other]\nmodel = 'x'\n")
        with pytest.raises(ValueError, match="not found"):
            load_tts_config(toml, "default")


# ---------------------------------------------------------------------------
# _run_pipeline (expand step with --save)
# ---------------------------------------------------------------------------


class TestRunPipelineExpandSave:
    async def test_expand_with_save_persists_context_and_record(self, tmp_path):
        """With --save, expand stores the last-expand context under the session and an InferenceRecord."""
        from unittest.mock import AsyncMock, create_autospec

        from synneon.apps.run_example import _run_pipeline
        from synneon.schemas.memory import InferenceType
        from synneon.services.memory.store import ContextStore

        (tmp_path / "conversation.txt").write_text("other: Hi, how are you?\n")
        (tmp_path / "abbreviation_for_inference.txt").write_text("G S Y T\n")

        # Autospec enforces the real ContextStore method signatures.
        store = create_autospec(ContextStore, instance=True)
        store.get_recent_conversation.return_value = []
        store.get_recent_memories.return_value = []
        store._redis = MagicMock(aclose=AsyncMock())

        agent = MagicMock()
        agent.builder.build_prompts.return_value = ("system prompt", "user prompt")
        agent.run = AsyncMock(return_value=MagicMock(output="1. Good, see you then\n2. Great, sounds yummy to"))
        agent.settings.model_dump.return_value = {"model": "openai:test"}

        with (
            patch("redis.asyncio.Redis.from_url", return_value=MagicMock()),
            patch("synneon.services.memory.store.ContextStore", return_value=store),
            patch("synneon.agents.base.CognitiveSteeringAgent", return_value=agent),
        ):
            await _run_pipeline(
                example=tmp_path,
                user="alice",
                interpreter="configs/interpreter/agent.toml",
                memory="configs/memory/default.toml",
                neons="configs/neons/default.toml",
                stt="configs/stt/default.toml",
                tts="configs/tts/default.toml",
                save=True,
                active_steps=["seed", "expand"],
                profile="default",
                neon_profile="image2context",
                frame_mode="middle",
                neon_scene="raw",
                verbose=False,
            )

        session_id = store.start_session.call_args.args[0]
        saved_session, saved_ctx = store.save_last_expand_context.call_args.args
        assert saved_session == session_id
        assert saved_ctx["tokens"] == "G S Y T"
        assert saved_ctx["system_prompt"] == "system prompt"
        assert saved_ctx["candidates"][0]["text"] == "Good, see you then"

        record = store.save_inference_record.call_args.args[0]
        assert record.inference_type == InferenceType.TEXT2TEXT
        assert record.session_id == session_id
        assert record.endpoint == "expand"


# ---------------------------------------------------------------------------
# extract_reference_frame_png
# ---------------------------------------------------------------------------


class TestExtractReferenceFramePng:
    @pytest.mark.parametrize(("mode", "expected"), [("first", 10), ("middle", 30), ("last", 40), ("bogus", 30)])
    def test_selects_same_frame_as_saliency_and_encodes_png(self, mode, expected):
        import cv2
        import numpy as np

        from synneon.apps.run_example import extract_reference_frame_png

        frames = [np.full((4, 4, 3), v, dtype=np.uint8) for v in (10, 20, 30, 40)]
        with patch("synneon.apps.run_example.extract_all_frames_as_ndarray", return_value=frames):
            png = extract_reference_frame_png(Path("video.mp4"), mode)

        assert png.startswith(b"\x89PNG")
        decoded = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert int(decoded[0, 0, 0]) == expected

    def test_returns_none_without_frames(self):
        from synneon.apps.run_example import extract_reference_frame_png

        with patch("synneon.apps.run_example.extract_all_frames_as_ndarray", return_value=[]):
            assert extract_reference_frame_png(Path("video.mp4")) is None


# ---------------------------------------------------------------------------
# _run_pipeline — neon scene image and neon → expand hand-off
# ---------------------------------------------------------------------------


class TestRunPipelineNeon:
    async def _run(self, tmp_path, steps, neon_scene="raw"):
        from unittest.mock import AsyncMock

        from synneon.apps.run_example import _run_pipeline
        from synneon.schemas.context import ParsedVisualContext, PragmaticContext
        from synneon.schemas.results import Image2ContextResult

        (tmp_path / "abbreviation_for_inference.txt").write_text("G S Y T\n")

        neon_output = Image2ContextResult(
            parsed_visual=ParsedVisualContext(focused_object="coffee cup", detected_objects=["laptop"]),
            pragmatic=PragmaticContext(physical_setting="cafe"),
        )
        neon_agent = MagicMock()
        neon_agent.run = AsyncMock(return_value=MagicMock(output=neon_output))

        expand_agent = MagicMock()
        expand_agent.builder.build_prompts.return_value = ("system prompt", "user prompt")
        expand_agent.run = AsyncMock(return_value=MagicMock(output="1. Good, see you then"))

        with (
            patch("synneon.agents.base.VLMNeon2ContextAgent", return_value=neon_agent),
            patch("synneon.agents.base.CognitiveSteeringAgent", return_value=expand_agent),
            patch("synneon.apps.run_example.extract_video_frames", return_value=[(0.0, b"jpeg")]),
            patch("synneon.apps.run_example.extract_reference_frame_png", return_value=b"\x89PNG-raw") as ref_frame,
        ):
            await _run_pipeline(
                example=tmp_path,
                user="alice",
                interpreter="configs/interpreter/agent.toml",
                memory="configs/memory/default.toml",
                neons="configs/neons/default.toml",
                stt="configs/stt/default.toml",
                tts="configs/tts/default.toml",
                save=False,
                active_steps=steps,
                profile="default",
                neon_profile="image2context",
                frame_mode="last",
                neon_scene=neon_scene,
                verbose=False,
            )
        return neon_agent, expand_agent, ref_frame

    async def test_raw_scene_attaches_the_reference_frame(self, tmp_path):
        (tmp_path / "video.mp4").write_bytes(b"stub")
        neon_agent, _, ref_frame = await self._run(tmp_path, ["neon"])

        payload = neon_agent.run.call_args.args[0]
        assert payload.scene_image == b"\x89PNG-raw"
        ref_frame.assert_called_once_with(tmp_path / "video.mp4", "last")

    async def test_peaks_scene_uses_saliency_peaks_png(self, tmp_path):
        (tmp_path / "video.mp4").write_bytes(b"stub")
        (tmp_path / "saliency_peaks.png").write_bytes(b"\x89PNG-peaks")
        neon_agent, _, ref_frame = await self._run(tmp_path, ["neon"], neon_scene="peaks")

        assert neon_agent.run.call_args.args[0].scene_image == b"\x89PNG-peaks"
        ref_frame.assert_not_called()

    async def test_expand_receives_neon_result_without_save(self, tmp_path):
        _, expand_agent, _ = await self._run(tmp_path, ["neon", "expand"])

        context = expand_agent.run.call_args.args[0]
        assert context.parsed_visual.focused_object == "coffee cup"
        assert context.pragmatic.physical_setting == "cafe"

    async def test_expand_alone_uses_neutral_context(self, tmp_path):
        from synneon.schemas.context import SteeringContext

        _, expand_agent, _ = await self._run(tmp_path, ["expand"])

        assert expand_agent.run.call_args.args[0] == SteeringContext()
