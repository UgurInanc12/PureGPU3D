"""Unit tests for desktop-worker JSON-lines communication protocol."""

import unittest
from pathlib import Path

from puregpu3d.runtime.protocol import (
    MessageType,
    PROTOCOL_VERSION,
    Stage,
    WorkerCommand,
    make_cancelled_msg,
    make_completed_msg,
    make_conversion_progress_msg,
    make_download_progress_msg,
    make_error_msg,
    make_status_msg,
    parse_protocol_line,
    serialize_protocol_line,
)


class TestProtocol(unittest.TestCase):
    """Test protocol dataclasses, serialization, and deserialization."""

    def test_worker_command_roundtrip(self) -> None:
        cmd = WorkerCommand(
            job_id="job123",
            input_path="C:/path/to/in.mp4",
            output_path="C:/path/to/out.mp4",
            model_id="DA3-SMALL",
            device="cuda:0",
            disparity_strength=0.045,
            q_screen=0.55,
            overwrite=True,
            acknowledge_license=True,
            cancel_file="C:/temp/cancel.tmp",
            ffmpeg_path="C:/bin/ffmpeg.exe",
            ffprobe_path="C:/bin/ffprobe.exe",
        )

        json_str = cmd.to_json()
        restored = WorkerCommand.from_json(json_str)

        self.assertEqual(restored.job_id, "job123")
        self.assertEqual(restored.input_path, "C:/path/to/in.mp4")
        self.assertEqual(restored.output_path, "C:/path/to/out.mp4")
        self.assertEqual(restored.model_id, "DA3-SMALL")
        self.assertEqual(restored.device, "cuda:0")
        self.assertAlmostEqual(restored.disparity_strength, 0.045)
        self.assertAlmostEqual(restored.q_screen, 0.55)
        self.assertTrue(restored.overwrite)
        self.assertTrue(restored.acknowledge_license)
        self.assertEqual(restored.cancel_file, "C:/temp/cancel.tmp")
        self.assertEqual(restored.ffmpeg_path, "C:/bin/ffmpeg.exe")
        self.assertEqual(restored.ffprobe_path, "C:/bin/ffprobe.exe")
        self.assertEqual(restored.protocol_version, PROTOCOL_VERSION)

    def test_status_message(self) -> None:
        msg = make_status_msg("job1", Stage.DOWNLOADING, "Downloading weights...")
        line = serialize_protocol_line(msg)
        parsed = parse_protocol_line(line)

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["type"], MessageType.STATUS)
        self.assertEqual(parsed["job_id"], "job1")
        self.assertEqual(parsed["stage"], Stage.DOWNLOADING)
        self.assertEqual(parsed["message"], "Downloading weights...")
        self.assertIn("timestamp", parsed)

    def test_download_progress_message(self) -> None:
        msg = make_download_progress_msg("job1", 5000, 10000, 50.0, "model.safetensors")
        line = serialize_protocol_line(msg)
        parsed = parse_protocol_line(line)

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["type"], MessageType.DOWNLOAD_PROGRESS)
        self.assertEqual(parsed["downloaded_bytes"], 5000)
        self.assertEqual(parsed["total_bytes"], 10000)
        self.assertEqual(parsed["percent"], 50.0)
        self.assertEqual(parsed["filename"], "model.safetensors")

    def test_conversion_progress_message(self) -> None:
        msg = make_conversion_progress_msg("job1", 12, 24, 50.0, fps=8.5, eta_seconds=1.4)
        line = serialize_protocol_line(msg)
        parsed = parse_protocol_line(line)

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["type"], MessageType.CONVERSION_PROGRESS)
        self.assertEqual(parsed["current_frame"], 12)
        self.assertEqual(parsed["total_frames"], 24)
        self.assertEqual(parsed["percent"], 50.0)
        self.assertEqual(parsed["fps"], 8.5)
        self.assertEqual(parsed["eta_seconds"], 1.4)

    def test_completed_message(self) -> None:
        res = {"frames": 24, "output": "out.mp4"}
        msg = make_completed_msg("job1", res)
        line = serialize_protocol_line(msg)
        parsed = parse_protocol_line(line)

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["type"], MessageType.COMPLETED)
        self.assertEqual(parsed["result"], res)

    def test_error_message(self) -> None:
        msg = make_error_msg("job1", "Out of memory", stage=Stage.CONVERTING, detail="Traceback details")
        line = serialize_protocol_line(msg)
        parsed = parse_protocol_line(line)

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["type"], MessageType.ERROR)
        self.assertEqual(parsed["error"], "Out of memory")
        self.assertEqual(parsed["stage"], Stage.CONVERTING)
        self.assertEqual(parsed["detail"], "Traceback details")

    def test_cancelled_message(self) -> None:
        msg = make_cancelled_msg("job1", stage=Stage.CONVERTING, message="User cancelled")
        line = serialize_protocol_line(msg)
        parsed = parse_protocol_line(line)

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["type"], MessageType.CANCELLED)
        self.assertEqual(parsed["stage"], Stage.CONVERTING)
        self.assertEqual(parsed["message"], "User cancelled")

    def test_parse_invalid_lines(self) -> None:
        self.assertIsNone(parse_protocol_line(""))
        self.assertIsNone(parse_protocol_line("   \n"))
        self.assertIsNone(parse_protocol_line("not json"))
        self.assertIsNone(parse_protocol_line("[1, 2, 3]"))
        self.assertIsNone(parse_protocol_line('{"no_type": 123}'))


if __name__ == "__main__":
    unittest.main()
