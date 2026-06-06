from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence, TYPE_CHECKING

try:
    import torch
except Exception:  # pragma: no cover - torch can be optional for CPU-only envs
    torch = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from faster_whisper.transcribe import Segment

LOGGER = logging.getLogger("audio_transcriber")

SUPPORTED_FORMATS = {"txt", "srt", "json", "md"}


@dataclass(slots=True)
class TranscribedSegment:
    index: int
    start: float
    end: float
    text: str


def configure_stdio() -> None:
    """Prefer UTF-8 output on Windows SSH sessions."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def cuda_available() -> bool:
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if cuda_available() else "cpu"


def resolve_compute_type(compute_type: str, device: str) -> str:
    if compute_type != "auto":
        return compute_type
    return "float16" if device == "cuda" else "int8"


def timestamp_srt(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def timestamp_text(seconds: float) -> str:
    whole = int(seconds)
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def normalize_formats(values: Sequence[str]) -> list[str]:
    formats: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip().lower().lstrip(".")
            if not item:
                continue
            if item not in SUPPORTED_FORMATS:
                raise ValueError(
                    f"Unsupported output format {item!r}; choose from {', '.join(sorted(SUPPORTED_FORMATS))}"
                )
            if item not in formats:
                formats.append(item)
    return formats or ["txt"]


def output_stem(input_path: Path, output_dir: Path, output_name: str | None) -> Path:
    stem = output_name or input_path.stem
    return output_dir / stem


def write_txt(path: Path, segments: Iterable[TranscribedSegment], timestamps: bool) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            if timestamps:
                handle.write(f"[{timestamp_text(segment.start)}–{timestamp_text(segment.end)}] {text}\n")
            else:
                handle.write(f"{text}\n")


def write_srt(path: Path, segments: Iterable[TranscribedSegment]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            handle.write(f"{segment.index}\n")
            handle.write(f"{timestamp_srt(segment.start)} --> {timestamp_srt(segment.end)}\n")
            handle.write(f"{text}\n\n")


def write_json(path: Path, segments: list[TranscribedSegment], metadata: dict[str, object]) -> None:
    payload = {
        "metadata": metadata,
        "segments": [asdict(segment) for segment in segments],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_md(path: Path, segments: Iterable[TranscribedSegment], metadata: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Transcript\n\n")
        handle.write("## Metadata\n\n")
        for key, value in metadata.items():
            handle.write(f"- **{key}**: `{value}`\n")
        handle.write("\n## Segments\n\n")
        for segment in segments:
            text = segment.text.strip()
            if text:
                handle.write(f"- `[{timestamp_text(segment.start)}–{timestamp_text(segment.end)}]` {text}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audio-transcriber",
        description="Transcribe audio with faster-whisper, preferring CUDA when available.",
    )
    parser.add_argument("input", type=Path, help="Input audio path (.m4a/.mp3/.wav/etc.; read directly by faster-whisper/PyAV).")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("out"), help="Directory for output files. Default: out")
    parser.add_argument("--output-name", help="Output filename stem. Default: input filename stem")
    parser.add_argument("--format", "--formats", action="append", default=["txt"], help="Output formats: txt,srt,json,md. Can be comma-separated or repeated. Default: txt")
    parser.add_argument("--model", default="large-v3", help="Whisper model name. Default: large-v3")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device. Default: auto")
    parser.add_argument("--compute-type", default="auto", help="CTranslate2 compute type. Default: auto (float16 on CUDA, int8 on CPU)")
    parser.add_argument("--language", default="ru", help="Language code, or empty string for auto-detect. Default: ru")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"], help="Whisper task. Default: transcribe")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size. Default: 5")
    parser.add_argument("--best-of", type=int, default=5, help="Best-of candidates. Default: 5")
    parser.add_argument("--vad-filter", action=argparse.BooleanOptionalAction, default=True, help="Enable Silero VAD filtering. Default: enabled")
    parser.add_argument("--condition-on-previous-text", action=argparse.BooleanOptionalAction, default=True, help="Condition each segment on previous text. Default: enabled")
    parser.add_argument("--word-timestamps", action="store_true", help="Ask faster-whisper for word timestamps in model inference. Outputs remain segment-level.")
    parser.add_argument("--clip-timestamps", help="Optional faster-whisper clip timestamps, e.g. '0,60' for smoke tests. Does not require ffmpeg.")
    parser.add_argument("--initial-prompt", help="Optional prompt/context to bias recognition vocabulary.")
    parser.add_argument("--hotwords", help="Optional hotwords string if supported by installed faster-whisper.")
    parser.add_argument("--no-txt-timestamps", action="store_true", help="Do not prefix txt lines with [HH:MM:SS–HH:MM:SS].")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def segment_to_result(index: int, segment: "Segment") -> TranscribedSegment:
    return TranscribedSegment(
        index=index,
        start=float(segment.start),
        end=float(segment.end),
        text=segment.text.strip(),
    )


def transcribe(args: argparse.Namespace) -> tuple[list[TranscribedSegment], dict[str, object]]:
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    device = resolve_device(args.device)
    compute_type = resolve_compute_type(args.compute_type, device)
    language = args.language.strip() or None

    LOGGER.info("Input: %s", input_path)
    LOGGER.info("Model: %s; device=%s; compute_type=%s", args.model, device, compute_type)
    LOGGER.info("Loading faster-whisper model...")
    from faster_whisper import WhisperModel

    started = time.time()
    model = WhisperModel(args.model, device=device, compute_type=compute_type)
    loaded_at = time.time()
    LOGGER.info("Model loaded in %.2fs", loaded_at - started)

    kwargs: dict[str, object] = {
        "language": language,
        "task": args.task,
        "vad_filter": args.vad_filter,
        "beam_size": args.beam_size,
        "best_of": args.best_of,
        "condition_on_previous_text": args.condition_on_previous_text,
        "word_timestamps": args.word_timestamps,
    }
    if args.clip_timestamps:
        kwargs["clip_timestamps"] = args.clip_timestamps
    if args.initial_prompt:
        kwargs["initial_prompt"] = args.initial_prompt
    if args.hotwords:
        kwargs["hotwords"] = args.hotwords

    LOGGER.info("Transcribing...")
    raw_segments, info = model.transcribe(str(input_path), **kwargs)

    segments: list[TranscribedSegment] = []
    for index, raw_segment in enumerate(raw_segments, start=1):
        result = segment_to_result(index, raw_segment)
        segments.append(result)
        if index % 25 == 0:
            LOGGER.info("Segments: %s; through %s", index, timestamp_text(result.end))

    finished = time.time()
    metadata: dict[str, object] = {
        "input": str(input_path),
        "model": args.model,
        "device": device,
        "compute_type": compute_type,
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
        "segments": len(segments),
        "model_load_seconds": round(loaded_at - started, 3),
        "transcribe_seconds": round(finished - loaded_at, 3),
        "total_seconds": round(finished - started, 3),
        "clip_timestamps": args.clip_timestamps or "",
    }
    LOGGER.info("Done: %s segments in %.2fs", len(segments), finished - started)
    return segments, metadata


def write_outputs(args: argparse.Namespace, segments: list[TranscribedSegment], metadata: dict[str, object]) -> list[Path]:
    formats = normalize_formats(args.format)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(args.input, output_dir, args.output_name)

    written: list[Path] = []
    if "txt" in formats:
        path = stem.with_suffix(".txt")
        write_txt(path, segments, timestamps=not args.no_txt_timestamps)
        written.append(path)
    if "srt" in formats:
        path = stem.with_suffix(".srt")
        write_srt(path, segments)
        written.append(path)
    if "json" in formats:
        path = stem.with_suffix(".json")
        write_json(path, segments, metadata)
        written.append(path)
    if "md" in formats:
        path = stem.with_suffix(".md")
        write_md(path, segments, metadata)
        written.append(path)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    try:
        segments, metadata = transcribe(args)
        written = write_outputs(args, segments, metadata)
    except Exception as exc:
        LOGGER.exception("Transcription failed: %s", exc)
        return 1

    print(json.dumps({"ok": True, "metadata": metadata, "outputs": [str(path) for path in written]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
