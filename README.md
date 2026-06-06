# audio-transcriber

GPU-friendly CLI for transcribing audio with `faster-whisper`.

The tool reads audio directly through faster-whisper/PyAV. It does **not** require batching with a system `ffmpeg` binary.

## Quick start

```powershell
cd C:\Share\repos\audio-transcriber
uv sync --group cuda-required
uv run audio-transcriber .\test\"Вертикальная фрактура ГО.m4a" --format txt,srt,json,md
```

Or without installing the project script:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m audio_transcriber.cli .\test\"Вертикальная фрактура ГО.m4a" --format txt,srt,json,md
```

## Examples

Smoke-test only the first minute:

```powershell
uv run audio-transcriber .\inbox\audio.mp3 --clip-timestamps 0,60 --format txt,json --output-dir .\out
```

Full Russian transcription on CUDA if available:

```powershell
uv run audio-transcriber .\inbox\audio.mp3 --language ru --model large-v3 --format txt,srt,json,md --output-dir .\out
```

Bias domain vocabulary:

```powershell
uv run audio-transcriber .\inbox\lecture.m4a --initial-prompt "Стоматология. Вертикальная фрактура корня зуба. Эндодонтия. Ортодонтия." --format txt,srt,json
```

## Useful options

- `--device auto|cuda|cpu` — default: `auto`
- `--compute-type auto|float16|int8|...` — default: `auto`; uses `float16` on CUDA and `int8` on CPU
- `--model large-v3|large-v3-turbo|...` — default: `large-v3`
- `--language ru` — default: `ru`; pass empty string for auto-detect
- `--clip-timestamps 0,60` — transcribe only a time range for smoke tests, without external ffmpeg
- `--format txt,srt,json,md` — output formats
