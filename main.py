# Batch transcribe audio samples and write results to the benchmark results folder.
# Azat (@azataiot) 2026-02-16

import sys
import tempfile
from pathlib import Path

import librosa
import soundfile as sf

from data import featurelen, sample_rate, hop_len
from UFormer import UFormer

SAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "samples"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "gheyret-asr"

# Model supports max 2500 frames. At sr=22050 and hop_len=200, that's ~22s.
# Use 20s chunks with 1s overlap to stay safe.
CHUNK_DURATION = 20.0
OVERLAP = 1.0


def transcribe_chunked(model, audio_path, device):
    """Transcribe audio, chunking if longer than the model's max input length."""
    audio, sr = librosa.load(audio_path, sr=sample_rate, res_type="polyphase")
    duration = len(audio) / sr

    if duration <= CHUNK_DURATION + OVERLAP:
        return model.predict(audio_path, device)

    # Split into overlapping chunks and transcribe each
    chunk_samples = int(CHUNK_DURATION * sr)
    step_samples = int((CHUNK_DURATION - OVERLAP) * sr)
    parts = []

    for start in range(0, len(audio), step_samples):
        chunk = audio[start : start + chunk_samples]
        if len(chunk) < sr:  # skip chunks shorter than 1 second
            break

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, chunk, sr)
            txt = model.predict(tmp.name, device)
            parts.append(txt)

    return " ".join(parts)


def main():
    device = "cpu"
    model = UFormer(featurelen)
    model.to(device)

    samples = sorted(SAMPLES_DIR.glob("*.mp3"))
    if not samples:
        print(f"No .mp3 files found in {SAMPLES_DIR}")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for audio_path in samples:
        print(f"\nRecognizing: {audio_path.name}")
        try:
            txt = transcribe_chunked(model, str(audio_path), device)
            print(f"  -> {txt}")

            result_path = RESULTS_DIR / f"{audio_path.stem}.txt"
            result_path.write_text(txt + "\n", encoding="utf-8")
            print(f"  Saved to: {result_path}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone. {len(samples)} file(s) processed.")


if __name__ == "__main__":
    main()
