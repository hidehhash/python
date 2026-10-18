import argparse
from pathlib import Path

# Qwenを使ってTTSを実行

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel


MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

# このモデルで利用できるプリセット話者
SPEAKERS = [
    "Ono_Anna",
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Sohee",
]


def main():
    parser = argparse.ArgumentParser(
        description="テキストファイルをQwen3-TTSで音声化します"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="input.txt",
        help="読み上げるテキストファイル（既定: input.txt）",
    )
    parser.add_argument(
        "-s",
        "--speaker",
        choices=SPEAKERS,
        default="Ono_Anna",
        help="話者名（既定: Ono_Anna）",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="出力するWAVファイル名（省略時は入力名.wav）",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(
            f"入力ファイルが見つかりません: {input_path}"
        )

    # UTF-8のテキストを読み込む
    text = input_path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError("テキストファイルが空です。")

    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_suffix(".wav")
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda:0" else torch.float32

    print(f"入力ファイル: {input_path}")
    print(f"話者: {args.speaker}")
    print(f"実行デバイス: {device}")
    print("モデルを読み込んでいます...")

    model = Qwen3TTSModel.from_pretrained(
        MODEL,
        device_map=device,
        dtype=dtype,
    )

    print("音声を生成しています...")

    wavs, sr = model.generate_custom_voice(
        text=text,
        language="Japanese",
        speaker=args.speaker,
    )

    sf.write(str(output_path), wavs[0], sr)

    print(f"生成完了: {output_path.resolve()}")


if __name__ == "__main__":
    main()
