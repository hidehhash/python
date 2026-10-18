from pathlib import Path

# 音声ファイルを複数生成するためのスクリプト

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module=r"librosa\.")

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import librosa

import concat_wav

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

# 原稿ファイル、話者、出力ファイル名
jobs = [
    ("i1.txt", "Ono_Anna", "i1.wav"),
    ("i2.txt", "Eric", "i2.wav"),
    ("i3.txt", "Ono_Anna", "i3.wav"),
    ("i4.txt", "Eric", "i4.wav"),
    ("i5.txt", "Ono_Anna", "i5.wav"),
]

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda:0" else torch.float32

print(f"実行デバイス: {device}")
print("モデルを読み込んでいます...")

# モデルの読み込みは最初の一度だけ
model = Qwen3TTSModel.from_pretrained(
    MODEL_NAME,
    device_map=device,
    dtype=dtype,
)

for input_name, speaker, output_name in jobs:
    input_path = BASE_DIR / input_name
    output_path = BASE_DIR / output_name

    print(f"\n生成中: {input_name} / {speaker}")

    try:
        text = input_path.read_text(encoding="utf-8").strip()

        if not text:
            print(f"スキップ: {input_name} は空です")
            continue

        wavs, sr = model.generate_custom_voice(
            text=text,
            language="Japanese",
            speaker=speaker,
        )
        audio = wavs[0]

        # 全話者共通の速度設定
        speed_rate = 1.15  # 1.15 = 約15%速くする

        # Ono_Annaだけ音量を上げる
        if speaker == "Ono_Anna":
            audio = audio * 1.3
        # 音割れを防ぐため、範囲を制限
        audio = audio.clip(-1.0, 1.0)

        # 音程をなるべく変えずに速度を上げる
        '''
        audio = librosa.effects.time_stretch(
            audio,
            rate=speed_rate,
        )
        '''

        # WAVファイルとして保存
        sf.write(str(output_path), audio, sr)
        print(f"完了: {output_name}")

    except Exception as e:
        print(f"エラー: {input_name} / {speaker}: {e}")

# 最後にconcat_wavを実行する
concat_wav.main()

print("\n全ジョブの処理が終了しました。")
