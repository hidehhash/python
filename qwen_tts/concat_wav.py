from pathlib import Path

import numpy as np
import soundfile as sf

# 生成した音声ファイルをConcat（batch_tts.pyの中から呼び出される）


BASE_DIR = Path(__file__).resolve().parent

# 連結するファイルを、希望する順番で指定
INPUT_FILES = [
    "i1.wav",
    "i2.wav",
    "i3.wav",
    "i4.wav",
    "i5.wav",
]

# 連結後のファイル名
OUTPUT_FILE = "combined.wav"

# 出力する音声のサンプリングレート
OUTPUT_SR = 24000


def convert_channels(data, channels):
    """モノラル／ステレオを統一する"""
    if channels == 1:
        # ステレオなら左右の平均を取ってモノラル化
        if data.ndim == 2:
            return data.mean(axis=1)
        return data

    # ステレオ出力に統一
    if data.ndim == 1:
        return np.column_stack((data, data))

    if data.shape[1] == 1:
        return np.repeat(data, 2, axis=1)

    return data[:, :2]


def main():
    audio_parts = []

    for filename in INPUT_FILES:
        path = BASE_DIR / filename

        if not path.is_file():
            print(f"ファイルが見つかりません: {filename}")
            continue

        print(f"読み込み中: {filename}")

        data, sr = sf.read(path, dtype="float32")

        # サンプリングレートが違う場合はエラーにして知らせる
        # ※必要なら後でリサンプリング処理を追加できます
        if sr != OUTPUT_SR:
            raise ValueError(
                f"{filename} のサンプリングレートは {sr} Hz です。"
                f"{OUTPUT_SR} Hz ではありません。"
            )

        data = convert_channels(data, channels=1)
        audio_parts.append(data)

    if not audio_parts:
        print("連結できるファイルがありません。")
        return

    # すべてモノラル・同一サンプリングレートなので連結可能
    combined = np.concatenate(audio_parts)

    output_path = BASE_DIR / OUTPUT_FILE

    sf.write(
        output_path,
        combined,
        OUTPUT_SR,
        subtype="PCM_16",
    )

    duration = len(combined) / OUTPUT_SR

    print(f"\n連結完了: {output_path}")
    print(f"音声の長さ: {duration:.1f} 秒")


if __name__ == "__main__":
    main()
