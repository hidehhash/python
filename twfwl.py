import time
from datetime import datetime
from faster_whisper import WhisperModel
import subprocess
import json
from pathlib import Path

# Transcription with faster_whisper_large3

# ====== 設定項目（環境に合わせて変更） ======
# 事前配布したモデルフォルダのローカルパスを指定
MODEL_DIR = r"C:\python\_fwl3\models"  # インストール済み
# 文字起こし対象の音声ファイルパス
AUDIO_FILE = "b4.mp4"

# 実行デバイスと精度（GPUを使うなら下記を有効化）
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

# 文字起こしオプション
BEAM_SIZE = 5
USE_VAD = True           # 無音区間の除外（推奨）
WITHOUT_TIMESTAMPS = False  # タイムスタンプ不要なら True
WORD_TIMESTAMPS = False     # 単語単位のタイムスタンプが必要なら True

# 文字起こし後のテキストファイルパス。main()内で設定。
OUTPUT_TEXT_FILE = ""

# MP4の長さを取得する関数（ffprobe）
def get_video_duration(video_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])

# 文字起こし後のテキストファイルパス
def get_output_text_file():
    # Pathオブジェクトに変換
    p = Path(AUDIO_FILE)

    # 拡張子を .txt に変更
    return p.with_suffix(".txt")

# float秒をhhmmssに変換
def sec_to_hhmmss(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"

def main():
    # ---------- 全体計測開始 ----------
    t_all_start = time.time()
    ts_all_start = datetime.now()

    print("=== faster-whisper 実行開始 ===")
    print(f"開始時刻: {ts_all_start.strftime('%Y-%m-%d %H:%M:%S')}")

    # ---------- モデルロード時間を計測 ----------
    t_load_start = time.time()
    print(f"\n[LOAD] モデル読み込み: {MODEL_DIR}")
    model = WhisperModel(MODEL_DIR, device=DEVICE, compute_type=COMPUTE_TYPE)
    t_load_end = time.time()
    load_sec = t_load_end - t_load_start
    print(f"[LOAD] 完了（{load_sec:.2f} 秒）")

    # ---------- 文字起こし時間を計測 ----------
    t_trans_start = time.time()
    duration_sec = get_video_duration(AUDIO_FILE)
    print(f"\n[TRANSCRIBE] 開始: ファイル={AUDIO_FILE}, 動画の長さ {int(duration_sec // 60)}分{int(duration_sec % 60)}秒")

    segments, info = model.transcribe(
        AUDIO_FILE,
        beam_size=BEAM_SIZE,
        vad_filter=USE_VAD,
        without_timestamps=WITHOUT_TIMESTAMPS,
        word_timestamps=WORD_TIMESTAMPS,
        # 必要に応じて language="ja", multilingual=True, initial_prompt="..." などを追加
    )
    t_trans_end = time.time()
    trans_sec = t_trans_end - t_trans_start
    print(f"[TRANSCRIBE] 完了（{trans_sec:.2f} 秒）")

    # ---------- 結果の表示（セグメント） ----------
    print("\n=== 文字起こし結果 ===")
    print(f"Detected language: {info.language} (prob={info.language_probability:.3f})")

    # ファイル保存
    # segments はジェネレータなので、反復しながら出力
    seg_count = 0
    OUTPUT_TEXT_FILE = get_output_text_file()
    print(f"[OUTPUT] 保存： {OUTPUT_TEXT_FILE}")
    with open(OUTPUT_TEXT_FILE, "w", encoding="utf-8") as f:
        for segment in segments:
            seg_count += 1
            start = segment.start
            end = segment.end
            text = segment.text.strip()

            start_str = sec_to_hhmmss(start)
            end_str = sec_to_hhmmss(end)

            f.write(f"[{start_str} - {end_str}] {text}\n")
            print(f" セグメント数: {seg_count}")

    # ---------- 総時間の表示 ----------
    t_all_end = time.time()
    ts_all_end = datetime.now()
    all_sec = t_all_end - t_all_start

    print("\n=== 実行時間サマリ ===")
    print(f"終了時刻: {ts_all_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"モデルロード時間:   {load_sec:.2f} 秒")
    print(f"文字起こし時間:     {trans_sec:.2f} 秒")
    print(f"総処理時間（合計）: {all_sec:.2f} 秒")
    print("=== 完了 ===")


if __name__ == "__main__":
    main()
