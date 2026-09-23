import os
import re
from mutagen.easyid3 import EasyID3

# MP3ファイルが入っているフォルダ
ALBUM_TAG = r"1_H"
FOLDER_PATH = r".\\music"

def update_mp3_tags(folder_path):
    # フォルダ内のファイルを取得
    for filename in os.listdir(folder_path):

        # MP3ファイル以外は無視
        if not filename.lower().endswith(".mp3"):
            continue

        # 拡張子を除いたファイル名
        name = os.path.splitext(filename)[0]

        # [アーティスト名] タイトル の形式を抽出
        match = re.match(r"^\[([^\]]+)\]\s*(.+)$", name)

        if not match:
            print(f"形式が違うためスキップ: {filename}")
            continue

        artist = match.group(1)
        title = match.group(2)

        filepath = os.path.join(folder_path, filename)

        try:
            # ID3タグを読み込む
            try:
                audio = EasyID3(filepath)
            except Exception:
                # ID3タグが存在しない場合は新規作成
                audio = EasyID3()
                audio.save(filepath)
                audio = EasyID3(filepath)

            # アーティストとタイトルを上書き
            audio["artist"] = artist
            audio["title"] = title

            # 上記以外のタグ
            audio["album"] = ALBUM_TAG

            # 保存
            audio.save()

            print(f"更新: {filename}")
            print(f"  アーティスト: {artist}")
            print(f"  タイトル    : {title}")
            print(f"  アルバム    : {ALBUM_TAG}")

        except Exception as e:
            print(f"エラー: {filename}")
            print(f"  {e}")


if __name__ == "__main__":
    update_mp3_tags(FOLDER_PATH)