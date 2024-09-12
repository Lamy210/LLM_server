import requests
import os
import concurrent.futures
from tqdm import tqdm

def download_file(url, local_filename):
    # ファイルをダウンロードする関数（進捗バー付き）
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))

    with open(local_filename, 'wb') as file, tqdm(
        desc=os.path.basename(local_filename),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)
    return local_filename

def download_repository(repo_url):
    # リポジトリの情報を取得
    api_url = f"https://huggingface.co/api/models/{repo_url}"
    response = requests.get(api_url)
    response.raise_for_status()
    repo_info = response.json()

    # ダウンロード先のディレクトリを作成
    local_dir = repo_info['id'].split('/')[-1]
    os.makedirs(local_dir, exist_ok=True)

    # ファイルの一覧を取得
    files_url = f"{api_url}/tree/main"
    response = requests.get(files_url)
    response.raise_for_status()
    files = response.json()

    # 全体の進捗バーを設定
    with tqdm(total=len(files), desc="Overall Progress") as overall_progress:
        # 並列ダウンロードの設定
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {}
            for file in files:
                if file['type'] == 'file':
                    file_url = f"https://huggingface.co/{repo_url}/resolve/main/{file['path']}"
                    local_filename = os.path.join(local_dir, file['path'])
                    os.makedirs(os.path.dirname(local_filename), exist_ok=True)
                    future = executor.submit(download_file, file_url, local_filename)
                    future_to_url[future] = file_url

            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    future.result()
                    overall_progress.update(1)
                except Exception as exc:
                    print(f"{url} generated an exception: {exc}")

# メイン処理
if __name__ == "__main__":
    repo_url = "TheBloke/Llama-2-13B-chat-GGUF"
    print(f"Starting download of repository: {repo_url}")
    download_repository(repo_url)
    print("ダウンロードが完了しました。GPUは休憩中です（笑）")