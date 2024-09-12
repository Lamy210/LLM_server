import os
from llama_cpp import Llama
import requests

# モデルのURL
model_url = "https://huggingface.co/hiyouga/ELYZA-japanese-Llama-2-7b-fast-instruct-q4_K_M/resolve/main/ELYZA-japanese-Llama-2-7b-fast-instruct-q4_K_M.gguf"

# モデルのパス
model_path = "./ELYZA-japanese-Llama-2-7b-fast-instruct-q4_K_M.gguf"

# モデルが存在しない場合、ダウンロードする
if not os.path.exists(model_path):
    print("モデルをダウンロードしています...")
    response = requests.get(model_url)
    with open(model_path, "wb") as f:
        f.write(response.content)
    print("モデルのダウンロードが完了しました。")

# モデルの準備
llm = Llama(model_path=model_path,
    n_gpu_layers=20 # gpuに処理させるlayerの数(設定しない場合はCPUだけで処理を行う)
)

# プロンプトの準備
prompt = """
質問: 日本の首都はどこですか？
答え: """

# 推論の実行
output = llm(
    prompt=prompt,
    stop=["質問:", "答え:", "\n"], # 停止文字列を設定
    echo=True, # 解答だけでなくpromptも出力する
)

print(output["choices"][0]["text"]) # 結果を出力