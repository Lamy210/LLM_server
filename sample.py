from llama_cpp import Llama
# プロンプトを記入
prompt = """[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>
リャマについての短い物語を書いてください。日本語で答えてください。[/INST]"""
# ダウンロードしたModelをセット.
llm = Llama(
    model_path="Llama-2-13B-chat-GGUF/llama-2-13b-chat.Q5_K_S.gguf",
    n_gpu_layers=-1,  # すべての層をGPUで処理
    n_ctx=2048,  # コンテキストウィンドウサイズを増やす
    n_batch=512,  # バッチサイズを増やす
    verbose=True  # 詳細なログを出力
)

# 生成実行
output = llm(
    prompt,
    max_tokens=500,
    stop=["System:", "User:", "Assistant:"],
    echo=True,
    temperature=0.7,  # 創造性を調整
    top_p=0.95,  # サンプリングの多様性を調整
)

print(output['choices'][0]['text'])