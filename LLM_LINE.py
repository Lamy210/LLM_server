import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

# GPUが利用可能かチェック
if torch.cuda.is_available():
    print("GPUを使用します：", torch.cuda.get_device_name(0))
else:
    print("GPUが利用できません。CPUを使用します。")

# モデルとトークナイザーの設定
model_name = "line-corporation/japanese-large-lm-3.6b"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, legacy=False)

# 8ビット量子化の設定
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# 量子化設定を使用してモデルをロード
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quantization_config
)

# テキスト生成パイプラインの設定
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# プロンプトの設定
prompt = "LINEについて教えて：\n\n"

# テキスト生成
text = generator(
    prompt,
    max_length=200,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    pad_token_id=tokenizer.pad_token_id,
    num_return_sequences=1,
    clean_up_tokenization_spaces=True
)

# 結果の出力
print("生成されたC言語のHello Worldプログラム：")
print("```c")
print(text[0]['generated_text'].replace(prompt, "").strip())
print("```")