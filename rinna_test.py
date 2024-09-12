import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import os

def load_or_download_model(model_name, local_dir="./models"):
    tokenizer_path = os.path.join(local_dir, model_name, "tokenizer")
    model_path = os.path.join(local_dir, model_name, "model")
    
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )
    
    try:
        print(f"ローカルからモデル '{model_name}' を読み込んでいます...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto"
        )
    except (OSError, ValueError):
        print(f"ローカルモデルの読み込みに失敗しました。モデル '{model_name}' をダウンロードしています...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
        
        # ローカルに保存
        print("モデルをローカルに保存しています...")
        tokenizer.save_pretrained(tokenizer_path)
        model.save_pretrained(model_path)
    
    print("モデルの準備が完了しました。")
    return tokenizer, model

def generate_response(prompt, tokenizer, model, max_length=500):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}  # 入力をモデルと同じデバイスに移動
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def extract_code(text):
    start = text.find('```c')
    end = text.rfind('```')
    if start != -1 and end != -1:
        return text[start+4:end].strip()
    return None

def create_json_output(prompt, response, code):
    return {
        "prompt": prompt,
        "response": response,
        "code": code
    }

if __name__ == "__main__":
    model_name = "rinna/japanese-gpt-neox-3.6b"
    tokenizer, model = load_or_download_model(model_name)
    
    while True:
        user_prompt = input("プロンプトを入力してください（終了する場合は 'q' を入力）: ")
        if user_prompt.lower() == 'q':
            break

        full_prompt = f"以下の指示に従って回答してください。回答には説明と、その説明に基づいたC言語のコードを含めてください。指示：{user_prompt}"
        response = generate_response(full_prompt, tokenizer, model)
        
        code = extract_code(response)
        json_output = create_json_output(user_prompt, response, code)
        
        print(json.dumps(json_output, ensure_ascii=False, indent=2))
        print("\n")