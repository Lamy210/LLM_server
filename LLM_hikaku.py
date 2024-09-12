import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_or_download_model(model_name, local_dir="./models"):
    tokenizer_path = os.path.join(local_dir, model_name.split('/')[-1], "tokenizer")
    model_path = os.path.join(local_dir, model_name.split('/')[-1], "model")
    
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
    
    print(f"モデル '{model_name}' の準備が完了しました。")
    return tokenizer, model

def generate_response(prompt, tokenizer, model, max_length=500):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
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

def create_json_output(model_name, prompt, response, code):
    return {
        "model": model_name,
        "prompt": prompt,
        "response": response,
        "code": code
    }

def process_model(model_name, prompt):
    tokenizer, model = load_or_download_model(model_name)
    response = generate_response(prompt, tokenizer, model)
    code = extract_code(response)
    return create_json_output(model_name, prompt, response, code)

if __name__ == "__main__":
    model_names = [
        "line-corporation/japanese-large-lm-3.6b",
        "stabilityai/japanese-stablelm-instruct-alpha-7b",
        "elyza/ELYZA-japanese-Llama-2-7b-instruct"
    ]
    
    while True:
        user_prompt = input("プロンプトを入力してください（終了する場合は 'q' を入力）: ")
        if user_prompt.lower() == 'q':
            break

        full_prompt = f"以下の指示に従って回答してください。回答には説明と、その説明に基づいたC言語のコードを含めてください。指示：{user_prompt}"
        
        results = []
        with ThreadPoolExecutor(max_workers=len(model_names)) as executor:
            future_to_model = {executor.submit(process_model, model_name, full_prompt): model_name for model_name in model_names}
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    print(f'{model_name} generated an exception: {exc}')
        
        for result in results:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print("\n" + "="*50 + "\n")