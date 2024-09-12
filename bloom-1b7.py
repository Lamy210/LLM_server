from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch
torch.set_default_tensor_type(torch.cuda.FloatTensor)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b7", use_cache=True,load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b7")

set_seed(42)
prompt = '四国の県名を全て列挙してください。'
input_ids = tokenizer(prompt, return_tensors="pt").to(device)
sample = model.generate(**input_ids, max_length=64, temperature=0.7)
print(tokenizer.decode(sample[0], truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"]))