import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_name = "google/gemma-2b"

# 🔹 Your checkpoint path
adapter_path = r"C:\Users\Abhim\OneDrive\Documents\Python_Programs\llm_finetuning\outputs\checkpoint-1228"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(model, adapter_path)

model.eval()

prompt = "What is 1 + 2?"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

# Decode
response = tokenizer.decode(output[0], skip_special_tokens=True)

print(response)