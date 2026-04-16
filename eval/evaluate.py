import torch
from datasets import load_dataset
from config.config import Config
from model.load_model import load_model
from eval.metrics import extract_boxed, compute_accuracy


def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def evaluate():
    config = Config()
    model, tokenizer = load_model(config)

    ds = load_dataset(config.dataset_name, split="train")
    ds = ds.shuffle(seed=42).select(range(min(20000, len(ds))))

    preds, gts = [], []

    for ex in ds.select(range(100)):
        conversations = ex["conversations"]

        # Build prompt from all turns except the final assistant turn
        messages = [{"role": "system", "content": ex["system"]}]
        for turn in conversations[:-1]:
            role = "user" if turn["from"] == "human" else "assistant"
            messages.append({"role": role, "content": turn["value"]})

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        output = generate(model, tokenizer, prompt)

        # Ground truth is the final assistant turn
        gt_text = conversations[-1]["value"]

        preds.append(extract_boxed(output))
        gts.append(extract_boxed(gt_text))

    acc = compute_accuracy(preds, gts)
    print("Final Answer Accuracy:", acc)
