import torch
from config.config import Config
from model.load_model import load_model
from data.preprocess import get_dataset
from eval.metrics import extract_boxed, compute_accuracy


def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def evaluate():
    config = Config()
    model, tokenizer = load_model(config)

    _, val_ds = get_dataset(config, tokenizer)

    preds, gts = [], []

    for ex in val_ds.select(range(100)):
        prompt = tokenizer.apply_chat_template(
            ex["messages"][:-1],
            tokenize=False,
            add_generation_prompt=True
        )

        output = generate(model, tokenizer, prompt)

        pred = extract_boxed(output)
        gt = extract_boxed(ex["messages"][-1]["content"])

        preds.append(pred)
        gts.append(gt)

    acc = compute_accuracy(preds, gts)
    print("Final Answer Accuracy:", acc)
