import torch
from datasets import load_dataset
from config.config import Config
from model.load_model import load_model
from eval.metrics import extract_boxed, compute_accuracy

EVAL_BATCH_SIZE = 8


def _build_prompt(ex, tokenizer):
    messages = [{"role": "system", "content": ex["system"]}]
    for turn in ex["conversations"][:-1]:
        role = "user" if turn["from"] == "human" else "assistant"
        messages.append({"role": role, "content": turn["value"]})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_batch(model, tokenizer, prompts):
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                       max_length=4096).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    return [tokenizer.decode(o[input_len:], skip_special_tokens=True) for o in outputs]


def evaluate(checkpoint_path=None, n_examples=100, train_subset_size=None):
    config = Config()
    if train_subset_size is not None:
        config.train_subset_size = train_subset_size
    model, tokenizer = load_model(config)

    if checkpoint_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint_path)
        # model = model.merge_and_unload()
        # model = model.to(torch.bfloat16)
    model.eval()

    ds = load_dataset(config.dataset_name, split="train")
    ds = ds.shuffle(seed=42)
    train_end = min(config.train_subset_size, len(ds))
    held_out = ds.select(range(train_end, len(ds)))
    print(f"Train window: {train_end}, held-out size: {len(held_out)}")
    if len(held_out) == 0:
        raise RuntimeError(
            f"No held-out examples: train_subset_size={config.train_subset_size} "
            f">= dataset size {len(ds)}. Reduce train_subset_size."
        )
    ds = held_out.select(range(min(n_examples, len(held_out))))

    examples = list(ds)
    preds, gts = [], []

    for i in range(0, len(examples), EVAL_BATCH_SIZE):
        batch = examples[i:i + EVAL_BATCH_SIZE]
        prompts = [_build_prompt(ex, tokenizer) for ex in batch]
        outputs = generate_batch(model, tokenizer, prompts)
        
        for j, output in enumerate(outputs):
            print(f"\n--- Example {j+1} output ---")
            print(output[-800:])
            print("--- end ---")
        for ex, output in zip(batch, outputs):
            gt_text = ex["conversations"][-1]["value"]
            pred = extract_boxed(output)
            gt = extract_boxed(gt_text)
            preds.append(pred)
            gts.append(gt)

        print(f"Processed {min(i + EVAL_BATCH_SIZE, len(examples))}/{len(examples)}")

    acc = compute_accuracy(preds, gts)
    print(f"Preds non-None: {sum(p is not None for p in preds)}/{len(preds)}")
    print(f"GTs non-None:   {sum(g is not None for g in gts)}/{len(gts)}")
    print("Final Answer Accuracy:", acc)

