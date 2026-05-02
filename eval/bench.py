import os
import subprocess
import json
import datetime

BENCHMARKS = {
    "math":     "hendrycks_math",
    "math500":  "minerva_math",
    "mmlu":     "mmlu",
    "gpqa":     "gpqa_diamond_generative_n_shot",
}

CATEGORIES = {
    "math": ["math", "math500"],
    "qa":   ["mmlu", "gpqa"],
}

PRESETS = {
    "easy": ["math500", "mmlu"],
    "hard": ["math500", "gpqa"],
}


def resolve_benchmarks(names=None, category=None, preset=None):
    selected = []
    if preset:
        selected.extend(PRESETS[preset])
    if category:
        selected.extend(CATEGORIES[category])
    if names:
        for n in names:
            if n not in BENCHMARKS:
                raise ValueError(f"Unknown benchmark '{n}'. Choose from: {list(BENCHMARKS)}")
            selected.append(n)
    # deduplicate preserving order
    seen = set()
    result = []
    for b in selected:
        if b not in seen:
            seen.add(b)
            result.append(b)
    return result


def run_bench(checkpoint_path=None, benchmarks=None, category=None, preset=None,
              batch_size=4, num_fewshot=0, model_name="Qwen/Qwen2.5-7B-Instruct"):
    selected = resolve_benchmarks(names=benchmarks, category=category, preset=preset)
    if not selected:
        raise ValueError("No benchmarks selected. Pass --benchmarks, --category, or --preset.")

    tasks = ",".join(BENCHMARKS[b] for b in selected)

    model_args = f"pretrained={model_name}"
    if checkpoint_path:
        model_args += f",peft={os.path.abspath(checkpoint_path)}"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.abspath(f"outputs/bench_{timestamp}")
    os.makedirs(output_path, exist_ok=True)

    cmd = [
        "python", "-m", "lm_eval",
        "--model", "hf",
        "--tasks", tasks,
        "--model_args", model_args,
        "--num_fewshot", str(num_fewshot),
        "--batch_size", str(batch_size),
        "--output_path", output_path,
    ]

    print(f"Running benchmarks: {selected}")
    print(f"Tasks: {tasks}")
    print(f"Output: {output_path}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nlm_eval exited with code {result.returncode}.")
        return

    # print summary from evalchemy output JSON if present
    for fname in os.listdir(output_path):
        if fname.endswith(".json"):
            with open(os.path.join(output_path, fname)) as f:
                data = json.load(f)
            results = data.get("results", {})
            if results:
                print("\n--- Results Summary ---")
                for task, metrics in results.items():
                    for k, v in metrics.items():
                        if isinstance(v, float):
                            print(f"  {task} / {k}: {v:.4f}")
            break

