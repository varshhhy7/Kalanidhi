import os
import json
from collections import Counter
from transformers import AutoTokenizer
from datasets import load_dataset

def main():
    print("Starting Tokenizer Pruning for Kalanidhi AI...")

    hf_token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert", token=hf_token)

    print("Loading Telugu TinyStories...")
    dataset = load_dataset("neuralnets/multilingual-tinystories", split="te")
    print(f"Found {len(dataset)} Telugu rows.")

    print("Scanning stories (batched)...")
    used_token_ids = Counter()
    batch = tokenizer(list(dataset["text"]), add_special_tokens=False, padding=False, truncation=False)
    for ids in batch["input_ids"]:
        used_token_ids.update(ids)

    MIN_FREQ = 5
    special_ids = set(tokenizer.all_special_ids)
    active_ids = {tok for tok, cnt in used_token_ids.items() if cnt >= MIN_FREQ}
    final_keep_ids = sorted(special_ids | active_ids)

    id_mapping = {old_id: new_id for new_id, old_id in enumerate(final_keep_ids)}

    coverage = len(used_token_ids) / tokenizer.vocab_size * 100
    print(f"Dataset uses {len(used_token_ids)} unique tokens ({coverage:.1f}% of vocab)")

    output_dir = "src/tokenizer"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "pruned_config.json"), "w") as f:
        json.dump({
            "vocab_size": len(final_keep_ids),
            "keep_ids": final_keep_ids,
            "old_to_new": id_mapping
        }, f, indent=2)

    print(f"Vocab pruned: {tokenizer.vocab_size} -> {len(final_keep_ids)} tokens.")

if __name__ == "__main__":
    main()