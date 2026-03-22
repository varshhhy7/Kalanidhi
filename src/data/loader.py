import json
import torch
from pathlib import Path
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

class KalanidhiDataset(Dataset):
    def __init__(self, max_length=256):
        root_path = Path(__file__).parent.parent.parent
        config_path = root_path / "data" / "tokenizer" / "pruned_config.json"

        with open(config_path, "r") as f:
            config = json.load(f)

        self.old_to_new = {int(k): v for k, v in config["old_to_new"].items()}

        print("Loading Telugu TinyStories...")
        self.dataset = load_dataset("neuralnets/multilingual-tinystories", split="te")
        self.tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None
        )

        new_ids = [self.old_to_new.get(tid, 0) for tid in tokens["input_ids"]]
        attention_mask = tokens["attention_mask"]

        return {
            "input_ids": torch.tensor(new_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }