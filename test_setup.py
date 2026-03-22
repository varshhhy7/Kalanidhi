import torch
from src.model.config import KalanidhiConfig
from src.model.transformer import KalanidhiModel
from src.data.loader import KalanidhiDataset

def test():
    print("Testing Kalanidhi AI Phase 2 Setup...")

    # 1. Test Config & Model
    config = KalanidhiConfig()
    model = KalanidhiModel(config)
    print(f"Model initialized with {config.vocab_size} tokens.")

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model size: {params:.2f}M parameters (Target: 4.4M)")

    # 2. Test Data Loader
    try:
        dataset = KalanidhiDataset()
        sample_batch = dataset[0]

        # dataset now returns a dict, not a tensor
        input_ids = sample_batch["input_ids"]
        attention_mask = sample_batch["attention_mask"]

        print(f"Data loader working. input_ids shape: {input_ids.shape}")
        print(f"Sample token IDs: {input_ids[:10]}...")
        print(f"Attention mask: {attention_mask[:10]}...")

    except Exception as e:
        print(f"Data loader error: {e}")

if __name__ == "__main__":
    test()  