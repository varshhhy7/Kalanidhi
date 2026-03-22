import torch

class KalanidhiDiffusion:
    def __init__(self, mask_token_id=4, cls_token_id=2, sep_token_id=3):
        self.mask_token_id = mask_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id

    def apply_noise(self, x, t):
        """
        x: Clean token IDs [batch, seq_len]
        t: Noise level, float or tensor of shape [batch] or scalar.
           0.0 = no masking, 1.0 = mask everything.
        """
        # Normalize t to [batch, 1] tensor
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32, device=x.device)
        if t.dim() == 0:
            t = t.expand(x.size(0))
        mask_prob = t.view(-1, 1)

        # Randomly select positions to mask
        noise_mask = torch.rand(x.shape, device=x.device) < mask_prob

        # Never mask [CLS] or [SEP] tokens
        special_tokens_mask = (x == self.cls_token_id) | (x == self.sep_token_id)
        final_mask = noise_mask & ~special_tokens_mask

        noised_x = x.clone()
        noised_x[final_mask] = self.mask_token_id

        return noised_x, final_mask

    def sample_t(self, batch_size, device):
        """Sample random noise levels for a training batch."""
        return torch.rand(batch_size, device=device)