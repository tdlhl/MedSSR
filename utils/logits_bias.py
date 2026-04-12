import torch
from transformers.generation.logits_process import LogitsProcessor


class LogitBiasProcess(LogitsProcessor):
    def __init__(self, activate_token_list=None, activate_scale=100):
        self.activate_token_list = activate_token_list or []
        self.activate_scale = activate_scale

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for token_id in self.activate_token_list:
            if scores.dim() == 2:
                scores[:, token_id] += self.activate_scale
            else:
                scores[token_id] += self.activate_scale
        return scores
