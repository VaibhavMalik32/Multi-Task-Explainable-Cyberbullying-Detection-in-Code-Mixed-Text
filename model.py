import torch
import torch.nn as nn
from transformers import AutoModel
class MultiTaskModel(nn.Module):
    def __init__(self, num_targets):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            "l3cube-pune/hing-bert"
        )
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.bully_head = nn.Linear(hidden, 2)
        self.target_head = nn.Linear(hidden, num_targets)
        self.rationale_head = nn.Linear(hidden, 2)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = outputs.last_hidden_state[:, 0]
        sequence_output = outputs.last_hidden_state 
        cls = self.dropout(cls)
        sequence_output = self.dropout(sequence_output)
        bully_logits = self.bully_head(cls)
        target_logits = self.target_head(cls)
        rationale_logits = self.rationale_head(sequence_output)
        return bully_logits, target_logits, rationale_logits