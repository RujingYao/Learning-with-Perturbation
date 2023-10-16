
import torch
from torch import nn
from transformers import BertModel, AutoTokenizer


class BERTModel(nn.Module):
    def __init__(self, args):
        super(BERTModel, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained(
            'bert-base-uncased', do_lower_case=True)
        self.fc = nn.Linear(self.bert.config.hidden_size, self.args.output_dim)
        
        self.bias_pert = nn.Embedding(self.args.train_num, 2)
        self.bias_pert.weight.data.copy_(torch.zeros(20000, 2))

        for param in self.bert.parameters():
            param.requires_grad = True
            
    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments,ids,train):
        last_hidden_state = self.bert(
            input_ids=batch_seqs, token_type_ids=batch_seq_masks, attention_mask=batch_seq_segments)[1]
        logits = self.fc(last_hidden_state)                
        bias_logpert = self.bias_pert(ids)
        if train:            
            logits = logits + bias_logpert            
        return logits, bias_logpert
