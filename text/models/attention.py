import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(torch.nn.Module):

    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.tanh = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x):
        H0 = self.tanh(x)
        H1 = torch.matmul(H0, self.w)
        H2 = nn.functional.softmax(H1, dim=1)
        alpha = H2.unsqueeze(-1)
        att_hidden = torch.sum(x * alpha, 1)
        return att_hidden, H2


class EMBEDDING(nn.Module):
    def __init__(self, embed_matrix, args):
        super().__init__()
        self.args = args
        
        self.embedding = nn.Embedding(
            embed_matrix.shape[0], embed_matrix.shape[1])
        self.embedding.weight.data.copy_(embed_matrix)
     

    def forward(self, x):
        x = self.embedding(x)
        return x


class BILSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_dim = self.args.hidden_dim
        self.output_dim = self.args.output_dim
        self.dropout = nn.Dropout(self.args.dropout)
        
        self.rnn = nn.LSTM(300, self.hidden_dim//2,
                           bidirectional=True,
                           num_layers=1)
        
        self.attention = Attention(self.hidden_dim)

        self.fc = nn.Linear(self.hidden_dim, self.output_dim)


    def forward(self, x):
        x = self.dropout(x)
        out, _ = self.rnn(x)
        attn, _ = self.attention(out)
        logit = self.fc(attn)
        return logit
