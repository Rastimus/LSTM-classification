import torch.nn as nn
import torch.nn.functional as F
class SentimentRNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_size = config.output_size
        self.n_layers = config.n_layers
        self.hidden_dim = config.hidden_dim
        self.corpus_size = config.vocab_size
        self.embedd_size = config.embed_size
        self.embedding = nn.Embedding(self.corpus_size, self.embedd_size)

        self.lstm = nn.LSTM(self.embedd_size, self.hidden_dim,self.n_layers, dropout=0.5, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.hidden_dim, self.output_size)
        self.act = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        # lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        lstm_out = lstm_out[:, -1, :]
        # 取最后一个时间步的输出，形状为 (batch_size, hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.act(out)
        out = out.view(batch_size, -1)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden
