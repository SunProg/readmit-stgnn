import torch.nn as nn

from model.embedder import EmbeddingGenerator


class SimpleRNN(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            output_size,
            num_layers=1,
            dropout=0.0,
            ehr_encoder_name=None,
            cat_idxs=None,
            cat_dims=None,
            cat_emb_dim=None,
    ):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.ehr_encoder_name = ehr_encoder_name
        if ehr_encoder_name is not None:
            self.embedder = EmbeddingGenerator(
                input_dim=input_size,
                cat_dims=cat_dims,
                cat_idxs=cat_idxs,
                cat_emb_dim=cat_emb_dim,
            )
            input_size = self.embedder.post_embed_dim
        else:
            self.embedder = None
        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          dropout=dropout,
                          batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if self.ehr_encoder_name == "embedder":
            batch_size, max_seq_len, _ = x.shape
            x = x.reshape(batch_size * max_seq_len, -1)
            x = self.embedder(x).reshape(batch_size, max_seq_len, -1)

        output, hidden = self.rnn(x)  # output shape: (batch, seq_len, hidden_size)
        last_output = output[:, -1, :]
        out = self.fc(last_output)
        return out
