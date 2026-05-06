import torch.nn as nn

from model.embedder import EmbeddingGenerator


class SimpleLSTM(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            output_size,
            num_layers,
            dropout,
            ehr_encoder_name=None,
            cat_idxs=None,
            cat_dims=None,
            cat_emb_dim=None,
    ):
        """
        Initialize the LSTM model.

        Parameters:
        - input_size: The number of expected features in the input `x`
        - hidden_size: The number of features in the hidden state `h`
        - num_layers: Number of recurrent layers.
        - dropout: If non-zero, introduces a `Dropout` layer on the outputs of each LSTM layer except the last layer.
        """
        super(SimpleLSTM, self).__init__()

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

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if self.ehr_encoder_name == "embedder":
            batch_size, max_seq_len, _ = x.shape
            x = x.reshape(batch_size * max_seq_len, -1)
            x = self.embedder(x).reshape(batch_size, max_seq_len, -1)

        output, _ = self.lstm(x)
        output = self.fc(output[:, -1, :])
        return output
