from torch import nn
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TFC(nn.Module):
    """
    The main model of TF-C, including two transformer encoders and two projectors.
    The transformer encoder has two layers, each with 2 attention heads.
    """
    def __init__(self, configs):
        """
        Parameters
        ----------
        configs: object
            configurations including model hyperparameters
            - TSlength_aligned: int, the aligned time-series length after preprocessing
        """
        super(TFC, self).__init__()

        # Time-based encoder
        encoder_layers_t = TransformerEncoderLayer(
            configs.TSlength_aligned, dim_feedforward=2*configs.TSlength_aligned,
            nhead=2, batch_first=True)
        self.transformer_encoder_t = TransformerEncoder(
            encoder_layers_t, 2)

        self.projector_t = nn.Sequential(
            nn.Linear(configs.TSlength_aligned, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )  # project to a common latent space (128-d)

        # Frequency-based encoder
        encoder_layers_f = TransformerEncoderLayer(
            configs.TSlength_aligned, dim_feedforward=2*configs.TSlength_aligned,
            nhead=2, batch_first=True)
        self.transformer_encoder_f = TransformerEncoder(encoder_layers_f, 2)

        self.projector_f = nn.Sequential(
            nn.Linear(configs.TSlength_aligned, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )  # project to a common latent space (128-d)


    def forward(self, x_in_t: torch.Tensor, x_in_f: torch.Tensor):
        """
        Parameters
        ----------
        x_in_t: torch.Tensor
            Input time-series data,
            shape: (batch_size, channels, time_length)
        x_in_f: torch.Tensor
            Input frequency-domain data,
            shape: (batch_size, channels, freq_length)

        Returns
        -------
        h_time: torch.Tensor
            Time-based representation
            shape: (batch_size, hidden_dim)
        z_time: torch.Tensor
            Time-based projection
            shape: (batch_size, projection_dim)
        h_freq: torch.Tensor
            Frequency-based representation
            shape: (batch_size, hidden_dim)
        z_freq: torch.Tensor
            Frequency-based projection
            shape: (batch_size, projection_dim)
        """
        x = self.transformer_encoder_t(x_in_t)
        h_time = x.reshape(x.shape[0], -1)

        z_time = self.projector_t(h_time)

        f = self.transformer_encoder_f(x_in_f)
        h_freq = f.reshape(f.shape[0], -1)

        z_freq = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq


class target_classifier(nn.Module):
    """Downstream classifier only used in finetuning"""
    def __init__(self, configs):
        """
        Parameters
        ----------
        configs: object
            configurations including model hyperparameters
            - num_classes_target: int, number of classes in the target dataset
        """
        super(target_classifier, self).__init__()
        self.logits = nn.Linear(2*128, 64)
        self.logits_simple = nn.Linear(64, configs.num_classes_target)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        emb: torch.Tensor
            Input embedding from the pre-trained TFC model,
            shape: (batch_size, embedding_dim)

        Returns
        -------
        pred: torch.Tensor
            Output logits for classification,
            shape: (batch_size, num_classes_target)
        """
        emb_flat = emb.reshape(emb.shape[0], -1)
        emb = torch.sigmoid(self.logits(emb_flat))
        pred = self.logits_simple(emb)
        return pred
