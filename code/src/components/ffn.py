import torch

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units: int, dropout_rate: float) -> None:
        super(PointWiseFeedForward, self).__init__()
        self.conv1: torch.nn.Conv1d = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1) 
        self.dropout1: torch.nn.Dropout = torch.nn.Dropout(p=dropout_rate)
        self.act: torch.nn.GELU = torch.nn.GELU() 
        self.conv2: torch.nn.Conv1d = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2: torch.nn.Dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = inputs.transpose(-1, -2)
        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        outputs: torch.Tensor = x.transpose(-1, -2)
        return outputs