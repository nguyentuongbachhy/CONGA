import torch

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units: int, dropout_rate: float) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(hidden_units, hidden_units * 4)
        self.fc2 = torch.nn.Linear(hidden_units * 4, hidden_units)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.act = torch.nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))