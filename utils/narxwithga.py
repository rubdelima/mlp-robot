import torch.nn as nn

# Define NARX model
class NARXModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=3, output_dim=2):
        super(NARXModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)
