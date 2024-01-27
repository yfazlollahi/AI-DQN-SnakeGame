import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        l1 = None  # TODO
        l2 = None  # TODO
        l3 = None  # TODO
        l4 = None  # TODO
        return l4


def get_network_input(player, apple):
    proximity = player.get_proximity()
    x = torch.cat([torch.from_numpy(player.pos).double(), torch.from_numpy(apple.pos).double(),
                   torch.from_numpy(player.dir).double(), torch.tensor(proximity).double()])
    return x
