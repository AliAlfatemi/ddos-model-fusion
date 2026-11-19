"""Research model definitions extracted without notebook execution."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1(nn.Module):
    def __init__(self, n_features):
        super(Conv1, self).__init__()
        self.n_features = n_features
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 50, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(50)
        self.conv3 = nn.Conv1d(50, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), self.n_features, -1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


class Model2(nn.Module):
    def __init__(self, n_features, hidden_size, output_size=1):
        super(Model2, self).__init__()
        self.gru = nn.GRU(n_features, hidden_size, batch_first=True)
        # Using just one Linear layer with dropout
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(self.dropout(out))

        return self.sigmoid(out)


class Model3(nn.Module):
    def __init__(self, n_features):
        super(Model3, self).__init__()

        # First hidden layer with 64 neurons
        self.layer1 = nn.Linear(n_features, 64)
        self.bn1 = nn.BatchNorm1d(64)

        # Second hidden layer with 32 neurons
        self.layer2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)

        # Third hidden layer with 16 neurons
        self.layer3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)

        # Dropout layer
        self.dropout = nn.Dropout(0.4)

        # Output layer
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.layer3(x)))
        x = self.dropout(x)

        x = torch.sigmoid(self.output(x))
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Model4(nn.Module):
    def __init__(self, n_features):
        super(Model4, self).__init__()
        self.block1 = ResidualBlock(n_features, 64)
        self.block2 = ResidualBlock(64, 32, 2)
        self.linear = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Assuming input is of shape [batch_size, features], we reshape it to [batch_size, features, 1] for Conv1D
        x = x.unsqueeze(2)
        out = self.block1(x)
        out = self.block2(out)
        # Global average pooling
        out = F.avg_pool1d(out, out.shape[2])
        out = out.squeeze(2)
        out = self.sigmoid(self.linear(out))
        return out
