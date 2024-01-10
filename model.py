import torch
import torch.nn as nn

class AE_3DNet(nn.Module):
    def __init__(self, num_classes=2):
        super(AE_3DNet, self).__init__()

        self.group1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.InstanceNorm3d(16),
            nn.ReLU(),
            STAM(time=81, inchannels=16, reduction=4),
            nn.MaxPool3d(kernel_size=2, stride=2))

        self.group2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32),
            nn.ReLU(),
            STAM(time=40, inchannels=32, reduction=4),
            nn.MaxPool3d(kernel_size=2, stride=2))

        self.group3 = nn.Sequential(
            nn.Conv3d(32,64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64),
            nn.ReLU(),
            STAM(time=20, inchannels=64, reduction=4),
            nn.MaxPool3d(kernel_size=2, stride=2))

        self.group4 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64),
            nn.ReLU(),
            STAM(time=10, inchannels=64, reduction=4),
            nn.MaxPool3d(kernel_size=2, stride=2))

        self.GAP = nn.AvgPool3d((5, 7, 7))

        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        out = self.group1(x)
        out = self.group2(out)
        out = self.group3(out)
        out = self.group4(out)
        out = self.GAP(out)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)

        return out


class TemporalAM(nn.Module):

    def __init__(self, time, reduction):
        super(TemporalAM, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        self.fc1 = nn.Conv1d(time, time // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv1d(time // reduction, time, kernel_size=1, padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x_permute = x.permute(0, 2, 1, 3, 4)
        x_mean = self.avg_pool(x_permute)
        x_mean = x_mean.squeeze(-1).squeeze(-1)

        x_mean = self.fc1(x_mean)
        x_mean = self.relu(x_mean)
        x_mean = self.fc2(x_mean)

        x = self.sigmoid(x_mean)
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(-1).unsqueeze(-1)


        return x

class ChannelAM(nn.Module):
    def __init__(self, time, reduction):
        super(ChannelAM, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        self.fc1 = nn.Conv1d(time, time // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv1d(time // reduction, time, kernel_size=1, padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x_permute = x.permute(0, 1, 2, 3, 4)
        x_mean = self.avg_pool(x_permute)
        x_mean = x_mean.squeeze(-1).squeeze(-1)

        x_mean = self.fc1(x_mean)
        x_mean = self.relu(x_mean)
        x_mean = self.fc2(x_mean)

        x = self.sigmoid(x_mean)
        x = x.permute(0, 1, 2)
        x = x.unsqueeze(-1).unsqueeze(-1)

        return x

class SpatialAM(nn.Module):
    def __init__(self, inchannels, reduction):
        super(SpatialAM, self).__init__()

        self.conv1 = nn.Conv3d(inchannels, inchannels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(inchannels // reduction, inchannels, kernel_size=1, padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x_mean = torch.mean(x, dim=(2), keepdim=True)
        x_mean = x_mean.squeeze(-1).squeeze(-1)

        x_mean = self.conv1(x_mean)
        x_mean = self.relu(x_mean)
        x_mean = self.conv2(x_mean)

        x = self.sigmoid(x_mean)

        return x

class STAM(nn.Module):
    def __init__(self, time, inchannels, reduction):
        super(STAM, self).__init__()
        # time, reduction

        self.TAM = TemporalAM(time, reduction)
        self.CAM = ChannelAM(inchannels, reduction)
        self.SAM = SpatialAM(inchannels, reduction)

    def forward(self, x):

        ## Channel attention
        c_a = self.CAM(x)
        x = x * c_a

        ## Temporal attention
        t_a = self.TAM(x)
        x = x * t_a

        ## Spatial attention
        s_a = self.SAM(x)
        x = x * s_a

        return x
