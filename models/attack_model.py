import torch
import torch.nn as nn
import torch.nn.functional as F


class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)


class MLP_CE(nn.Module):
    def __init__(self):
        super(MLP_CE, self).__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 32)
        # self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        x = x.view(-1, 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x


class MLP_CE(nn.Module):
    def __init__(self, selected_posteriors=2):
        super(MLP_CE, self).__init__()
        # selected posteriors + labels
        self.input_dim = selected_posteriors + 1
        self.fc1 = nn.Linear(self.input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        # self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x


class MLP_OL(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(MLP_OL, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.fc1 = nn.Linear(self.dim_in, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc_last = nn.Linear(128, self.dim_out)

    def forward(self, x):
        # x = x.view(-1,self.dim_in)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))

        x = F.relu(self.fc_last(x))
        return x


class MLP_Adv(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(MLP_Adv, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.fc1 = nn.Linear(self.dim_in, 64)
        self.fc2 = nn.Linear(64, 64)
        # self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(64, self.dim_out)

    def forward(self, x, constant):
        x = GradReverse.grad_reverse(x, constant)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x


class MLP_Adv_Olympus(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(MLP_Adv_Olympus, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.fc1 = nn.Linear(self.dim_in, 64)
        self.fc2 = nn.Linear(64, 64)
        # self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(64, self.dim_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x


class MLP_Adv_AttriGuard(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(MLP_Adv_AttriGuard, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.fc1 = nn.Linear(self.dim_in, 64)
        self.fc2 = nn.Linear(64, 64)
        # self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(64, self.dim_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x
