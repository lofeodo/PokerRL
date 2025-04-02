import torch
from torch import nn
from torch.nn import Linear, Conv2d
import torch.nn.functional as F
class CNN_A(nn.Module):
    def __init__(self) -> None:
        super(CNN_A, self).__init__()
        k = 3
        s = 1
        p = 1
        self.cnn1 = Conv2d(in_channels=24, out_channels=128, kernel_size=k, stride=s, padding=p)
        self.cnn2 = Conv2d(in_channels=128, out_channels=256, kernel_size=k, stride=s, padding=p)
        self.cnn3 = Conv2d(in_channels=256, out_channels=256, kernel_size=k, stride=s, padding=p)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ 
        input: torch.Tensor, shape: (24, 4, 4)
        output: torch.Tensor, shape: (256, 4, 4)
        """
        assert input.shape[0] == 24, f"Wrong input shape of {input.shape[0]}, should be 24."
        output = self.cnn1(input)
        output = self.cnn2(output)
        output = self.cnn3(output)
        return output

class CNN_C(nn.Module):
    def __init__(self) -> None:
        super(CNN_C, self).__init__()
        k = 3
        s = 1
        p = 1
        self.cnn1 = Conv2d(in_channels=6, out_channels=64, kernel_size=k, stride=s, padding=p)
        self.cnn2 = Conv2d(in_channels=64, out_channels=128, kernel_size=k, stride=s, padding=p)
        self.cnn3 = Conv2d(in_channels=128, out_channels=256, kernel_size=k, stride=s, padding=p)
        self.cnn4 = Conv2d(in_channels=256, out_channels=256, kernel_size=k, stride=s, padding=p)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ 
        input: torch.Tensor, shape: (6, 13, 4)
        output: torch.Tensor, shape: (256, 13, 4)
        """
        assert input.shape[0] == 6, f"Wrong input shape of {input.shape[0]}, should be 6."
        output = self.cnn1(input)
        output = self.cnn2(output)
        output = self.cnn3(output)
        output = self.cnn4(output)
        return output

class MLP(nn.Module):
    def __init__(self) -> None:
        super(MLP, self).__init__()
        self.input_dim = 256*(4*4 + 13*4)
        self.fc1 = Linear(in_features=self.input_dim, out_features=390)
        self.fc2 = Linear(in_features=390, out_features=4)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ 
        input: torch.Tensor, shape: (17048,)
        output: torch.Tensor, shape: (4,)
        """
        assert input.shape[0] == self.input_dim, f"Wrong input shape of {input.shape[0]}, should be {self.input_dim}."
        output = self.fc1(input)
        output = self.fc2(output)
        return output

class BayesianHoldem(nn.Module):
    """
    Bayesian Hold'em model.
    Used to predict an action given a card representation and an action representation.
    0: Fold
    1: Check/Call
    2: Bet/Raise (BB)
    3: All-in
    """
    def __init__(self) -> None:
        super(BayesianHoldem, self).__init__()
        self.cnn_a = CNN_A()
        self.cnn_c = CNN_C()
        self.mlp = MLP()

    def forward(self, action_representation: torch.Tensor, card_representation: torch.Tensor) -> torch.Tensor:
        """ 
        card_representation: torch.Tensor, shape: (6, 13, 4)
        action_representation: torch.Tensor, shape: (24, 4, 4)
        output: torch.Tensor, shape: (4,)
        """
        output_a = self.cnn_a(action_representation)
        output_c = self.cnn_c(card_representation)
        output_a_flat = output_a.view(-1)
        output_c_flat = output_c.view(-1)
        input = torch.cat((output_a_flat, output_c_flat), dim=0)
        output = self.mlp(input)
        return output
    
    def predict_action(self, action_representation: torch.Tensor, card_representation: torch.Tensor) -> torch.Tensor:
        """
        action_representation: torch.Tensor, shape: (24, 4, 4)
        card_representation: torch.Tensor, shape: (6, 13, 4)
        output: int
        """
        output = self.forward(action_representation, card_representation)
        output_prob = F.softmax(output, dim=0)
        return output_prob.argmax().item()

