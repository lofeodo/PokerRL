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

    def compute_rewards(self, 
                        output: torch.Tensor,
                        action: int,
                        pot_size: float,
                        stack_size: float,
                        is_terminal: bool,
                        is_winner: bool = None,
                        max_stack: int = 20000) -> float:
        """
        Compute rewards for the current action and game state.
        
        Args:
            output: torch.Tensor, shape: (4,)
            action: int, the action taken (0: Fold, 1: Check/Call, 2: Bet/Raise, 3: All-in)
            pot_size: float, current size of the pot
            stack_size: float, current stack size
            is_terminal: bool, whether this is a terminal state
            is_winner: bool, whether the player won (only used for terminal states)
            
        Returns:
            float: The computed reward
        """
        action_probs = F.softmax(output, dim=0)
        
        reward = 0.0
        
        if is_terminal:
            if is_winner:
                reward = pot_size
            else:
                reward = -pot_size
        else:
            if action == 0:  # Fold
                reward = -0.3 * pot_size  # Penalty for folding
            elif action == 1:  # Check/Call
                reward = 0.0  # Neutral for checking/calling
            elif action == 2:  # Bet/Raise
                reward = 0.2 * pot_size  # Medium positive reward for aggressive play
            elif action == 3:  # All-in
                reward = 0.1 * pot_size  # Small positive reward for going all-in
                
            # Scale reward by action probability to encourage exploration
            reward *= action_probs[action].item()
            
            # Add a small penalty based on stack size to encourage chip preservation
            reward -= 0.01 * (1 - stack_size / max_stack)
            
        return reward

    def value_function(self, 
                       output: torch.Tensor,
                       pot_size: float,
                       stack_size: float,
                       bet_size: float,
                       max_stack: int = 20000,
                       discount_factor: float = 0.95) -> float:
        """
        Compute the value (expected future rewards) of a given state.
        
        Args:
            output: torch.Tensor, shape: (4,)
            pot_size: float, current size of the pot
            stack_size: float, current stack size
            bet_size: float, current bet size
            max_stack: int, maximum possible stack size
            discount_factor: float, discount factor for future rewards (gamma)
            
        Returns:
            float: The estimated value of the state
        """
        action_probs = F.softmax(output, dim=0)
        
        value = 0.0
        
        # For each possible action, compute Q(s,a) and weight by action probability
        for action in range(4):
            # Compute immediate reward for the action
            immediate_reward = self.compute_rewards(
                output=output,
                action=action,
                pot_size=pot_size,
                stack_size=stack_size,
                is_terminal=False,
                max_stack=max_stack
            )
            
            # Estimate future value based on action type
            if action == 0:  # Fold
                future_value = stack_size  # Preserve remaining stack
            elif action == 1:  # Check/Call
                future_value = stack_size + 0.5 * pot_size  # Expected value of continuing
            elif action == 2:  # Bet/Raise
                future_value = stack_size + pot_size + bet_size  # Potential to win increased pot
            else:  # All-in
                future_value = 2 * stack_size  # Potential to double up
            
            # Scale future value by stack ratio
            future_value *= (stack_size / max_stack)
            
            # Combine immediate and future rewards with discount factor
            q_value = immediate_reward + discount_factor * future_value
            
            # Weight by action probability
            value += action_probs[action].item() * q_value
        
        return value

    def compute_loss(self,
                     output: torch.Tensor,
                     target_action: torch.Tensor,
                     target_value: torch.Tensor,
                     pot_size: float,
                     stack_size: float,
                     bet_size: float,
                     max_stack: int = 20000,
                     discount_factor: float = 0.95,
                     policy_weight: float = 1.0,
                     value_weight: float = 0.5) -> torch.Tensor:
        """
        Compute the combined policy and value loss.
        
        Args:
            output: torch.Tensor, shape: (4,)
            target_action: torch.Tensor, shape: (4,), one-hot encoded target action
            target_value: torch.Tensor, shape: (1,), target state value
            pot_size: float, current size of the pot
            stack_size: float, current stack size
            bet_size: float, current bet size
            max_stack: int, maximum possible stack size
            discount_factor: float, discount factor for future rewards
            policy_weight: float, weight for policy loss
            value_weight: float, weight for value loss
            
        Returns:
            torch.Tensor: The combined loss
        """
        action_probs = F.softmax(output, dim=0)
        
        # Compute policy loss (cross-entropy)
        policy_loss = F.cross_entropy(output, target_action)
        
        # Compute value loss (MSE)
        predicted_value = self.value_function(
            output=output,
            pot_size=pot_size,
            stack_size=stack_size,
            bet_size=bet_size,
            max_stack=max_stack,
            discount_factor=discount_factor
        )
        value_loss = F.mse_loss(torch.tensor(predicted_value, dtype=torch.float32), target_value)
        
        # Combine losses with weights
        total_loss = policy_weight * policy_loss + value_weight * value_loss
        
        return total_loss

    def get_training_targets(self,
                             action_representation: torch.Tensor,
                             card_representation: torch.Tensor,
                             pot_size: float,
                             stack_size: float,
                             bet_size: float,
                             max_stack: int = 20000,
                             epsilon: float = 0.1,
                             use_expert: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get target action and value for training.
        
        Args:
            action_representation: torch.Tensor, shape: (24, 4, 4)
            card_representation: torch.Tensor, shape: (6, 13, 4)
            pot_size: float, current size of the pot
            stack_size: float, current stack size
            bet_size: float, current bet size
            max_stack: int, maximum possible stack size
            epsilon: float, exploration rate for epsilon-greedy
            use_expert: bool, whether to use expert targets when available
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Target action (one-hot) and target value
        """
        # Get model's predictions
        output = self.forward(action_representation, card_representation)
        action_probs = F.softmax(output, dim=0)
        
        # Initialize target action
        target_action = torch.zeros(4, dtype=torch.float32)
        
        if use_expert and hasattr(self, 'expert_model'):
            # Use expert model's action as target
            expert_action = self.expert_model.predict_action(
                action_representation, card_representation)
            target_action[expert_action] = 1.0
        else:
            # Epsilon-greedy exploration
            if torch.rand(1) < epsilon:
                # Random action
                action = torch.randint(0, 4, (1,)).item()
            else:
                # Greedy action
                action = action_probs.argmax().item()
            target_action[action] = 1.0
        
        # Compute target value
        if use_expert and hasattr(self, 'expert_model'):
            # Use expert model's value estimate
            target_value = self.expert_model.value_function(
                output=output,
                pot_size=pot_size,
                stack_size=stack_size,
                bet_size=bet_size,
                max_stack=max_stack
            )
        else:
            # Use Monte Carlo returns or value iteration
            target_value = self.value_function(
                output=output,
                pot_size=pot_size,
                stack_size=stack_size,
                bet_size=bet_size,
                max_stack=max_stack
            )
        
        return target_action, torch.tensor(target_value, dtype=torch.float32)

