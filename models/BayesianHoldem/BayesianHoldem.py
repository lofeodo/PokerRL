import torch
from torch import nn
from torch.nn import Linear, Conv2d
import torch.nn.functional as F
from typing import Tuple
import torch.optim as optim

class CNN_A(nn.Module):
    def __init__(self) -> None:
        super(CNN_A, self).__init__()
        k = 3
        s = 1
        p = 1
        self.cnn1 = Conv2d(in_channels=24, out_channels=128, kernel_size=k, stride=s, padding=p)
        self.cnn2 = Conv2d(in_channels=128, out_channels=256, kernel_size=k, stride=s, padding=p)
        self.cnn3 = Conv2d(in_channels=256, out_channels=256, kernel_size=k, stride=s, padding=p)
        self.initialize_weights()

    def initialize_weights(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0)

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
        self.initialize_weights()

    def initialize_weights(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0)

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
        self.initialize_weights()

    def initialize_weights(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0)

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
    def __init__(self, learning_rate: float = 0.0001, use_polyak: bool = False) -> None:
        super(BayesianHoldem, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_polyak = use_polyak
        
        # Initialize networks
        self.cnn_a = CNN_A().to(self.device)
        self.cnn_c = CNN_C().to(self.device)
        self.mlp = MLP().to(self.device)
        
        # Exploration parameters
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.1
        self.epsilon_decay = 0.999999
        self.current_epsilon = self.initial_epsilon
        self.training_steps = 0
        
        if use_polyak:
            # Store a copy of initial parameters for polyak averaging
            self.stored_params = {
                'cnn_a': {k: v.clone() for k, v in self.cnn_a.state_dict().items()},
                'cnn_c': {k: v.clone() for k, v in self.cnn_c.state_dict().items()},
                'mlp': {k: v.clone() for k, v in self.mlp.state_dict().items()}
            }
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Pre-allocate tensors for efficiency
        self.action_probs = torch.zeros(4, device=self.device)
        self.target_action = torch.zeros(4, device=self.device)
        self.target_value = torch.zeros(1, device=self.device)

    def update_target_network(self, tau: float):
        """Update network parameters using polyak averaging: θnew = τθcandidate + (1-τ)θold"""
        if not self.use_polyak:
            return
            
        # Update CNN_A parameters
        current_cnn_a = self.cnn_a.state_dict()
        for key in current_cnn_a:
            current_cnn_a[key].copy_(tau * current_cnn_a[key] + (1.0 - tau) * self.stored_params['cnn_a'][key])
            self.stored_params['cnn_a'][key].copy_(current_cnn_a[key])
            
        # Update CNN_C parameters
        current_cnn_c = self.cnn_c.state_dict()
        for key in current_cnn_c:
            current_cnn_c[key].copy_(tau * current_cnn_c[key] + (1.0 - tau) * self.stored_params['cnn_c'][key])
            self.stored_params['cnn_c'][key].copy_(current_cnn_c[key])
            
        # Update MLP parameters
        current_mlp = self.mlp.state_dict()
        for key in current_mlp:
            current_mlp[key].copy_(tau * current_mlp[key] + (1.0 - tau) * self.stored_params['mlp'][key])
            self.stored_params['mlp'][key].copy_(current_mlp[key])

    def update_epsilon(self):
        """Update epsilon based on decay rate and training steps."""
        self.current_epsilon = max(
            self.final_epsilon,
            self.initial_epsilon * (self.epsilon_decay ** self.training_steps)
        )
        self.training_steps += 1

    def forward(self, action_representation: torch.Tensor, card_representation: torch.Tensor) -> torch.Tensor:
        """ 
        card_representation: torch.Tensor, shape: (6, 13, 4)
        action_representation: torch.Tensor, shape: (24, 4, 4)
        output: torch.Tensor, shape: (4,)
        """
        # Ensure inputs are on the correct device
        action_representation = action_representation.to(self.device)
        card_representation = card_representation.to(self.device)
        
        output_a = self.cnn_a(action_representation)
        output_c = self.cnn_c(card_representation)
        output_a_flat = output_a.view(-1)
        output_c_flat = output_c.view(-1)
        input = torch.cat((output_a_flat, output_c_flat), dim=0)
        output = self.mlp(input)
        return output
    
    def predict_action(self, action_representation: torch.Tensor, card_representation: torch.Tensor, 
                      training: bool = False) -> torch.Tensor:
        """
        Predict an action using epsilon-greedy exploration.
        
        Args:
            action_representation: torch.Tensor, shape: (24, 4, 4)
            card_representation: torch.Tensor, shape: (6, 13, 4)
            training: bool, whether this is during training (affects epsilon update)
            
        Returns:
            int: The chosen action
        """
        with torch.no_grad():
            output = self.forward(action_representation, card_representation)
            output_prob = F.softmax(output, dim=0)
            
            # Epsilon-greedy exploration
            if training and torch.rand(1, device=self.device) < self.current_epsilon:
                # Random action
                action = torch.randint(0, 4, (1,), device=self.device).item()
                #if self.training_steps % 5000 == 0:  # Print exploration rate periodically
                    #print(f"Exploration rate: {self.current_epsilon:.4f}")
            else:
                # Greedy action
                action = output_prob.argmax().item()
            
            if training:
                self.update_epsilon()
            
            return action

    def compute_rewards(self, 
                        output: torch.Tensor,
                        action: int,
                        pot_size: float,
                        stack_size: float,
                        is_terminal: bool,
                        is_winner: bool = None,
                        max_stack: int = 20000,
                        bb: int = 100) -> float:
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
        
        # Normalize pot and stack sizes consistently
        normalized_pot = pot_size / max_stack
        
        reward = torch.zeros(1, device=self.device)
        
        if is_terminal:
            if is_winner:
                # Terminal state win: gain the pot
                reward = torch.tensor(2 * normalized_pot, device=self.device)
            else:
                # Terminal state loss: lose our contribution to pot
                reward = torch.tensor(-1.5 * normalized_pot, device=self.device)
        else:
            # Non-terminal states: immediate stack change + future pot value
            if action == 0:  # Fold
                reward = torch.tensor(-0.5 * normalized_pot, device=self.device)  # Penalty for folding
            elif action == 1:  # Check/Call
                pot_contribution = 1 * normalized_pot  # Expected value from pot
                reward = torch.tensor(pot_contribution, device=self.device)
            elif action == 2:  # Bet/Raise
                stack_delta = -bb  # Immediate stack decrease
                pot_contribution = 2 * normalized_pot  # Higher expected value from pot due to fold equity
                reward = torch.tensor((stack_delta / max_stack) + pot_contribution, device=self.device)
            elif action == 3:  # All-in
                pot_contribution = 0.25 * normalized_pot  # Maximum expected value from pot
                reward = torch.tensor(pot_contribution, device=self.device)

            # Scale reward by action probability to encourage exploration
            reward *= action_probs[action]

            
        return reward.item()

    def value_function(self, 
                       output: torch.Tensor,
                       pot_size: float,
                       stack_size: float,
                       bet_size: float,
                       max_stack: int = 20000,
                       discount_factor: float = 0.95,
                       is_terminal: bool = False,
                       is_winner: bool = None) -> float:
        """
        Compute the value (expected future rewards) of a given state.
        
        Args:
            output: torch.Tensor, shape: (4,)
            pot_size: float, current size of the pot
            stack_size: float, current stack size
            bet_size: float, current bet size
            max_stack: int, maximum possible stack size
            discount_factor: float, discount factor for future rewards (gamma)
            is_terminal: bool, whether this is a terminal state
            is_winner: bool, whether the player won (only used for terminal states)
            
        Returns:
            float: The estimated value of the state
        """
        action_probs = F.softmax(output, dim=0)
        
        # Normalize pot, stack and bet sizes consistently
        normalized_pot = pot_size / max_stack
        normalized_stack = stack_size / max_stack
        normalized_bet = bet_size / max_stack
        
        value = torch.zeros(1, device=self.device)
        
        # For each possible action, compute Q(s,a) and weight by action probability
        for action in range(4):  
            # Compute immediate reward for the action
            immediate_reward = self.compute_rewards(
                output=output,
                action=action,
                pot_size=pot_size,
                stack_size=stack_size,
                is_terminal=is_terminal,
                is_winner=is_winner,
                max_stack=max_stack
            )
            
            # Estimate future value based on action type and normalized values
            if action == 0:  # Fold
                future_value = normalized_stack  # Keep current stack
            elif action == 1:  # Check/Call
                future_value = normalized_stack + 0.5 * normalized_pot  # Current stack + expected pot share
            elif action == 2:  # Bet/Raise
                # Current stack - bet + expected increased pot share
                future_value = normalized_stack - normalized_bet + 0.7 * (normalized_pot + normalized_bet)
            else:  # All-in
                # Maximum risk/reward: lose all stack but potential to win full pot
                future_value = normalized_stack
            
            # Combine immediate and future rewards with discount factor
            q_value = torch.tensor(immediate_reward, device=self.device) + discount_factor * future_value
            
            # Weight by action probability
            value += action_probs[action] * q_value
        
        return value.item()

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
                     value_weight: float = 2.0) -> torch.Tensor:
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
        # Ensure tensors are on the correct device
        target_action = target_action.to(self.device)
        target_value = target_value.to(self.device)
        
        # Compute policy loss (cross-entropy)
        policy_loss = F.cross_entropy(output, target_action)

        # Normalize policy loss to be between 0 and 1
        policy_loss = policy_loss / torch.log(torch.tensor(4.0, device=self.device))
        policy_loss = torch.clamp(policy_loss, 0.0, 500.0)
        policy_loss = policy_loss / 100.0
        
        # Compute value loss (MSE)
        predicted_value = self.value_function(
            output=output,
            pot_size=pot_size,
            stack_size=stack_size,
            bet_size=bet_size,
            max_stack=max_stack,
            discount_factor=discount_factor
        )
        value_loss = F.mse_loss(torch.tensor(predicted_value, device=self.device), target_value)

        # Combine losses with weights
        total_loss = policy_weight * policy_loss + value_weight * value_loss
        
        return total_loss

    def get_training_targets(self,
                             action_representation: torch.Tensor,
                             card_representation: torch.Tensor,
                             pot_size: float,
                             stack_size: float,
                             bet_size: float,
                             actual_return: float,
                             is_terminal: bool = False,
                             is_winner: bool = None,
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
            actual_return: float, the actual return (or bootstrapped return) for this state
            is_terminal: bool, whether this is a terminal state
            is_winner: bool, whether the player won (only used for terminal states)
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
        target_action = torch.zeros(4, device=self.device)
        
        if use_expert and hasattr(self, 'expert_model'):
            # Use expert model's action as target
            expert_action = self.expert_model.predict_action(
                action_representation, card_representation)
            target_action[expert_action] = 1.0
        else:
            # Epsilon-greedy exploration
            action = self.predict_action(action_representation, card_representation, training=True)
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
            target_value = actual_return
        
        return target_action, torch.tensor(target_value, device=self.device)

    def train_step(self,
                  action_representation: torch.Tensor,
                  card_representation: torch.Tensor,
                  pot_size: float,
                  stack_size: float,
                  bet_size: float,
                  actual_return: float,
                  is_terminal: bool = False,
                  is_winner: bool = None,
                  max_stack: int = 20000,
                  discount_factor: float = 0.95,
                  policy_weight: float = 1.0,
                  value_weight: float = 2.0) -> Tuple[float, float, float]:
        """
        Perform a single training step.
        
        Args:
            action_representation: torch.Tensor, shape: (24, 4, 4)
            card_representation: torch.Tensor, shape: (6, 13, 4)
            pot_size: float, current size of the pot
            stack_size: float, current stack size
            bet_size: float, current bet size
            actual_return: float, the actual return (or bootstrapped return) for this state
            is_terminal: bool, whether this is a terminal state
            is_winner: bool, whether the player won (only used for terminal states)
            max_stack: int, maximum possible stack size
            discount_factor: float, discount factor for future rewards
            policy_weight: float, weight for policy loss
            value_weight: float, weight for value loss
            
        Returns:
            Tuple[float, float, float]: Total loss, policy loss, and value loss
        """
        target_action, target_value = self.get_training_targets(
            action_representation=action_representation,
            card_representation=card_representation,
            actual_return=actual_return,
            pot_size=pot_size,
            stack_size=stack_size,
            bet_size=bet_size,
            is_terminal=is_terminal,
            is_winner=is_winner
        )

        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        output = self.forward(action_representation, card_representation)
        
        # Compute predicted value using current model
        predicted_value = self.value_function(
            output=output,
            pot_size=pot_size,
            stack_size=stack_size,
            bet_size=bet_size,
            max_stack=max_stack,
            discount_factor=discount_factor,
            is_terminal=is_terminal,
            is_winner=is_winner
        )
                
        # Compute loss
        total_loss = self.compute_loss(
            output=output,
            target_action=target_action,
            target_value=target_value,
            pot_size=pot_size,
            stack_size=stack_size,
            bet_size=bet_size,
            max_stack=max_stack,
            discount_factor=discount_factor,
            policy_weight=policy_weight,
            value_weight=value_weight
        )
        
        # Backward pass
        total_loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        # Update weights
        self.optimizer.step()
        
        # Get individual losses for logging
        with torch.no_grad():
            policy_loss = F.cross_entropy(output, target_action)
            value_loss = F.mse_loss(torch.tensor(predicted_value, device=self.device), target_value)
        
        return total_loss.item(), policy_loss.item(), value_loss.item()

