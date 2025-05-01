import torch
import torch.nn as nn
import torch.nn.functional as F
from depthLimitedCFR import DepthLimitedCFR
from handEvaluator import HandEvaluator
import numpy as np
import random

class CFRLoss(nn.Module):
    """
    Loss function that combines traditional RL loss with CFR-based loss.
    """
    
    def __init__(self, blueprint_path=None, max_depth=2, cfr_weight=0.5):
        """
        Initialise la fonction de perte CFR.
        
        Args:
            blueprint_path: Chemin vers une stratégie CFR pré-entraînée
            max_depth: Profondeur maximale de recherche pour le solveur CFR
            cfr_weight: Poids de la composante CFR dans la perte totale
        """
        super(CFRLoss, self).__init__()
        self.cfr_solver = DepthLimitedCFR(max_depth=max_depth)
        self.cfr_weight = cfr_weight
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Charger le blueprint s'il est fourni
        if blueprint_path:
            try:
                self.cfr_solver.load(blueprint_path)
                print(f"Loaded CFR blueprint from {blueprint_path}")
            except Exception as e:
                print(f"Failed to load blueprint: {e}. Training new blueprint...")
                self.cfr_solver.train(num_iterations=1000)
    
    def pokerkit_card_to_card(self, pokerkit_card):
        """
        Convertit une carte pokerkit en carte de notre système.
        Gère correctement les différents formats de cartes possibles.
        """
        try:
            from card import Card
            
            # Si c'est une liste avec un seul élément (cas des cartes communautaires)
            if isinstance(pokerkit_card, list) and len(pokerkit_card) == 1:
                # L'élément est généralement une chaîne comme '3s', 'Ts', etc.
                card_str = pokerkit_card[0]
                if len(card_str) >= 2:
                    rank = card_str[0]
                    suit = card_str[1].lower()
                    return Card(f"{rank}{suit}")
            
            # Si c'est un objet Card avec attributs rank et suit
            elif hasattr(pokerkit_card, 'rank') and hasattr(pokerkit_card, 'suit'):
                rank = str(pokerkit_card.rank)
                # Prendre le premier caractère de la couleur
                suit = pokerkit_card.suit[0].lower() if len(pokerkit_card.suit) > 0 else 's'
                return Card(f"{rank}{suit}")
                
            # Si c'est un tuple ou une liste de longueur >= 2
            elif isinstance(pokerkit_card, (list, tuple)) and len(pokerkit_card) >= 2:
                rank = str(pokerkit_card[0])
                # Prendre le premier caractère de la couleur
                suit = str(pokerkit_card[1])[0].lower()
                return Card(f"{rank}{suit}")
                
            # Si c'est une chaîne de caractères
            elif isinstance(pokerkit_card, str):
                if len(pokerkit_card) >= 2:
                    rank = pokerkit_card[0]
                    suit = pokerkit_card[1].lower()
                    return Card(f"{rank}{suit}")
            
            # Cas par défaut si aucun des cas ci-dessus ne correspond
            return Card("As")  # Carte par défaut
            
        except Exception as e:
            # Importer Card en cas d'erreur
            from card import Card
            return Card("As")  # Carte par défaut
    
    def create_info_set_from_pokerkit(self, pokerkit_state, player_id):
        """
        Crée une représentation d'ensemble d'information à partir de l'état pokerkit.
        """
        try:
            # Cartes du joueur
            player_cards = []
            if hasattr(pokerkit_state, 'hole_cards') and pokerkit_state.hole_cards:
                if 0 <= player_id < len(pokerkit_state.hole_cards):
                    for card in pokerkit_state.hole_cards[player_id]:
                        converted_card = self.pokerkit_card_to_card(card)
                        player_cards.append(str(converted_card))
            
            # Cartes communautaires
            board_cards = []
            if hasattr(pokerkit_state, 'board_cards') and pokerkit_state.board_cards:
                for card in pokerkit_state.board_cards:
                    converted_card = self.pokerkit_card_to_card(card)
                    board_cards.append(str(converted_card))
            
            # Autres informations de l'état
            street = pokerkit_state.street_index if hasattr(pokerkit_state, 'street_index') and pokerkit_state.street_index is not None else 0
            
            pot = 0
            if hasattr(pokerkit_state, 'pot_amounts') and pokerkit_state.pot_amounts:
                pot = sum(pokerkit_state.pot_amounts)
            
            bets = [0, 0]
            if hasattr(pokerkit_state, 'bets') and len(pokerkit_state.bets) >= 2:
                bets = [pokerkit_state.bets[0], pokerkit_state.bets[1]]
            
            # Création du format final
            info_set = ''.join(player_cards) + '|' + ''.join(board_cards) + f"|{street}|{pot}|{bets[0]}|{bets[1]}"
            
            return info_set
            
        except Exception as e:
            print(f"Erreur lors de la création de l'info_set: {e}")
            return "As|0|0|0|0"
    
    def forward(self, 
               output, 
               target_action, 
               target_value,
               pokerkit_state=None,
               player_id=None,
               pot_size=0.0,
               stack_size=0.0,
               bet_size=0.0,
               policy_weight=1.0,
               value_weight=0.5):
        """
        Calcule la perte combinée utilisant à la fois la perte RL traditionnelle et le CFR.
        """
        # Calculer la perte RL traditionnelle (perte de politique et de valeur)
        policy_loss = F.cross_entropy(output, target_action)
        predicted_value = self.estimate_value(output, pot_size, stack_size, bet_size)
        value_loss = F.mse_loss(torch.tensor(predicted_value, dtype=torch.float32, device=self.device), target_value)
        
        # Perte RL traditionnelle
        rl_loss = policy_weight * policy_loss + value_weight * value_loss
        
        # Si nous avons l'état pokerkit, ajouter la perte CFR
        if pokerkit_state is not None and player_id is not None:
            try:
                cfr_loss = self.compute_cfr_loss(output, pokerkit_state, player_id)
                # Limiter la perte CFR pour éviter les explosions numériques
                cfr_loss = torch.clamp(cfr_loss, min=0.0, max=100.0)
                return rl_loss + self.cfr_weight * cfr_loss
            except Exception as e:
                print(f"Erreur dans le calcul de la perte CFR: {e}")
                # En cas d'erreur, retourner seulement la perte RL
                return rl_loss
        else:
            return rl_loss
    
    def compute_cfr_loss(self, output, pokerkit_state, player_id):
        """
        Calcule la composante de perte CFR.
        """
        # Créer l'info_set
        info_set = self.create_info_set_from_pokerkit(pokerkit_state, player_id)
        
        # Vérifier si l'info_set est valide et présent dans le solver CFR
        if info_set in self.cfr_solver.nodes:
            # Obtenir la stratégie CFR
            cfr_strategy = self.cfr_solver.get_strategy(info_set)
            
            # La stratégie CFR a 3 actions alors que le modèle BayesianHoldem a 4 actions
            cfr_strategy_expanded = torch.zeros(4, dtype=torch.float32, device=self.device)
            cfr_strategy_expanded[:3] = torch.tensor(cfr_strategy, dtype=torch.float32, device=self.device)
            
            # Convertir les sorties du modèle en probabilités
            output_probs = F.softmax(output, dim=0)
            
            # Ajouter un petit epsilon pour éviter log(0)
            epsilon = 1e-10
            cfr_strategy_expanded = cfr_strategy_expanded + epsilon
            output_probs = output_probs + epsilon
            
            # Normaliser
            cfr_strategy_expanded = cfr_strategy_expanded / cfr_strategy_expanded.sum()
            output_probs = output_probs / output_probs.sum()
            
            # KL divergence: sum(CFR_strat * log(CFR_strat / model_strat))
            kl_loss = torch.sum(cfr_strategy_expanded * torch.log(cfr_strategy_expanded / output_probs))
            
            return kl_loss
        else:
            # Info_set non trouvé dans le solver CFR
            # Retourner une petite perte constante
            return torch.tensor(0.1, device=self.device)
    
    def estimate_value(self, output, pot_size, stack_size, bet_size):
        """
        Estime la valeur d'un état étant donné la sortie du modèle.
        """
        action_probs = F.softmax(output, dim=0)
        
        value = 0.0
        
        # Pour chaque action possible, calculer Q(s,a) et pondérer par la probabilité d'action
        for action in range(4):
            # Calculer la récompense immédiate pour l'action
            if action == 0:  # Fold
                immediate_reward = -0.3 * pot_size
                future_value = stack_size
            elif action == 1:  # Check/Call
                immediate_reward = 0.0
                future_value = stack_size + 0.5 * pot_size
            elif action == 2:  # Bet/Raise
                immediate_reward = 0.2 * pot_size
                future_value = stack_size + pot_size + bet_size
            else:  # All-in
                immediate_reward = 0.1 * pot_size
                future_value = 2 * stack_size
            
            # Mettre à l'échelle la valeur future par le ratio de la pile
            max_stack = 20000  # Supposer une pile maximale de 20 000
            future_value *= (stack_size / max_stack)
            
            # Combiner les récompenses immédiates et futures avec un facteur d'actualisation
            discount_factor = 0.95
            q_value = immediate_reward + discount_factor * future_value
            
            # Pondérer par la probabilité d'action
            value += action_probs[action].item() * q_value
        
        return value


class BayesianHoldemWithCFR(nn.Module):
    """
    Extension de BayesianHoldem qui incorpore CFR pour l'entraînement amélioré.
    """
    
    def __init__(self, base_model, blueprint_path=None, cfr_weight=0.5):
        """
        Initialise le modèle avec entraînement basé sur CFR.
        
        Args:
            base_model: Instance du modèle BayesianHoldem de base
            blueprint_path: Chemin vers une stratégie CFR pré-entraînée
            cfr_weight: Poids de la composante CFR dans la perte totale
        """
        super(BayesianHoldemWithCFR, self).__init__()
        self.base_model = base_model
        self.cfr_loss = CFRLoss(blueprint_path=blueprint_path, cfr_weight=cfr_weight)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, action_representation, card_representation):
        return self.base_model(action_representation, card_representation)
    
    def predict_action(self, action_representation, card_representation):
        return self.base_model.predict_action(action_representation, card_representation)
    
    def train_step(self,
                  action_representation,
                  card_representation,
                  pot_size,
                  stack_size,
                  bet_size,
                  actual_return,
                  pokerkit_state=None,
                  player_id=None,
                  policy_weight=1.0,
                  value_weight=0.5):
        """
        Training step for the BayesianHoldem model with CFR loss.
        """
        try:
            # Mise à zéro des gradients
            self.optimizer.zero_grad()
            
            # Obtenir les cibles à partir du modèle de base
            target_action, target_value = self.base_model.get_training_targets(
                action_representation=action_representation,
                card_representation=card_representation,
                pot_size=pot_size,
                stack_size=stack_size,
                bet_size=bet_size,
                actual_return=actual_return
            )
            
            # S'assurer que le tenseur cible est sur le bon appareil
            target_value = target_value.to(self.device)
            
            # Passe avant
            output = self.forward(action_representation, card_representation)
            
            # Calculer la perte en utilisant la perte guidée par CFR
            total_loss = self.cfr_loss(
                output=output,
                target_action=target_action,
                target_value=target_value,
                pokerkit_state=pokerkit_state,
                player_id=player_id,
                pot_size=pot_size,
                stack_size=stack_size,
                bet_size=bet_size,
                policy_weight=policy_weight,
                value_weight=value_weight
            )
            
            # Passe arrière
            total_loss.backward()
            
            # Clipper les gradients pour éviter les gradients explosifs
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            # Mettre à jour les poids
            self.optimizer.step()
            
            # Obtenir les pertes individuelles pour la journalisation
            with torch.no_grad():
                policy_loss = F.cross_entropy(output, target_action)
                predicted_value = self.cfr_loss.estimate_value(output, pot_size, stack_size, bet_size)
                value_loss = F.mse_loss(torch.tensor(predicted_value, dtype=torch.float32, device=self.device), target_value)
            
            return total_loss.item(), policy_loss.item(), value_loss.item()
            
        except Exception as e:
            print(f"Erreur dans train_step: {e}")
            # En cas d'erreur, retourner des valeurs par défaut
            return 0.0, 0.0, 0.0