import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import os

class Utilities:
    @staticmethod
    def plot_training_stats(stats: Dict, save_path: str = None, show: bool = True):
        """
        Plot various training statistics including win rates and losses.
        
        Args:
            stats: Dictionary containing training statistics from run_training_session
            save_path: Optional path to save the plots
            show: Whether to display the plots
        """
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Statistics', fontsize=16)
        
        # Plot win rate over time
        total_games = np.arange(1, len(stats['total_losses']) + 1)
        win_rates = [stats['player0_wins'] / (stats['player0_wins'] + stats['player1_wins'])] * len(total_games)
        
        axes[0, 0].plot(total_games, win_rates, label='Win Rate')
        axes[0, 0].set_title('Win Rate Over Time')
        axes[0, 0].set_xlabel('Game Number')
        axes[0, 0].set_ylabel('Win Rate')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        # Plot total losses
        axes[0, 1].plot(total_games, stats['total_losses'], label='Total Loss')
        axes[0, 1].set_title('Total Loss Over Time')
        axes[0, 1].set_xlabel('Game Number')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        axes[0, 1].legend()
        
        # Plot policy and value losses
        axes[1, 0].plot(total_games, stats['policy_losses'], label='Policy Loss')
        axes[1, 0].plot(total_games, stats['value_losses'], label='Value Loss')
        axes[1, 0].set_title('Policy and Value Losses Over Time')
        axes[1, 0].set_xlabel('Game Number')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
        
        # Plot moving averages of losses
        window_size = 50
        total_loss_ma = np.convolve(stats['total_losses'], np.ones(window_size)/window_size, mode='valid')
        policy_loss_ma = np.convolve(stats['policy_losses'], np.ones(window_size)/window_size, mode='valid')
        value_loss_ma = np.convolve(stats['value_losses'], np.ones(window_size)/window_size, mode='valid')
        
        axes[1, 1].plot(total_games[window_size-1:], total_loss_ma, label='Total Loss MA')
        axes[1, 1].plot(total_games[window_size-1:], policy_loss_ma, label='Policy Loss MA')
        axes[1, 1].plot(total_games[window_size-1:], value_loss_ma, label='Value Loss MA')
        axes[1, 1].set_title('Moving Averages of Losses (Window Size = 50)')
        axes[1, 1].set_xlabel('Game Number')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True)
        axes[1, 1].legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot if path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        
        # Show plot if requested
        if show:
            plt.show()
        
        plt.close()

    @staticmethod
    def plot_win_distribution(stats: Dict, save_path: str = None, show: bool = True):
        """
        Plot the distribution of wins between players.
        
        Args:
            stats: Dictionary containing training statistics from run_training_session
            save_path: Optional path to save the plot
            show: Whether to display the plot
        """
        plt.figure(figsize=(8, 6))
        
        # Create pie chart
        labels = ['Player 0', 'Player 1']
        sizes = [stats['player0_wins'], stats['player1_wins']]
        colors = ['lightblue', 'lightcoral']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Win Distribution')
        
        # Save plot if path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        
        # Show plot if requested
        if show:
            plt.show()
        
        plt.close()

    @staticmethod
    def plot_loss_distribution(stats: Dict, save_path: str = None, show: bool = True):
        """
        Plot the distribution of losses.
        
        Args:
            stats: Dictionary containing training statistics from run_training_session
            save_path: Optional path to save the plot
            show: Whether to display the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Create histogram of losses
        plt.hist(stats['total_losses'], bins=50, alpha=0.7, label='Total Loss')
        plt.hist(stats['policy_losses'], bins=50, alpha=0.7, label='Policy Loss')
        plt.hist(stats['value_losses'], bins=50, alpha=0.7, label='Value Loss')
        
        plt.title('Distribution of Losses')
        plt.xlabel('Loss Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        
        # Save plot if path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        
        # Show plot if requested
        if show:
            plt.show()
        
        plt.close()

    @staticmethod
    def plot_all_stats(stats: Dict, save_dir: str = None, show: bool = True):
        """
        Plot all available statistics and save them to the specified directory.
        
        Args:
            stats: Dictionary containing training statistics from run_training_session
            save_dir: Optional directory to save the plots
            show: Whether to display the plots
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            Utilities.plot_training_stats(stats, os.path.join(save_dir, 'training_stats.png'), show)
            Utilities.plot_win_distribution(stats, os.path.join(save_dir, 'win_distribution.png'), show)
            Utilities.plot_loss_distribution(stats, os.path.join(save_dir, 'loss_distribution.png'), show)
        else:
            Utilities.plot_training_stats(stats, None, show)
            Utilities.plot_win_distribution(stats, None, show)
            Utilities.plot_loss_distribution(stats, None, show) 