import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from typing import Dict, List

class Utilities:
    @staticmethod
    def plot_training_stats(stats: dict, save_path: str = None, show: bool = True):
        """Plot comprehensive training statistics."""
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        fig.suptitle('Training Statistics', fontsize=16)
        
        # Plot win rate over sessions
        sessions = np.arange(1, len(stats['session_win_rates']) + 1)
        axes[0, 0].plot(sessions, stats['session_win_rates'], label='Win Rate')
        axes[0, 0].set_title('Win Rate Over Sessions')
        axes[0, 0].set_xlabel('Session')
        axes[0, 0].set_ylabel('Win Rate')
        axes[0, 0].grid(True)
        
        # Plot BB/hand over sessions
        axes[0, 1].plot(sessions, stats['session_bb_per_hand'], label='BB/Hand')
        axes[0, 1].set_title('BB/Hand Over Sessions')
        axes[0, 1].set_xlabel('Session')
        axes[0, 1].set_ylabel('BB/Hand')
        axes[0, 1].grid(True)
        
        # Plot Elo ratings over sessions
        axes[1, 0].plot(sessions, stats['session_player0_elo'], label='Player 0 Elo')
        axes[1, 0].plot(sessions, stats['session_player1_elo'], label='Player 1 Elo')
        axes[1, 0].set_title('Elo Ratings Over Sessions')
        axes[1, 0].set_xlabel('Session')
        axes[1, 0].set_ylabel('Elo Rating')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot total losses over sessions
        axes[1, 1].plot(sessions, stats['session_total_losses'], label='Total Loss')
        axes[1, 1].set_title('Total Loss Over Sessions')
        axes[1, 1].set_xlabel('Session')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True)
        
        # Plot policy and value losses separately
        axes[2, 0].plot(sessions, stats['session_policy_losses'], label='Policy Loss')
        axes[2, 0].plot(sessions, stats['session_value_losses'], label='Value Loss')
        axes[2, 0].set_title('Policy and Value Losses Over Sessions')
        axes[2, 0].set_xlabel('Session')
        axes[2, 0].set_ylabel('Loss')
        axes[2, 0].legend()
        axes[2, 0].grid(True)
        
        # Plot hands played per session
        axes[2, 1].plot(sessions, stats['session_hands_played'], label='Hands Played')
        axes[2, 1].set_title('Hands Played Per Session')
        axes[2, 1].set_xlabel('Session')
        axes[2, 1].set_ylabel('Number of Hands')
        axes[2, 1].grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()

    @staticmethod
    def plot_win_distribution(stats: dict, save_path: str = None, show: bool = True):
        """Plot distribution of wins and losses."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('Win/Loss Distribution', fontsize=16)
        
        # Plot win/loss pie chart
        wins = stats['total_player0_wins']
        losses = stats['total_player1_wins']
        total = wins + losses
        axes[0].pie([wins, losses], labels=['Wins', 'Losses'], autopct='%1.1f%%')
        axes[0].set_title(f'Win/Loss Ratio (Total Games: {total})')
        
        # Plot win rate over time (rolling average)
        sessions = np.arange(1, len(stats['session_win_rates']) + 1)
        rolling_avg = pd.Series(stats['session_win_rates']).rolling(window=5).mean()
        axes[1].plot(sessions, stats['session_win_rates'], alpha=0.3, label='Raw')
        axes[1].plot(sessions, rolling_avg, label='Rolling Avg (5)')
        axes[1].set_title('Win Rate Trend')
        axes[1].set_xlabel('Session')
        axes[1].set_ylabel('Win Rate')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()

    @staticmethod
    def plot_loss_distribution(stats: dict, save_path: str = None, show: bool = True):
        """Plot distribution of losses."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Loss Distribution', fontsize=16)
        
        # Plot total loss distribution
        axes[0, 0].hist(stats['session_total_losses'], bins=20)
        axes[0, 0].set_title('Total Loss Distribution')
        axes[0, 0].set_xlabel('Loss')
        axes[0, 0].set_ylabel('Frequency')
        
        # Plot policy loss distribution
        axes[0, 1].hist(stats['session_policy_losses'], bins=20)
        axes[0, 1].set_title('Policy Loss Distribution')
        axes[0, 1].set_xlabel('Loss')
        axes[0, 1].set_ylabel('Frequency')
        
        # Plot value loss distribution
        axes[1, 0].hist(stats['session_value_losses'], bins=20)
        axes[1, 0].set_title('Value Loss Distribution')
        axes[1, 0].set_xlabel('Loss')
        axes[1, 0].set_ylabel('Frequency')
        
        # Plot loss correlation
        axes[1, 1].scatter(stats['session_policy_losses'], stats['session_value_losses'])
        axes[1, 1].set_title('Policy vs Value Loss Correlation')
        axes[1, 1].set_xlabel('Policy Loss')
        axes[1, 1].set_ylabel('Value Loss')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()

    @staticmethod
    def plot_learning_progress(stats: dict, save_path: str = None, show: bool = True):
        """Plot learning progress metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Learning Progress', fontsize=16)
        
        # Plot BB/hand trend
        sessions = np.arange(1, len(stats['session_bb_per_hand']) + 1)
        rolling_avg = pd.Series(stats['session_bb_per_hand']).rolling(window=5).mean()
        axes[0, 0].plot(sessions, stats['session_bb_per_hand'], alpha=0.3, label='Raw')
        axes[0, 0].plot(sessions, rolling_avg, label='Rolling Avg (5)')
        axes[0, 0].set_title('BB/Hand Trend')
        axes[0, 0].set_xlabel('Session')
        axes[0, 0].set_ylabel('BB/Hand')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot Elo rating trend
        axes[0, 1].plot(sessions, stats['session_player0_elo'], label='Player 0')
        axes[0, 1].plot(sessions, stats['session_player1_elo'], label='Player 1')
        axes[0, 1].set_title('Elo Rating Trend')
        axes[0, 1].set_xlabel('Session')
        axes[0, 1].set_ylabel('Elo Rating')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot hands played trend
        axes[1, 0].plot(sessions, stats['session_hands_played'])
        axes[1, 0].set_title('Hands Played Trend')
        axes[1, 0].set_xlabel('Session')
        axes[1, 0].set_ylabel('Number of Hands')
        axes[1, 0].grid(True)
        
        # Plot session duration
        axes[1, 1].plot(sessions, stats['session_durations'])
        axes[1, 1].set_title('Session Duration Trend')
        axes[1, 1].set_xlabel('Session')
        axes[1, 1].set_ylabel('Duration (seconds)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()

    @staticmethod
    def plot_action_distribution(stats: dict, save_path: str = None, show: bool = True):
        """Plot the distribution of actions across sessions."""
        if 'session_actions' not in stats:
            print("No action data available to plot")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Action Distribution Across Sessions', fontsize=16)
        
        action_names = ['Fold', 'Check/Call', 'Bet/Raise', 'All-in']
        sessions = np.arange(1, len(stats['session_actions']) + 1)
        
        # Plot each action type
        for i, (ax, action_name) in enumerate(zip(axes.flat, action_names)):
            action_counts = [session[i] for session in stats['session_actions']]
            ax.plot(sessions, action_counts, label=f'{action_name} Count')
            ax.set_title(f'{action_name} Distribution')
            ax.set_xlabel('Session')
            ax.set_ylabel('Count')
            ax.grid(True)
            ax.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()

    @staticmethod
    def plot_all_stats(stats: dict, save_dir: str = None, show: bool = True):
        """Plot all statistics."""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            Utilities.plot_training_stats(stats, os.path.join(save_dir, 'training_stats.png'), show)
            Utilities.plot_win_distribution(stats, os.path.join(save_dir, 'win_distribution.png'), show)
            Utilities.plot_loss_distribution(stats, os.path.join(save_dir, 'loss_distribution.png'), show)
            Utilities.plot_learning_progress(stats, os.path.join(save_dir, 'learning_progress.png'), show)
            Utilities.plot_action_distribution(stats, os.path.join(save_dir, 'action_distribution.png'), show)
        else:
            Utilities.plot_training_stats(stats, show=show)
            Utilities.plot_win_distribution(stats, show=show)
            Utilities.plot_loss_distribution(stats, show=show)
            Utilities.plot_learning_progress(stats, show=show)
            Utilities.plot_action_distribution(stats, show=show) 