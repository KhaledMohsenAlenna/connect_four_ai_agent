import numpy as np
import time
import copy
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyBboxPatch
import json

class Connect4Game:
    """
    Connect 4 Game Environment
    
    Game State Representation:
    - Board: 6x7 numpy array (6 rows, 7 columns)
    - Cell values: 0 (empty), 1 (Player 1), 2 (Player 2)
    - Current player: 1 or 2
    """
    
    def __init__(self):
        self.ROWS = 6
        self.COLS = 7
        self.CONNECT = 4  # Number of pieces to connect to win
        self.board = np.zeros((self.ROWS, self.COLS), dtype=int)
        self.current_player = 1
        
    def get_initial_state(self):
        """Returns the initial game state (empty board)"""
        return np.zeros((self.ROWS, self.COLS), dtype=int)
    
    def get_valid_moves(self, board):
        """
        Returns list of valid column indices where a piece can be dropped
        A column is valid if the top row is empty
        """
        valid_moves = []
        for col in range(self.COLS):
            if board[0][col] == 0:  # Top row is empty
                valid_moves.append(col)
        return valid_moves
    
    def apply_move(self, board, col, player):
        """
        Transition Function: Applies a move and returns new state
        Drops a piece in the specified column for the given player
        """
        new_board = board.copy()
        # Find the lowest empty row in the column
        for row in range(self.ROWS - 1, -1, -1):
            if new_board[row][col] == 0:
                new_board[row][col] = player
                return new_board
        return new_board  # Column is full (shouldn't happen with valid moves)
    
    def check_winner(self, board, player):
        """
        Terminal State Check: Determines if player has won
        Checks horizontal, vertical, and diagonal connections
        """
        # Check horizontal
        for row in range(self.ROWS):
            for col in range(self.COLS - 3):
                if all(board[row][col + i] == player for i in range(4)):
                    return True
        
        # Check vertical
        for row in range(self.ROWS - 3):
            for col in range(self.COLS):
                if all(board[row + i][col] == player for i in range(4)):
                    return True
        
        # Check diagonal (bottom-left to top-right)
        for row in range(3, self.ROWS):
            for col in range(self.COLS - 3):
                if all(board[row - i][col + i] == player for i in range(4)):
                    return True
        
        # Check diagonal (top-left to bottom-right)
        for row in range(self.ROWS - 3):
            for col in range(self.COLS - 3):
                if all(board[row + i][col + i] == player for i in range(4)):
                    return True
        
        return False
    
    def is_terminal(self, board):
        """
        Checks if the game has reached a terminal state
        Returns: (is_terminal, winner)
        winner: 1 or 2 if there's a winner, 0 for draw, None for ongoing
        """
        if self.check_winner(board, 1):
            return True, 1
        if self.check_winner(board, 2):
            return True, 2
        if len(self.get_valid_moves(board)) == 0:
            return True, 0  # Draw
        return False, None
    
    def print_board(self, board):
        """Displays the current board state"""
        print("\n  0 1 2 3 4 5 6")
        print(" +" + "-" * 15 + "+")
        for row in board:
            print(" |", end="")
            for cell in row:
                if cell == 0:
                    print(" .", end="")
                elif cell == 1:
                    print(" X", end="")
                else:
                    print(" O", end="")
            print(" |")
        print(" +" + "-" * 15 + "+")


class PerformanceMetrics:
    """Tracks performance metrics for algorithm comparison"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.nodes_expanded = 0
        self.max_depth_reached = 0
        self.time_taken = 0
        self.pruning_count = 0  # For Alpha-Beta
        self.move_sequence = []


class MinimaxAgent:
    """
    Minimax Algorithm Implementation
    
    The Minimax algorithm explores the entire game tree up to a specified depth,
    alternating between maximizing and minimizing players.
    """
    
    def __init__(self, game: Connect4Game, max_depth=4):
        self.game = game
        self.max_depth = max_depth
        self.metrics = PerformanceMetrics()
    
    def evaluate_board(self, board, player):
        """
        Evaluation Function (Heuristic)
        
        Scores the board state from the perspective of the given player.
        Higher scores are better for the player.
        
        Components:
        1. Win/Loss detection: ±10000
        2. Three-in-a-row with empty space: +100/-100
        3. Two-in-a-row with two empty spaces: +10/-10
        4. Center column control: +3 per piece
        """
        terminal, winner = self.game.is_terminal(board)
        
        if terminal:
            if winner == player:
                return 10000
            elif winner == 0:
                return 0  # Draw
            else:
                return -10000
        
        score = 0
        opponent = 3 - player  # If player is 1, opponent is 2, and vice versa
        
        # Evaluate all possible windows of 4
        # Horizontal
        for row in range(self.game.ROWS):
            for col in range(self.game.COLS - 3):
                window = [board[row][col + i] for i in range(4)]
                score += self.evaluate_window(window, player, opponent)
        
        # Vertical
        for row in range(self.game.ROWS - 3):
            for col in range(self.game.COLS):
                window = [board[row + i][col] for i in range(4)]
                score += self.evaluate_window(window, player, opponent)
        
        # Diagonal (/)
        for row in range(3, self.game.ROWS):
            for col in range(self.game.COLS - 3):
                window = [board[row - i][col + i] for i in range(4)]
                score += self.evaluate_window(window, player, opponent)
        
        # Diagonal (\)
        for row in range(self.game.ROWS - 3):
            for col in range(self.game.COLS - 3):
                window = [board[row + i][col + i] for i in range(4)]
                score += self.evaluate_window(window, player, opponent)
        
        # Bonus for center column control
        center_col = self.game.COLS // 2
        center_count = sum(1 for row in range(self.game.ROWS) if board[row][center_col] == player)
        score += center_count * 3
        
        return score
    
    def evaluate_window(self, window, player, opponent):
        """Evaluates a window of 4 positions"""
        score = 0
        player_count = window.count(player)
        opponent_count = window.count(opponent)
        empty_count = window.count(0)
        
        # Player's potential
        if player_count == 3 and empty_count == 1:
            score += 100
        elif player_count == 2 and empty_count == 2:
            score += 10
        
        # Opponent's threats
        if opponent_count == 3 and empty_count == 1:
            score -= 100
        elif opponent_count == 2 and empty_count == 2:
            score -= 10
        
        return score
    
    def minimax(self, board, depth, maximizing_player, player):
        """
        Minimax algorithm implementation
        
        Args:
            board: Current board state
            depth: Current depth in the game tree
            maximizing_player: True if maximizing, False if minimizing
            player: The player we're evaluating for (1 or 2)
        
        Returns:
            Best score for the current player
        """
        self.metrics.nodes_expanded += 1
        self.metrics.max_depth_reached = max(self.metrics.max_depth_reached, 
                                             self.max_depth - depth)
        
        terminal, winner = self.game.is_terminal(board)
        
        # Base cases
        if depth == 0 or terminal:
            return self.evaluate_board(board, player)
        
        valid_moves = self.game.get_valid_moves(board)
        
        if maximizing_player:
            max_eval = float('-inf')
            for col in valid_moves:
                new_board = self.game.apply_move(board, col, player)
                eval_score = self.minimax(new_board, depth - 1, False, player)
                max_eval = max(max_eval, eval_score)
            return max_eval
        else:
            min_eval = float('inf')
            opponent = 3 - player
            for col in valid_moves:
                new_board = self.game.apply_move(board, col, opponent)
                eval_score = self.minimax(new_board, depth - 1, True, player)
                min_eval = min(min_eval, eval_score)
            return min_eval
    
    def get_best_move(self, board, player):
        """
        Determines the best move using Minimax
        
        Returns: (best_column, metrics)
        """
        self.metrics.reset()
        start_time = time.time()
        
        valid_moves = self.game.get_valid_moves(board)
        best_move = valid_moves[0]
        best_score = float('-inf')
        
        for col in valid_moves:
            new_board = self.game.apply_move(board, col, player)
            score = self.minimax(new_board, self.max_depth - 1, False, player)
            
            if score > best_score:
                best_score = score
                best_move = col
        
        self.metrics.time_taken = time.time() - start_time
        return best_move, self.metrics


class AlphaBetaAgent:
    """
    Alpha-Beta Pruning Algorithm Implementation
    
    Alpha-Beta pruning is an optimization of Minimax that prunes branches
    that cannot influence the final decision, significantly reducing nodes explored.
    """
    
    def __init__(self, game: Connect4Game, max_depth=4):
        self.game = game
        self.max_depth = max_depth
        self.metrics = PerformanceMetrics()
    
    def evaluate_board(self, board, player):
        """Same evaluation function as Minimax (for fair comparison)"""
        terminal, winner = self.game.is_terminal(board)
        
        if terminal:
            if winner == player:
                return 10000
            elif winner == 0:
                return 0
            else:
                return -10000
        
        score = 0
        opponent = 3 - player
        
        # Horizontal
        for row in range(self.game.ROWS):
            for col in range(self.game.COLS - 3):
                window = [board[row][col + i] for i in range(4)]
                score += self.evaluate_window(window, player, opponent)
        
        # Vertical
        for row in range(self.game.ROWS - 3):
            for col in range(self.game.COLS):
                window = [board[row + i][col] for i in range(4)]
                score += self.evaluate_window(window, player, opponent)
        
        # Diagonal (/)
        for row in range(3, self.game.ROWS):
            for col in range(self.game.COLS - 3):
                window = [board[row - i][col + i] for i in range(4)]
                score += self.evaluate_window(window, player, opponent)
        
        # Diagonal (\)
        for row in range(self.game.ROWS - 3):
            for col in range(self.game.COLS - 3):
                window = [board[row + i][col + i] for i in range(4)]
                score += self.evaluate_window(window, player, opponent)
        
        # Center control
        center_col = self.game.COLS // 2
        center_count = sum(1 for row in range(self.game.ROWS) if board[row][center_col] == player)
        score += center_count * 3
        
        return score
    
    def evaluate_window(self, window, player, opponent):
        """Evaluates a window of 4 positions"""
        score = 0
        player_count = window.count(player)
        opponent_count = window.count(opponent)
        empty_count = window.count(0)
        
        if player_count == 3 and empty_count == 1:
            score += 100
        elif player_count == 2 and empty_count == 2:
            score += 10
        
        if opponent_count == 3 and empty_count == 1:
            score -= 100
        elif opponent_count == 2 and empty_count == 2:
            score -= 10
        
        return score
    
    def alpha_beta(self, board, depth, alpha, beta, maximizing_player, player):
        """
        Alpha-Beta Pruning algorithm implementation
        
        Args:
            board: Current board state
            depth: Current depth in the game tree
            alpha: Best value for maximizer found so far
            beta: Best value for minimizer found so far
            maximizing_player: True if maximizing, False if minimizing
            player: The player we're evaluating for
        
        Returns:
            Best score for the current player
        """
        self.metrics.nodes_expanded += 1
        self.metrics.max_depth_reached = max(self.metrics.max_depth_reached, 
                                             self.max_depth - depth)
        
        terminal, winner = self.game.is_terminal(board)
        
        # Base cases
        if depth == 0 or terminal:
            return self.evaluate_board(board, player)
        
        valid_moves = self.game.get_valid_moves(board)
        
        if maximizing_player:
            max_eval = float('-inf')
            for col in valid_moves:
                new_board = self.game.apply_move(board, col, player)
                eval_score = self.alpha_beta(new_board, depth - 1, alpha, beta, False, player)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                
                # Beta cutoff (pruning)
                if beta <= alpha:
                    self.metrics.pruning_count += 1
                    break
            return max_eval
        else:
            min_eval = float('inf')
            opponent = 3 - player
            for col in valid_moves:
                new_board = self.game.apply_move(board, col, opponent)
                eval_score = self.alpha_beta(new_board, depth - 1, alpha, beta, True, player)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                # Alpha cutoff (pruning)
                if beta <= alpha:
                    self.metrics.pruning_count += 1
                    break
            return min_eval
    
    def get_best_move(self, board, player):
        """
        Determines the best move using Alpha-Beta Pruning
        
        Returns: (best_column, metrics)
        """
        self.metrics.reset()
        start_time = time.time()
        
        valid_moves = self.game.get_valid_moves(board)
        best_move = valid_moves[0]
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for col in valid_moves:
            new_board = self.game.apply_move(board, col, player)
            score = self.alpha_beta(new_board, self.max_depth - 1, alpha, beta, False, player)
            
            if score > best_score:
                best_score = score
                best_move = col
            
            alpha = max(alpha, score)
        
        self.metrics.time_taken = time.time() - start_time
        return best_move, self.metrics


def compare_algorithms():
    """
    Runs experiments comparing Minimax and Alpha-Beta performance
    """
    print("=" * 70)
    print("CONNECT 4 - MINIMAX VS ALPHA-BETA PERFORMANCE COMPARISON")
    print("=" * 70)
    
    results = {
        'minimax': {'time': [], 'nodes': [], 'depth': [], 'pruning': []},
        'alphabeta': {'time': [], 'nodes': [], 'depth': [], 'pruning': []}
    }
    
    depths = [3, 4, 5]
    
    for depth in depths:
        print(f"\n{'=' * 70}")
        print(f"Testing at Depth: {depth}")
        print(f"{'=' * 70}")
        
        # Run 3 games at each depth
        for game_num in range(3):
            print(f"\n--- Game {game_num + 1} ---")
            
            game = Connect4Game()
            minimax_agent = MinimaxAgent(game, max_depth=depth)
            alphabeta_agent = AlphaBetaAgent(game, max_depth=depth)
            
            board = game.get_initial_state()
            
            # Make a few random moves to create a non-trivial position
            for _ in range(3):
                valid_moves = game.get_valid_moves(board)
                if valid_moves:
                    import random
                    col = random.choice(valid_moves)
                    player = 1 if _ % 2 == 0 else 2
                    board = game.apply_move(board, col, player)
            
            print("\nStarting Board State:")
            game.print_board(board)
            
            # Test Minimax
            print("\n--- MINIMAX ---")
            move_mm, metrics_mm = minimax_agent.get_best_move(board, 1)
            print(f"Best Move: Column {move_mm}")
            print(f"Time Taken: {metrics_mm.time_taken:.4f} seconds")
            print(f"Nodes Expanded: {metrics_mm.nodes_expanded}")
            print(f"Max Depth Reached: {metrics_mm.max_depth_reached}")
            
            results['minimax']['time'].append(metrics_mm.time_taken)
            results['minimax']['nodes'].append(metrics_mm.nodes_expanded)
            results['minimax']['depth'].append(metrics_mm.max_depth_reached)
            
            # Test Alpha-Beta
            print("\n--- ALPHA-BETA PRUNING ---")
            move_ab, metrics_ab = alphabeta_agent.get_best_move(board, 1)
            print(f"Best Move: Column {move_ab}")
            print(f"Time Taken: {metrics_ab.time_taken:.4f} seconds")
            print(f"Nodes Expanded: {metrics_ab.nodes_expanded}")
            print(f"Max Depth Reached: {metrics_ab.max_depth_reached}")
            print(f"Pruning Events: {metrics_ab.pruning_count}")
            
            results['alphabeta']['time'].append(metrics_ab.time_taken)
            results['alphabeta']['nodes'].append(metrics_ab.nodes_expanded)
            results['alphabeta']['depth'].append(metrics_ab.max_depth_reached)
            results['alphabeta']['pruning'].append(metrics_ab.pruning_count)
            
            # Calculate improvements
            time_improvement = ((metrics_mm.time_taken - metrics_ab.time_taken) / 
                               metrics_mm.time_taken * 100)
            nodes_improvement = ((metrics_mm.nodes_expanded - metrics_ab.nodes_expanded) / 
                                metrics_mm.nodes_expanded * 100)
            
            print(f"\n--- IMPROVEMENT ---")
            print(f"Time Reduction: {time_improvement:.2f}%")
            print(f"Nodes Reduction: {nodes_improvement:.2f}%")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"\n{'Criterion':<25} {'Minimax':<20} {'Alpha-Beta':<20}")
    print("-" * 70)
    print(f"{'Avg Time (seconds)':<25} {np.mean(results['minimax']['time']):<20.4f} "
          f"{np.mean(results['alphabeta']['time']):<20.4f}")
    print(f"{'Avg Nodes Expanded':<25} {np.mean(results['minimax']['nodes']):<20.0f} "
          f"{np.mean(results['alphabeta']['nodes']):<20.0f}")
    print(f"{'Avg Depth Reached':<25} {np.mean(results['minimax']['depth']):<20.1f} "
          f"{np.mean(results['alphabeta']['depth']):<20.1f}")
    print(f"{'Avg Pruning Events':<25} {'N/A':<20} "
          f"{np.mean(results['alphabeta']['pruning']):<20.0f}")
    
    avg_time_improvement = ((np.mean(results['minimax']['time']) - 
                            np.mean(results['alphabeta']['time'])) / 
                           np.mean(results['minimax']['time']) * 100)
    avg_nodes_improvement = ((np.mean(results['minimax']['nodes']) - 
                             np.mean(results['alphabeta']['nodes'])) / 
                            np.mean(results['minimax']['nodes']) * 100)
    
    print(f"\n{'Overall Time Improvement':<25} {avg_time_improvement:.2f}%")
    print(f"{'Overall Nodes Improvement':<25} {avg_nodes_improvement:.2f}%")
    
    # Generate visualizations
    print("\n" + "=" * 70)
    print("Generating visualizations...")
    print("=" * 70)
    visualize_results(results)
    print("Visualizations saved successfully!")


def visualize_results(results):
    """
    Creates comprehensive visualizations for the report
    """
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Time Comparison Bar Chart
    ax1 = plt.subplot(2, 3, 1)
    algorithms = ['Minimax', 'Alpha-Beta']
    avg_times = [np.mean(results['minimax']['time']), 
                 np.mean(results['alphabeta']['time'])]
    colors = ['#FF6B6B', '#4ECDC4']
    bars = ax1.bar(algorithms, avg_times, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Average Time Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}s', ha='center', va='bottom', fontweight='bold')
    
    # 2. Nodes Expanded Comparison
    ax2 = plt.subplot(2, 3, 2)
    avg_nodes = [np.mean(results['minimax']['nodes']), 
                 np.mean(results['alphabeta']['nodes'])]
    bars = ax2.bar(algorithms, avg_nodes, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Nodes Expanded', fontsize=12, fontweight='bold')
    ax2.set_title('Average Nodes Expanded', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Efficiency Improvement Pie Chart
    ax3 = plt.subplot(2, 3, 3)
    time_improvement = ((np.mean(results['minimax']['time']) - 
                        np.mean(results['alphabeta']['time'])) / 
                       np.mean(results['minimax']['time']) * 100)
    nodes_improvement = ((np.mean(results['minimax']['nodes']) - 
                         np.mean(results['alphabeta']['nodes'])) / 
                        np.mean(results['minimax']['nodes']) * 100)
    
    improvements = [time_improvement, nodes_improvement]
    labels = [f'Time\nReduction\n{time_improvement:.1f}%', 
              f'Nodes\nReduction\n{nodes_improvement:.1f}%']
    colors_pie = ['#95E1D3', '#F38181']
    
    ax3.pie(improvements, labels=labels, colors=colors_pie, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax3.set_title('Alpha-Beta Efficiency Gains', fontsize=14, fontweight='bold')
    
    # 4. Time per Depth Level
    ax4 = plt.subplot(2, 3, 4)
    depths = [3, 4, 5]
    mm_times = [np.mean(results['minimax']['time'][i:i+3]) for i in range(0, 9, 3)]
    ab_times = [np.mean(results['alphabeta']['time'][i:i+3]) for i in range(0, 9, 3)]
    
    x = np.arange(len(depths))
    width = 0.35
    
    ax4.bar(x - width/2, mm_times, width, label='Minimax', color='#FF6B6B', 
            alpha=0.8, edgecolor='black')
    ax4.bar(x + width/2, ab_times, width, label='Alpha-Beta', color='#4ECDC4', 
            alpha=0.8, edgecolor='black')
    
    ax4.set_xlabel('Search Depth', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax4.set_title('Time vs Search Depth', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(depths)
    ax4.legend(fontsize=10)
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Nodes Expanded per Depth Level
    ax5 = plt.subplot(2, 3, 5)
    mm_nodes = [np.mean(results['minimax']['nodes'][i:i+3]) for i in range(0, 9, 3)]
    ab_nodes = [np.mean(results['alphabeta']['nodes'][i:i+3]) for i in range(0, 9, 3)]
    
    ax5.plot(depths, mm_nodes, marker='o', linewidth=2, markersize=8, 
             label='Minimax', color='#FF6B6B')
    ax5.plot(depths, ab_nodes, marker='s', linewidth=2, markersize=8, 
             label='Alpha-Beta', color='#4ECDC4')
    
    ax5.set_xlabel('Search Depth', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Nodes Expanded', fontsize=12, fontweight='bold')
    ax5.set_title('Nodes Expanded vs Depth', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # 6. Pruning Effectiveness
    ax6 = plt.subplot(2, 3, 6)
    avg_pruning_per_depth = [np.mean(results['alphabeta']['pruning'][i:i+3]) 
                             for i in range(0, 9, 3)]
    
    ax6.bar(depths, avg_pruning_per_depth, color='#AA96DA', alpha=0.8, 
            edgecolor='black')
    ax6.set_xlabel('Search Depth', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Pruning Events', fontsize=12, fontweight='bold')
    ax6.set_title('Alpha-Beta Pruning Events', fontsize=14, fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(avg_pruning_per_depth):
        ax6.text(depths[i], v, f'{int(v)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: algorithm_comparison.png")
    
    # Create a separate detailed comparison table visualization
    create_comparison_table(results)
    
    # Create game tree visualization example
    create_game_tree_visualization()
    
    plt.show()


def create_comparison_table(results):
    """
    Creates a detailed comparison table as an image
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Calculate statistics
    mm_avg_time = np.mean(results['minimax']['time'])
    ab_avg_time = np.mean(results['alphabeta']['time'])
    mm_avg_nodes = np.mean(results['minimax']['nodes'])
    ab_avg_nodes = np.mean(results['alphabeta']['nodes'])
    ab_avg_pruning = np.mean(results['alphabeta']['pruning'])
    
    time_improvement = ((mm_avg_time - ab_avg_time) / mm_avg_time * 100)
    nodes_improvement = ((mm_avg_nodes - ab_avg_nodes) / mm_avg_nodes * 100)
    
    # Create table data
    table_data = [
        ['Criterion', 'Minimax', 'Alpha-Beta', 'Improvement'],
        ['Time (seconds)', f'{mm_avg_time:.4f}', f'{ab_avg_time:.4f}', 
         f'{time_improvement:.2f}%'],
        ['Nodes Expanded', f'{int(mm_avg_nodes)}', f'{int(ab_avg_nodes)}', 
         f'{nodes_improvement:.2f}%'],
        ['Max Depth Reached', f'{np.mean(results["minimax"]["depth"]):.1f}', 
         f'{np.mean(results["alphabeta"]["depth"]):.1f}', '-'],
        ['Pruning Events', 'N/A', f'{int(ab_avg_pruning)}', '-'],
        ['Optimality', 'Optimal', 'Optimal', '-'],
        ['Efficiency', 'Baseline', 'Highly Efficient', f'{nodes_improvement:.1f}% faster']
    ]
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style the header row
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#4A90E2')
        cell.set_text_props(weight='bold', color='white', fontsize=12)
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(4):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#F0F0F0')
            cell.set_text_props(fontsize=11)
    
    plt.title('Minimax vs Alpha-Beta: Detailed Performance Comparison', 
              fontsize=16, fontweight='bold', pad=20)
    plt.savefig('comparison_table.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: comparison_table.png")


def create_game_tree_visualization():
    """
    Creates a simplified game tree showing pruning
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Minimax tree (all nodes explored)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Minimax Algorithm\n(All Nodes Explored)', 
                  fontsize=14, fontweight='bold')
    
    # Draw nodes for Minimax
    # Root
    root = Circle((5, 9), 0.3, color='#4ECDC4', ec='black', linewidth=2)
    ax1.add_patch(root)
    ax1.text(5, 9, 'MAX', ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Level 1 (MIN)
    level1_x = [2, 5, 8]
    for x in level1_x:
        circle = Circle((x, 7), 0.3, color='#FF6B6B', ec='black', linewidth=2)
        ax1.add_patch(circle)
        ax1.text(x, 7, 'MIN', ha='center', va='center', fontweight='bold', fontsize=8)
        ax1.plot([5, x], [8.7, 7.3], 'k-', linewidth=1.5)
    
    # Level 2 (MAX)
    level2_positions = [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5), (7, 5), (8, 5), (9, 5)]
    for x, y in level2_positions:
        circle = Circle((x, y), 0.25, color='#95E1D3', ec='black', linewidth=1.5)
        ax1.add_patch(circle)
        ax1.text(x, y, 'MAX', ha='center', va='center', fontsize=7)
    
    # Connect level 1 to level 2
    for i, parent_x in enumerate(level1_x):
        start_idx = i * 3
        for j in range(3):
            child_x = level2_positions[start_idx + j][0]
            ax1.plot([parent_x, child_x], [6.7, 5.3], 'k-', linewidth=1, alpha=0.7)
    
    # Add evaluation values
    values = [3, 5, 2, 7, 4, 6, 8, 3, 5]
    for i, (x, y) in enumerate(level2_positions):
        ax1.text(x, y - 0.7, str(values[i]), ha='center', fontsize=9, 
                fontweight='bold', color='blue')
    
    # Add node count
    ax1.text(5, 1, 'Nodes Explored: 13', ha='center', fontsize=12, 
            fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    # Alpha-Beta tree (pruned nodes)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Alpha-Beta Pruning\n(Pruned Branches Marked)', 
                  fontsize=14, fontweight='bold')
    
    # Root
    root2 = Circle((5, 9), 0.3, color='#4ECDC4', ec='black', linewidth=2)
    ax2.add_patch(root2)
    ax2.text(5, 9, 'MAX', ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Level 1
    for x in level1_x:
        circle = Circle((x, 7), 0.3, color='#FF6B6B', ec='black', linewidth=2)
        ax2.add_patch(circle)
        ax2.text(x, 7, 'MIN', ha='center', va='center', fontweight='bold', fontsize=8)
        ax2.plot([5, x], [8.7, 7.3], 'k-', linewidth=1.5)
    
    # Level 2 - some explored, some pruned
    explored = [True, True, True, True, False, False, True, False, False]
    for i, (x, y) in enumerate(level2_positions):
        if explored[i]:
            circle = Circle((x, y), 0.25, color='#95E1D3', ec='black', linewidth=1.5)
            ax2.add_patch(circle)
            ax2.text(x, y, 'MAX', ha='center', va='center', fontsize=7)
            ax2.text(x, y - 0.7, str(values[i]), ha='center', fontsize=9, 
                    fontweight='bold', color='blue')
        else:
            # Pruned nodes
            circle = Circle((x, y), 0.25, color='lightgray', ec='red', 
                           linewidth=2, linestyle='--')
            ax2.add_patch(circle)
            ax2.text(x, y, 'X', ha='center', va='center', fontsize=10, 
                    color='red', fontweight='bold')
    
    # Connect with different styles
    for i, parent_x in enumerate(level1_x):
        start_idx = i * 3
        for j in range(3):
            child_idx = start_idx + j
            child_x = level2_positions[child_idx][0]
            if explored[child_idx]:
                ax2.plot([parent_x, child_x], [6.7, 5.3], 'k-', linewidth=1, alpha=0.7)
            else:
                ax2.plot([parent_x, child_x], [6.7, 5.3], 'r--', linewidth=1, 
                        alpha=0.5)
    
    # Add annotations
    ax2.text(5, 1.5, 'Nodes Explored: 8', ha='center', fontsize=12, 
            fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgreen'))
    ax2.text(5, 0.5, 'Pruned: 5 nodes (38% reduction)', ha='center', 
            fontsize=11, fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig('game_tree_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: game_tree_comparison.png")


def visualize_board_state(board, title="Connect 4 Board State"):
    """
    Creates a visual representation of the Connect 4 board
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Draw the blue board
    board_rect = FancyBboxPatch((0, 0), 7, 6, boxstyle="round,pad=0.1", 
                                facecolor='#0066CC', edgecolor='black', linewidth=3)
    ax.add_patch(board_rect)
    
    # Draw the grid and pieces
    for row in range(6):
        for col in range(7):
            # Draw cell circle
            circle = Circle((col + 0.5, 5.5 - row), 0.4, facecolor='white', 
                          edgecolor='#004080', linewidth=2)
            ax.add_patch(circle)
            
            # Draw pieces
            if board[row][col] == 1:
                piece = Circle((col + 0.5, 5.5 - row), 0.35, facecolor='#FF4444', 
                             edgecolor='darkred', linewidth=2)
                ax.add_patch(piece)
            elif board[row][col] == 2:
                piece = Circle((col + 0.5, 5.5 - row), 0.35, facecolor='#FFEB3B', 
                             edgecolor='#FBC02D', linewidth=2)
                ax.add_patch(piece)
    
    # Add column numbers
    for col in range(7):
        ax.text(col + 0.5, -0.5, str(col), ha='center', va='center', 
               fontsize=14, fontweight='bold')
    
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-1, 6.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    red_patch = mpatches.Patch(color='#FF4444', label='Player 1 (X)')
    yellow_patch = mpatches.Patch(color='#FFEB3B', label='Player 2 (O)')
    ax.legend(handles=[red_patch, yellow_patch], loc='upper right', 
             bbox_to_anchor=(1.15, 1), fontsize=11)
    
    plt.tight_layout()
    return 


def play_game_human_vs_ai():
    """
    Play a game: Human vs AI (Alpha-Beta)
    """
    print("\n" + "=" * 70)
    print("CONNECT 4 - HUMAN VS AI")
    print("=" * 70)
    print("\nYou are Player 1 (X), AI is Player 2 (O)")
    print("Enter column number (0-6) to drop your piece\n")
    
    game = Connect4Game()
    ai = AlphaBetaAgent(game, max_depth=5)
    board = game.get_initial_state()
    
    while True:
        game.print_board(board)
        
        terminal, winner = game.is_terminal(board)
        if terminal:
            if winner == 1:
                print("\nCongratulations! You win!")
            elif winner == 2:
                print("\nAI wins!")
            else:
                print("\nIt's a draw!")
            break
        
        if game.current_player == 1:
            # Human turn
            valid_moves = game.get_valid_moves(board)
            while True:
                try:
                    col = int(input(f"\nYour move (valid columns: {valid_moves}): "))
                    if col in valid_moves:
                        break
                    else:
                        print("Invalid column! Try again.")
                except ValueError:
                    print("Please enter a number!")
            
            board = game.apply_move(board, col, 1)
            game.current_player = 2
        else:
            # AI turn
            print("\nAI is thinking...")
            move, metrics = ai.get_best_move(board, 2)
            print(f"AI chooses column {move}")
            print(f"(Explored {metrics.nodes_expanded} nodes in {metrics.time_taken:.3f}s)")
            
            board = game.apply_move(board, move, 2)
            game.current_player = 1
def play_game_ai_vs_ai_with_snapshots():
    """
    Watch AI vs AI game (Minimax vs Alpha-Beta) and save board snapshots after each move.
    """
    print("\n" + "=" * 70)
    print("CONNECT 4 - AI VS AI (Minimax vs Alpha-Beta)")
    print("=" * 70)
    
    game = Connect4Game()
    minimax_ai = MinimaxAgent(game, max_depth=4)
    alphabeta_ai = AlphaBetaAgent(game, max_depth=4)
    board = game.get_initial_state()
    
    move_count = 0
    
    while True:
        game.print_board(board)
        print(f"\nMove {move_count + 1}")
        
        # Save snapshot
        visualize_board_state(board, title=f"Move {move_count + 1}")
        plt.savefig(f'board_move_{move_count + 1}.png', dpi=300, bbox_inches='tight')
        
        terminal, winner = game.is_terminal(board)
        if terminal:
            if winner == 1:
                print("\nMinimax (Player 1) wins!")
            elif winner == 2:
                print("\nAlpha-Beta (Player 2) wins!")
            else:
                print("\nIt's a draw!")
            break
        
        if game.current_player == 1:
            print("Minimax's turn...")
            move, metrics = minimax_ai.get_best_move(board, 1)
            print(f"Minimax chooses column {move}")
            print(f"Nodes: {metrics.nodes_expanded}, Time: {metrics.time_taken:.3f}s")
            board = game.apply_move(board, move, 1)
            game.current_player = 2
        else:
            print("Alpha-Beta's turn...")
            move, metrics = alphabeta_ai.get_best_move(board, 2)
            print(f"Alpha-Beta chooses column {move}")
            print(f"Nodes: {metrics.nodes_expanded}, Time: {metrics.time_taken:.3f}s, "
                  f"Prunes: {metrics.pruning_count}")
            board = game.apply_move(board, move, 2)
            game.current_player = 1
        
        move_count += 1

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CONNECT 4 - ADVERSARIAL SEARCH ALGORITHMS")
    print("Phase II: Minimax vs Alpha-Beta Pruning")
    print("=" * 70)
    
    while True:
        print("\nSelect an option:")
        print("1. Run Performance Comparison (for report data)")
        print("2. Play Human vs AI")
        print("3. Watch AI vs AI")
        print("4. Visualize Sample Board State")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            compare_algorithms()
        elif choice == '2':
            play_game_human_vs_ai()
        elif choice == '3':
            play_game_ai_vs_ai_with_snapshots()
        elif choice == '4':
            print("\nGenerating sample board visualization...")
            game = Connect4Game()
            # Create a sample board with a few moves
            board = game.get_initial_state()
            moves = [(3, 1), (3, 2), (2, 1), (4, 2), (2, 1), (5, 2)]
            for col, player in moves:
                board = game.apply_move(board, col, player)
            
            visualize_board_state(board, "Sample Connect 4 Game State")
            plt.savefig('sample_board_state.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: sample_board_state.png")
            plt.show()
        elif choice == '5':
            print("\nThank you for using the Connect 4 AI system!")
            break
        else:
            print("\nInvalid choice! Please enter a number from 1 to 5.")
