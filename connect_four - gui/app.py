import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import io
import base64
from flask import Flask, render_template, jsonify, request
import numpy as np
from connect_four import Connect4Game, MinimaxAgent, AlphaBetaAgent

app = Flask(__name__)

game = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_game():
    global game
    game = Connect4Game()
    return jsonify({'board': game.board.tolist()})

@app.route('/move', methods=['POST'])
def human_move():
    global game
    if not game:
        return jsonify({'error': 'Game not started'}), 400
    
    col = int(request.json.get('col'))
    valid_moves = game.get_valid_moves(game.board)
    
    if col not in valid_moves:
        return jsonify({'error': 'Invalid move'}), 400
    
    game.board = game.apply_move(game.board, col, 1)
    terminal, winner = game.is_terminal(game.board)
    
    return jsonify({
        'board': game.board.tolist(), 
        'game_over': terminal, 
        'winner': winner
    })

@app.route('/ai_move', methods=['POST'])
def ai_move():
    global game
    if not game:
        return jsonify({'error': 'Game not started'}), 400
    
    data = request.json
    player_num = data.get('player', 2)
    depth = int(data.get('depth', 4))
    
    if player_num == 1:
        agent = MinimaxAgent(game, max_depth=depth)
    else:
        agent = AlphaBetaAgent(game, max_depth=depth)
        
    col, metrics = agent.get_best_move(game.board, player_num)
    game.board = game.apply_move(game.board, col, player_num)
    terminal, winner = game.is_terminal(game.board)
    
    return jsonify({
        'board': game.board.tolist(), 
        'game_over': terminal, 
        'winner': winner, 
        'metrics': {
            'time': metrics.time_taken, 
            'nodes': metrics.nodes_expanded, 
            'pruning': metrics.pruning_count
        }
    })

@app.route('/compare', methods=['POST'])
def compare_performance():
    global game
    if not game:
        game = Connect4Game()
    
    target_depth = int(request.json.get('depth', 4))
    
    test_depths = [target_depth - 1, target_depth] if target_depth > 2 else [2]
    mm_times, ab_times, mm_nodes, ab_nodes, ab_prunes = [], [], [], [], []
    
    term, _ = game.is_terminal(game.board)
    base_board = Connect4Game().board if term else game.board.copy()
    
    for d in test_depths:
        mm = MinimaxAgent(game, max_depth=d)
        _, m_met = mm.get_best_move(base_board, 1)
        mm_times.append(m_met.time_taken)
        mm_nodes.append(m_met.nodes_expanded)
        
        ab = AlphaBetaAgent(game, max_depth=d)
        _, a_met = ab.get_best_move(base_board, 1)
        ab_times.append(a_met.time_taken)
        ab_nodes.append(a_met.nodes_expanded)
        ab_prunes.append(a_met.pruning_count)

    fig1 = plt.figure(figsize=(15, 10))
    
    ax1 = plt.subplot(2, 3, 1)
    ax1.bar(['Minimax', 'AlphaBeta'], [mm_times[-1], ab_times[-1]], color=['#FF6B6B', '#4ECDC4'], edgecolor='black')
    ax1.set_title('Time (s) @ Depth ' + str(target_depth))
    ax1.grid(axis='y', alpha=0.3)
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.bar(['Minimax', 'AlphaBeta'], [mm_nodes[-1], ab_nodes[-1]], color=['#FF6B6B', '#4ECDC4'], edgecolor='black')
    ax2.set_title('Nodes Expanded')
    ax2.grid(axis='y', alpha=0.3)
    
    ax3 = plt.subplot(2, 3, 3)
    saved = max(0, mm_nodes[-1] - ab_nodes[-1])
    ax3.pie([saved, ab_nodes[-1]], labels=['Pruned/Saved', 'Explored'], colors=['#2ecc71', '#95a5a6'], autopct='%1.1f%%', startangle=90)
    ax3.set_title('Search Space Reduction')
    
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(test_depths, mm_times, 'o-', color='#FF6B6B', label='Minimax')
    ax4.plot(test_depths, ab_times, 's-', color='#4ECDC4', label='AlphaBeta')
    ax4.set_title('Time vs Depth')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(test_depths, mm_nodes, 'o-', color='#FF6B6B', label='Minimax')
    ax5.plot(test_depths, ab_nodes, 's-', color='#4ECDC4', label='AlphaBeta')
    ax5.set_title('Nodes vs Depth')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    ax6 = plt.subplot(2, 3, 6)
    ax6.bar([str(d) for d in test_depths], ab_prunes, color='#AA96DA', edgecolor='black')
    ax6.set_title('Pruning Count')
    ax6.grid(axis='y', alpha=0.3)
    
    buf1 = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf1, format='png', dpi=100)
    plt.close(fig1)
    buf1.seek(0)
    charts_b64 = base64.b64encode(buf1.getvalue()).decode('utf8')

    fig_tree, (ax_t1, ax_t2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax_t1.set_xlim(0, 10)
    ax_t1.set_ylim(0, 10)
    ax_t1.axis('off')
    ax_t1.set_title('Minimax Algorithm\n(All Nodes Explored)', fontsize=14, fontweight='bold')
    
    ax_t1.add_patch(Circle((5, 9), 0.3, color='#4ECDC4', ec='black'))
    ax_t1.text(5, 9, 'MAX', ha='center', va='center', fontsize=9)
    
    level1_x = [2, 5, 8]
    for x in level1_x:
        ax_t1.add_patch(Circle((x, 7), 0.3, color='#FF6B6B', ec='black'))
        ax_t1.text(x, 7, 'MIN', ha='center', va='center', fontsize=8)
        ax_t1.plot([5, x], [8.7, 7.3], 'k-', linewidth=1.5)
    
    level2_pos = [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5), (7, 5), (8, 5), (9, 5)]
    values = [3, 5, 2, 7, 4, 6, 8, 3, 5]
    for i, (x, y) in enumerate(level2_pos):
        ax_t1.add_patch(Circle((x, y), 0.25, color='#95E1D3', ec='black'))
        ax_t1.text(x, y-0.6, str(values[i]), ha='center', color='blue', fontweight='bold')
    
    for i, px in enumerate(level1_x):
        for j in range(3):
            cx = level2_pos[i*3+j][0]
            ax_t1.plot([px, cx], [6.7, 5.3], 'k-', linewidth=1, alpha=0.7)
            
    ax_t1.text(5, 1, 'Nodes Explored: 13', ha='center', fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat'))

    ax_t2.set_xlim(0, 10)
    ax_t2.set_ylim(0, 10)
    ax_t2.axis('off')
    ax_t2.set_title('Alpha-Beta Pruning\n(Pruned Branches Marked)', fontsize=14, fontweight='bold')
    
    ax_t2.add_patch(Circle((5, 9), 0.3, color='#4ECDC4', ec='black'))
    ax_t2.text(5, 9, 'MAX', ha='center', va='center', fontsize=9)
    
    for x in level1_x:
        ax_t2.add_patch(Circle((x, 7), 0.3, color='#FF6B6B', ec='black'))
        ax_t2.text(x, 7, 'MIN', ha='center', va='center', fontsize=8)
        ax_t2.plot([5, x], [8.7, 7.3], 'k-', linewidth=1.5)
    
    explored = [True, True, True, True, False, False, True, False, False]
    for i, (x, y) in enumerate(level2_pos):
        if explored[i]:
            ax_t2.add_patch(Circle((x, y), 0.25, color='#95E1D3', ec='black'))
            ax_t2.text(x, y-0.6, str(values[i]), ha='center', color='blue', fontweight='bold')
        else:
            ax_t2.add_patch(Circle((x, y), 0.25, color='lightgray', ec='red', linestyle='--'))
            ax_t2.text(x, y, 'X', ha='center', va='center', color='red', fontsize=12, fontweight='bold')
            
    for i, px in enumerate(level1_x):
        for j in range(3):
            idx = i*3+j
            cx = level2_pos[idx][0]
            if explored[idx]:
                ax_t2.plot([px, cx], [6.7, 5.3], 'k-', linewidth=1, alpha=0.7)
            else:
                ax_t2.plot([px, cx], [6.7, 5.3], 'r--', linewidth=1, alpha=0.3)

    ax_t2.text(5, 1, 'Nodes Explored: 8', ha='center', fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgreen'))
    ax_t2.text(5, 0.3, 'Pruned: 5 Nodes (38% Less)', ha='center', fontsize=10, color='red', fontweight='bold')

    buf2 = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf2, format='png', dpi=100)
    plt.close(fig_tree)
    buf2.seek(0)
    tree_b64 = base64.b64encode(buf2.getvalue()).decode('utf8')

    return jsonify({
        'charts_image': charts_b64,
        'tree_image': tree_b64
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)