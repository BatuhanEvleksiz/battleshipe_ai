# Self-Play Training Script for Battleship DQN AI
# Bu script AI'nın kendine karşı oynayarak öğrenmesini sağlar

import numpy as np
import torch
import random
from typing import List, Tuple, Optional
import os
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Ana oyun dosyasından import
from battleship_ai_dqn import (
    Board, DQNAgent, BOARD_SIZE, SHIP_SPECS, 
    EMPTY, SHIP, HIT, MISS, UNKNOWN, Coord
)

class SelfPlayTrainer:
    def __init__(self, num_episodes=1000, save_interval=100):
        """
        Self-play training için trainer sınıfı
        
        Args:
            num_episodes: Toplam oyun sayısı
            save_interval: Model kaydetme aralığı
        """
        self.num_episodes = num_episodes
        self.save_interval = save_interval
        
        # İki AI agent oluştur
        self.agent1 = DQNAgent()
        self.agent2 = DQNAgent()
        
        # Model yükle (varsa)
        self.model_path = "battleship_dqn_model.pth"
        if os.path.exists(self.model_path):
            self.agent1.load_model(self.model_path)
            self.agent2.load_model(self.model_path)
            print(f"Mevcut model yüklendi: {self.model_path}")
        
        # İstatistikler
        self.stats = {
            'agent1_wins': 0,
            'agent2_wins': 0,
            'avg_game_length': [],
            'hit_rates': [],
            'rewards': [],
            'epsilon_values': []
        }
        
    def play_game(self, verbose=False) -> Tuple[str, int, float]:
        """
        İki AI arasında tek bir oyun oynat
        
        Returns:
            winner: 'agent1' veya 'agent2'
            num_moves: Toplam hamle sayısı
            hit_rate: Ortalama isabet oranı
        """
        # Tahtaları oluştur
        board1 = Board()
        board2 = Board()
        board1.random_place_all(SHIP_SPECS)
        board2.random_place_all(SHIP_SPECS)
        
        # AI'ları sıfırla
        self.agent1.reset()
        self.agent2.reset()
        
        # Oyun değişkenleri
        turn = 'agent1'
        num_moves = 0
        total_hits = 0
        
        # Experience buffer'ları
        experiences_1 = []
        experiences_2 = []
        
        while True:
            num_moves += 1
            
            if turn == 'agent1':
                # Agent1 Agent2'ye atış yapıyor
                current_agent = self.agent1
                target_board = board2
                experiences = experiences_1
            else:
                # Agent2 Agent1'e atış yapıyor
                current_agent = self.agent2
                target_board = board1
                experiences = experiences_2
            
            # State'i al
            board_state = target_board.get_state_for_ai()
            state = current_agent.get_state_features(board_state)
            
            # Valid actions
            valid_actions = current_agent.get_valid_actions()
            
            if not valid_actions:
                # Oyun patlama durumu (nadir)
                break
            
            # Action seç (training modunda)
            action = current_agent.choose_action(state, valid_actions, training=True)
            shot = current_agent.action_to_coord(action)
            current_agent.shots_taken.add(shot)
            
            # Atış yap
            hit, sunk, sunk_len = target_board.receive_shot(shot)
            current_agent.feedback(shot, hit, sunk, sunk_len)
            
            if hit:
                total_hits += 1
            
            # Reward hesapla
            if hit:
                if sunk:
                    # Gemi batırma bonusu
                    reward = 50 + (sunk_len * 20)
                else:
                    # Sadece vurma
                    reward = 10
            else:
                # Iska cezası
                reward = -2
            
            # Stratejik bonus/ceza
            if len(current_agent.consecutive_hits) > 1:
                # Ardışık vuruşları teşvik et
                reward += 5
            
            # Yeni state
            new_board_state = target_board.get_state_for_ai()
            new_state = current_agent.get_state_features(new_board_state)
            
            # Oyun bitti mi?
            done = target_board.all_sunk()
            
            # Experience kaydet
            experiences.append((state, action, reward, new_state, done))
            
            if verbose and num_moves % 10 == 0:
                print(f"Move {num_moves}: {turn} - Hit: {hit}, Sunk: {sunk}")
            
            if done:
                winner = turn
                # Kazanan bonus
                for i in range(len(experiences)):
                    s, a, r, ns, d = experiences[i]
                    experiences[i] = (s, a, r + 100, ns, d)  # Kazanma bonusu
                
                # Kaybeden ceza
                loser_experiences = experiences_2 if turn == 'agent1' else experiences_1
                for i in range(len(loser_experiences)):
                    s, a, r, ns, d = loser_experiences[i]
                    loser_experiences[i] = (s, a, r - 50, ns, True)  # Kaybetme cezası
                
                break
            
            # Sıra değiştir
            turn = 'agent2' if turn == 'agent1' else 'agent1'
            
            # Maksimum hamle kontrolü
            if num_moves > 200:
                winner = 'draw'
                break
        
        # Experiences'ları replay memory'ye ekle
        for exp in experiences_1:
            self.agent1.memory.push(*exp)
        for exp in experiences_2:
            self.agent2.memory.push(*exp)
        
        # Hit rate hesapla
        hit_rate = total_hits / num_moves if num_moves > 0 else 0
        
        return winner, num_moves, hit_rate
    
    def train_agents(self, batch_updates=10):
        """Her iki ajanı da eğit"""
        for _ in range(batch_updates):
            self.agent1.update_model()
            self.agent2.update_model()
    
    def sync_models(self):
        """
        İki ajanın modellerini senkronize et
        Daha iyi performans gösteren modeli diğerine kopyala
        """
        # Basit senkronizasyon: agent1'i agent2'ye kopyala
        self.agent2.policy_net.load_state_dict(self.agent1.policy_net.state_dict())
        self.agent2.target_net.load_state_dict(self.agent1.target_net.state_dict())
    
    def run_training(self):
        """Ana training döngüsü"""
        print(f"Starting self-play training for {self.num_episodes} episodes...")
        print("=" * 60)
        
        best_hit_rate = 0
        
        for episode in range(1, self.num_episodes + 1):
            # Oyun oyna
            winner, num_moves, hit_rate = self.play_game(verbose=False)
            
            # İstatistikleri güncelle
            if winner == 'agent1':
                self.stats['agent1_wins'] += 1
            elif winner == 'agent2':
                self.stats['agent2_wins'] += 1
            
            self.stats['avg_game_length'].append(num_moves)
            self.stats['hit_rates'].append(hit_rate)
            self.stats['epsilon_values'].append(self.agent1.epsilon)
            
            # Ajanları eğit
            self.train_agents(batch_updates=5)
            
            # Periyodik olarak modelleri senkronize et
            if episode % 50 == 0:
                self.sync_models()
            
            # İlerleme raporu
            if episode % 10 == 0:
                recent_games = self.stats['avg_game_length'][-10:]
                recent_hits = self.stats['hit_rates'][-10:]
                avg_length = np.mean(recent_games)
                avg_hit_rate = np.mean(recent_hits)
                
                print(f"Episode {episode}/{self.num_episodes}")
                print(f"  Agent1 Wins: {self.stats['agent1_wins']}, Agent2 Wins: {self.stats['agent2_wins']}")
                print(f"  Avg Game Length (last 10): {avg_length:.1f}")
                print(f"  Avg Hit Rate (last 10): {avg_hit_rate:.3f}")
                print(f"  Epsilon: {self.agent1.epsilon:.4f}")
                print("-" * 40)
                
                # En iyi model kontrolü
                if avg_hit_rate > best_hit_rate:
                    best_hit_rate = avg_hit_rate
                    print(f"  NEW BEST HIT RATE: {best_hit_rate:.3f}")
            
            # Model kaydet
            if episode % self.save_interval == 0:
                self.agent1.save_model(self.model_path)
                self.save_stats(f"training_stats_{episode}.json")
                print(f"  Model and stats saved at episode {episode}")
                self.plot_progress(episode)
    
    def save_stats(self, filename):
        """İstatistikleri JSON olarak kaydet"""
        stats_to_save = {
            'agent1_wins': self.stats['agent1_wins'],
            'agent2_wins': self.stats['agent2_wins'],
            'avg_game_length': self.stats['avg_game_length'][-100:],  # Son 100 oyun
            'hit_rates': self.stats['hit_rates'][-100:],
            'epsilon_values': self.stats['epsilon_values'][-100:]
        }
        
        with open(filename, 'w') as f:
            json.dump(stats_to_save, f, indent=2)
    
    def plot_progress(self, episode):
        """Training ilerlemesini görselleştir"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Training Progress - Episode {episode}', fontsize=16)
        
        # 1. Win Rate
        total_games = self.stats['agent1_wins'] + self.stats['agent2_wins']
        if total_games > 0:
            win_rate1 = self.stats['agent1_wins'] / total_games
            win_rate2 = self.stats['agent2_wins'] / total_games
            axes[0, 0].bar(['Agent 1', 'Agent 2'], [win_rate1, win_rate2])
            axes[0, 0].set_ylabel('Win Rate')
            axes[0, 0].set_title('Win Distribution')
            axes[0, 0].set_ylim([0, 1])
        
        # 2. Game Length Over Time
        if len(self.stats['avg_game_length']) > 0:
            window = min(50, len(self.stats['avg_game_length']))
            smoothed_length = np.convolve(self.stats['avg_game_length'], 
                                         np.ones(window)/window, mode='valid')
            axes[0, 1].plot(smoothed_length)
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Average Game Length')
            axes[0, 1].set_title(f'Game Length (smoothed, window={window})')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Hit Rate Over Time
        if len(self.stats['hit_rates']) > 0:
            window = min(50, len(self.stats['hit_rates']))
            smoothed_hits = np.convolve(self.stats['hit_rates'], 
                                       np.ones(window)/window, mode='valid')
            axes[1, 0].plot(smoothed_hits)
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Hit Rate')
            axes[1, 0].set_title(f'Hit Rate (smoothed, window={window})')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Epsilon Decay
        if len(self.stats['epsilon_values']) > 0:
            axes[1, 1].plot(self.stats['epsilon_values'])
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Epsilon')
            axes[1, 1].set_title('Exploration Rate (Epsilon)')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim([0, 1])
        
        plt.tight_layout()
        
        # Grafikleri kaydet
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'training_progress_{timestamp}.png', dpi=100)
        plt.close()
        print(f"  Progress plot saved: training_progress_{timestamp}.png")

def test_trained_model(num_games=100):
    """
    Eğitilmiş modeli test et
    Random agent'a karşı performansını ölç
    """
    print("\nTesting trained model against random agent...")
    print("=" * 60)
    
    # Eğitilmiş AI
    trained_ai = DQNAgent("battleship_dqn_model.pth")
    trained_ai.epsilon = 0.01  # Çok az exploration
    
    wins = 0
    total_moves = []
    hit_rates = []
    
    for game_num in range(1, num_games + 1):
        # Tahtalar
        ai_board = Board()
        random_board = Board()
        ai_board.random_place_all(SHIP_SPECS)
        random_board.random_place_all(SHIP_SPECS)
        
        trained_ai.reset()
        
        # Rastgele başlayan
        random_shots = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)]
        random.shuffle(random_shots)
        random_shot_idx = 0
        
        moves = 0
        ai_hits = 0
        
        # Kim önce bitirir?
        ai_finished = False
        random_finished = False
        
        while not ai_finished and not random_finished:
            moves += 1
            
            # AI atışı
            if not ai_finished:
                board_state = random_board.get_state_for_ai()
                state = trained_ai.get_state_features(board_state)
                valid_actions = trained_ai.get_valid_actions()
                
                if valid_actions:
                    action = trained_ai.choose_action(state, valid_actions, training=False)
                    shot = trained_ai.action_to_coord(action)
                    trained_ai.shots_taken.add(shot)
                    
                    hit, sunk, _ = random_board.receive_shot(shot)
                    trained_ai.feedback(shot, hit, sunk)
                    
                    if hit:
                        ai_hits += 1
                    
                    if random_board.all_sunk():
                        ai_finished = True
                        wins += 1
            
            # Random atışı
            if not random_finished and random_shot_idx < len(random_shots):
                shot = random_shots[random_shot_idx]
                random_shot_idx += 1
                hit, _, _ = ai_board.receive_shot(shot)
                
                if ai_board.all_sunk():
                    random_finished = True
        
        total_moves.append(moves)
        hit_rate = ai_hits / len(trained_ai.shots_taken) if len(trained_ai.shots_taken) > 0 else 0
        hit_rates.append(hit_rate)
        
        if game_num % 10 == 0:
            win_rate = wins / game_num
            avg_moves = np.mean(total_moves)
            avg_hit_rate = np.mean(hit_rates)
            print(f"Games: {game_num}/{num_games} | Win Rate: {win_rate:.2%} | "
                  f"Avg Moves: {avg_moves:.1f} | Avg Hit Rate: {avg_hit_rate:.3f}")
    
    # Final sonuçlar
    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print(f"  Total Games: {num_games}")
    print(f"  Wins: {wins}/{num_games} ({wins/num_games*100:.1f}%)")
    print(f"  Average Game Length: {np.mean(total_moves):.1f} moves")
    print(f"  Average Hit Rate: {np.mean(hit_rates):.3f}")
    print(f"  Std Dev Hit Rate: {np.std(hit_rates):.3f}")
    print("=" * 60)

def main():
    """Ana fonksiyon"""
    import sys
    
    print("=" * 60)
    print("BATTLESHIP DQN SELF-PLAY TRAINER")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            # Sadece test et
            test_trained_model(num_games=100)
        elif sys.argv[1] == 'train':
            # Eğit
            episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
            trainer = SelfPlayTrainer(num_episodes=episodes, save_interval=100)
            trainer.run_training()
            
            # Eğitim sonrası test
            print("\n" + "=" * 60)
            print("Training complete! Now testing the model...")
            test_trained_model(num_games=50)
        else:
            print("Usage: python train_self_play.py [train|test] [episodes]")
    else:
        # Varsayılan: kısa eğitim + test
        print("Running default training (500 episodes) + testing...")
        trainer = SelfPlayTrainer(num_episodes=500, save_interval=50)
        trainer.run_training()
        test_trained_model(num_games=50)

if __name__ == "__main__":
    main()
