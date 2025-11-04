# Amiral Battı (Battleship) - Pygame + Deep Q-Learning AI
# pip install pygame torch numpy

import pygame
import random
import string
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import pickle
from typing import List, Tuple, Optional, Set
import os

# ---------------- Config ----------------
BOARD_SIZE = 10            # Tahta boyutu (10x10)
CELL = 40                  # Hücre piksel boyutu
PADDING = 40               # Kenar boşluğu
GAP = 80                   # Panolar arası boşluk
WIDTH = PADDING + BOARD_SIZE*CELL + GAP + BOARD_SIZE*CELL + PADDING
HEIGHT = PADDING + BOARD_SIZE*CELL + 140
SHIP_SPECS = [5, 4, 3, 3, 2]  # Gemilerin boyları
FONT_NAME = "arial"
FPS = 60

# Hücre durumları
EMPTY = 0
SHIP = 1
HIT = 2
MISS = 3
UNKNOWN = 4  # AI için görünmeyen kare

Coord = Tuple[int, int]

# -------------- DQN Hiperparametreleri --------------
LEARNING_RATE = 0.001
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
MEMORY_SIZE = 10000
TARGET_UPDATE = 100
SAVE_INTERVAL = 100

# -------------- Yardımcılar --------------
def in_bounds(r: int, c: int) -> bool:
    """Koordinat tahtanın içinde mi?"""
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

def rc_to_label(rc: Coord) -> str:
    """(r,c) -> 'A0' etiketi."""
    r, c = rc
    return f"{string.ascii_uppercase[r]}{c}"

# -------------- Deep Q-Network Model --------------
class DQN(nn.Module):
    """Tahta -> Q-değerleri üreten çok katmanlı ağ."""
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        # Tam bağlantılı katmanlar
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.fc4 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc5 = nn.Linear(hidden_size, output_size)
        # Normalizasyon + dropout (daha stabil eğitim için)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size * 2)
        self.bn3 = nn.BatchNorm1d(hidden_size * 2)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        """İleri yayılım (relu + bn + dropout)."""
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.fc1(x)) if batch_size > 1 else self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)) if batch_size > 1 else self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)) if batch_size > 1 else self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc4(x)) if batch_size > 1 else self.fc4(x))
        x = self.dropout(x)
        return self.fc5(x)

# -------------- Experience Replay Memory --------------
class ReplayMemory:
    """(state, action, reward, next_state, done) deneyim kuyruğu."""
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Yeni deneyim ekle."""
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Rastgele minibatch döndür."""
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.memory)

# -------------- Tahta Sınıfı --------------
class Board:
    """Gemi yerleştirme, atış ve durum takibi."""
    def __init__(self):
        self.grid = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.ships: List[List[Coord]] = []
        self.remaining_cells = 0
        
    def can_place(self, r: int, c: int, length: int, horizontal: bool) -> bool:
        """Gemi bu başlangıçtan sığar mı ve çakışma yok mu?"""
        for i in range(length):
            rr = r + (0 if horizontal else i)
            cc = c + (i if horizontal else 0)
            if not in_bounds(rr, cc) or self.grid[rr][cc] != EMPTY:
                return False
        # Komşulukta gemi olmamalı (köşegen dahil)
        for i in range(length):
            rr = r + (0 if horizontal else i)
            cc = c + (i if horizontal else 0)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = rr + dr, cc + dc
                    if in_bounds(nr, nc) and self.grid[nr][nc] == SHIP:
                        return False
        return True
    
    def place_ship(self, r: int, c: int, length: int, horizontal: bool) -> bool:
        """Gemiyi yerleştir ve sayaçları güncelle."""
        if not self.can_place(r, c, length, horizontal):
            return False
        coords = []
        for i in range(length):
            rr = r + (0 if horizontal else i)
            cc = c + (i if horizontal else 0)
            self.grid[rr][cc] = SHIP
            coords.append((rr, cc))
        self.ships.append(coords)
        self.remaining_cells += length
        return True
    
    def random_place_all(self, specs: List[int]) -> None:
        """Tüm gemileri rastgele yerleştir (deneme-yanılma)."""
        for length in specs:
            placed = False
            for _ in range(500):
                horizontal = random.choice([True, False])
                r = random.randrange(BOARD_SIZE)
                c = random.randrange(BOARD_SIZE)
                if self.place_ship(r, c, length, horizontal):
                    placed = True
                    break
            if not placed:
                raise RuntimeError("Random yerleştirme başarısız.")
    
    def receive_shot(self, rc: Coord) -> Tuple[bool, bool, Optional[int]]:
        """Atış sonucunu işle: (is_hit, is_sunk, sunk_len)."""
        r, c = rc
        if self.grid[r][c] in (HIT, MISS):
            return False, False, None  # Aynı yere tekrar atış
        
        if self.grid[r][c] == SHIP:
            self.grid[r][c] = HIT
            self.remaining_cells -= 1
            # İlgili gemi tamamen vuruldu mu?
            sunk = False
            sunk_len = None
            for ship in self.ships:
                if rc in ship:
                    if all(self.grid[rr][cc] == HIT for rr, cc in ship):
                        sunk = True
                        sunk_len = len(ship)
                    break
            return True, sunk, sunk_len
        else:
            self.grid[r][c] = MISS
            return False, False, None
    
    def all_sunk(self) -> bool:
        """Tüm gemiler battı mı?"""
        return self.remaining_cells == 0
    
    def get_state_for_ai(self) -> np.ndarray:
        """AI için görünür tahta (gemi yerleri UNKNOWN)."""
        state = np.zeros((BOARD_SIZE, BOARD_SIZE))
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.grid[r][c] == HIT:
                    state[r][c] = HIT
                elif self.grid[r][c] == MISS:
                    state[r][c] = MISS
                else:
                    state[r][c] = UNKNOWN
        return state

# -------------- DQN AI Agent --------------
class DQNAgent:
    """Eylem seçme, deneyim biriktirme ve modeli güncelleme."""
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Girdi boyutu: 3 kanal (hit/miss/unknown) + ekstra 20 özellik
        self.state_size = BOARD_SIZE * BOARD_SIZE * 3 + 20
        self.action_size = BOARD_SIZE * BOARD_SIZE
        self.hidden_size = 256
        
        # Politika ve hedef ağlar
        self.policy_net = DQN(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizasyon ve replay hafızası
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory(MEMORY_SIZE)
        
        # Keşif/sömürü değişkenleri
        self.epsilon = EPSILON_START
        self.steps_done = 0
        self.episodes_done = 0
        
        # Oyun içi takip
        self.shots_taken: Set[Coord] = set()
        self.last_hits: List[Coord] = []
        self.consecutive_hits: List[Coord] = []
        
        # Varsa modeli yükle
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            print(f"Model yüklendi: {model_path}")
    
    def get_state_features(self, board_state: np.ndarray) -> torch.Tensor:
        """Tahtayı vektöre dönüştür (kanallar + özet istatistikler)."""
        features = []
        # Kanallar
        hit_channel = (board_state == HIT).astype(np.float32)
        miss_channel = (board_state == MISS).astype(np.float32)
        unknown_channel = (board_state == UNKNOWN).astype(np.float32)
        features.extend(hit_channel.flatten())
        features.extend(miss_channel.flatten())
        features.extend(unknown_channel.flatten())
        
        # Ekstra özellikler (ısı haritası, oranlar, hizalanma, vb.)
        extra_features = []
        if self.last_hits:
            heat_map = np.zeros((BOARD_SIZE, BOARD_SIZE))
            for r, c in self.last_hits[-5:]:
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        nr, nc = r + dr, c + dc
                        if in_bounds(nr, nc):
                            distance = max(abs(dr), abs(dc))
                            heat_map[nr][nc] += 1.0 / (distance + 1)
            extra_features.append(np.mean(heat_map))
            extra_features.append(np.max(heat_map))
        else:
            extra_features.extend([0, 0])
        
        total_hits = np.sum(hit_channel)
        total_misses = np.sum(miss_channel)
        total_unknown = np.sum(unknown_channel)
        extra_features.append(total_hits / 100)
        extra_features.append(total_misses / 100)
        extra_features.append(total_unknown / 100)
        
        # Ardışık hizalanma (yatay/dikey)
        if len(self.consecutive_hits) >= 2:
            rows = [h[0] for h in self.consecutive_hits]
            cols = [h[1] for h in self.consecutive_hits]
            extra_features.append(1.0 if len(set(rows)) == 1 else 0.0)
            extra_features.append(1.0 if len(set(cols)) == 1 else 0.0)
            extra_features.append(len(self.consecutive_hits) / 5.0)
        else:
            extra_features.extend([0, 0, 0])
        
        # Olasılık haritası
        prob_map = self._calculate_ship_probability(board_state)
        extra_features.extend([np.mean(prob_map), np.max(prob_map), np.std(prob_map)])
        
        # Kenar/köşe bilinmeyen oranları
        edge_unknown = 0
        corner_unknown = 0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board_state[r][c] == UNKNOWN:
                    if r in (0, BOARD_SIZE-1) or c in (0, BOARD_SIZE-1):
                        edge_unknown += 1
                    if (r in (0, BOARD_SIZE-1)) and (c in (0, BOARD_SIZE-1)):
                        corner_unknown += 1
        extra_features.append(edge_unknown / 36.0)
        extra_features.append(corner_unknown / 4.0)
        
        # 20 elemana tamamla
        while len(extra_features) < 20:
            extra_features.append(0)
        features.extend(extra_features[:20])
        return torch.FloatTensor(features).unsqueeze(0).to(self.device)
    
    def _calculate_ship_probability(self, board_state: np.ndarray) -> np.ndarray:
        """Basit olasılık haritası (uygun yerlere sayım)."""
        prob_map = np.zeros((BOARD_SIZE, BOARD_SIZE))
        remaining_ships = [2, 3, 3, 4, 5]  # Basitleştirilmiş varsayım
        # Yatay denemeler
        for ship_len in remaining_ships:
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE - ship_len + 1):
                    can_place = True
                    for i in range(ship_len):
                        if board_state[r][c+i] in [HIT, MISS]:
                            can_place = False
                            break
                    if can_place:
                        for i in range(ship_len):
                            if board_state[r][c+i] == UNKNOWN:
                                prob_map[r][c+i] += 1
            # Dikey denemeler
            for r in range(BOARD_SIZE - ship_len + 1):
                for c in range(BOARD_SIZE):
                    can_place = True
                    for i in range(ship_len):
                        if board_state[r+i][c] in [HIT, MISS]:
                            can_place = False
                            break
                    if can_place:
                        for i in range(ship_len):
                            if board_state[r+i][c] == UNKNOWN:
                                prob_map[r+i][c] += 1
        # Normalize 0..1
        if np.max(prob_map) > 0:
            prob_map = prob_map / np.max(prob_map)
        return prob_map
    
    def choose_action(self, state: torch.Tensor, valid_actions: List[int], training=False) -> int:
        """Epsilon-greedy aksiyon seçimi (keşif/sömürü)."""
        if training and random.random() < self.epsilon:
            # Keşif: bazen son isabet çevresini tercih et
            if self.consecutive_hits and random.random() < 0.7:
                action = self._get_strategic_random_action(valid_actions)
            else:
                action = random.choice(valid_actions)
        else:
            # Sömürü: Q-değerlerine göre en iyi hamle
            with torch.no_grad():
                q_values = self.policy_net(state)
                masked_q = q_values.clone()
                for i in range(self.action_size):
                    if i not in valid_actions:
                        masked_q[0][i] = -float('inf')
                action = masked_q.max(1)[1].item()
        return action
    
    def _get_strategic_random_action(self, valid_actions: List[int]) -> int:
        """Son vuruş etrafındaki komşuları öncele."""
        if not self.consecutive_hits:
            return random.choice(valid_actions)
        r, c = self.consecutive_hits[-1]
        neighbors = []
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc):
                idx = nr * BOARD_SIZE + nc
                if idx in valid_actions:
                    neighbors.append(idx)
        return random.choice(neighbors) if neighbors else random.choice(valid_actions)
    
    def action_to_coord(self, action: int) -> Coord:
        """Aksiyon indeksini (r,c)'ye çevir."""
        return (action // BOARD_SIZE, action % BOARD_SIZE)
    
    def coord_to_action(self, coord: Coord) -> int:
        """(r,c)'yi aksiyon indeksine çevir."""
        return coord[0] * BOARD_SIZE + coord[1]
    
    def get_valid_actions(self) -> List[int]:
        """Henüz atış yapılmamış tüm hücreler."""
        valid = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if (r, c) not in self.shots_taken:
                    valid.append(self.coord_to_action((r, c)))
        return valid
    
    def update_model(self):
        """Replay'den örnekle ve DQN güncellemesi yap."""
        if len(self.memory) < BATCH_SIZE:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        state_batch = torch.cat(states)
        action_batch = torch.LongTensor(actions).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        next_state_batch = torch.cat(next_states)
        done_batch = torch.FloatTensor(dones).to(self.device)
        # Q(s,a)
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        # max_a' Q_target(s',a)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        target_q_values = reward_batch + (GAMMA * next_q_values * (1 - done_batch))
        # Kayıp ve geri yayılım
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step()
        # Epsilon azalt ve hedef ağı güncelle
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
        self.steps_done += 1
        if self.steps_done % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def feedback(self, shot: Coord, hit: bool, sunk: bool, sunk_len: Optional[int] = None):
        """Atış sonucuna göre iç durumu güncelle (ardışık isabetler vb.)."""
        if hit:
            self.last_hits.append(shot)
            self.consecutive_hits.append(shot)
            if sunk:
                self.consecutive_hits.clear()
        else:
            # Iska: son çizgiyi daralt (tam sıfırlama yerine sonuncuyu tut)
            if len(self.consecutive_hits) > 1:
                self.consecutive_hits = [self.consecutive_hits[-1]]
    
    def save_model(self, path: str):
        """Modeli diske kaydet."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done
        }, path)
        print(f"Model kaydedildi: {path}")
    
    def load_model(self, path: str):
        """Diskten modeli yükle."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', EPSILON_END)
        self.steps_done = checkpoint.get('steps_done', 0)
        self.episodes_done = checkpoint.get('episodes_done', 0)
    
    def reset(self):
        """Yeni oyun için iç durum sıfırlama."""
        self.shots_taken.clear()
        self.last_hits.clear()
        self.consecutive_hits.clear()

# -------------- Oyun Durumu --------------
class Game:
    """Pygame akışını ve kullanıcı etkileşimini yönetir."""
    def __init__(self, ai_model_path=None):
        self.player = Board()
        self.enemy = Board()
        self.enemy.random_place_all(SHIP_SPECS)
        # DQN AI
        self.ai = DQNAgent(ai_model_path)
        # Yerleştirme durumu
        self.placing_index = 0
        self.horizontal = True
        self.state = "placing"    # placing | playing | gameover
        self.turn = "player"      # player | ai
        self.msg = "Gemileri yerleştir: uzunluk 5 (R ile yön)"
        self.result = ""
        # Oyuncu atışlarının kaydı
        self.player_shots: Set[Coord] = set()
        # Eğitim moduna özel alanlar
        self.training_mode = False
        self.current_state = None
        self.current_action = None
    
    def reset(self, keep_ai=True):
        """Yeni oyuna başla (AI objesini koruyarak)."""
        player_ai = self.ai if keep_ai else None
        self.__init__(ai_model_path=None)
        if keep_ai and player_ai:
            self.ai = player_ai
            self.ai.reset()

# -------------- Pygame Çizim Fonksiyonları --------------
def draw_grid(surface, start_x, start_y):
    """Izgarayı çizer."""
    for i in range(BOARD_SIZE+1):
        pygame.draw.line(surface, (200,200,200),
                        (start_x + i*CELL, start_y),
                        (start_x + i*CELL, start_y + BOARD_SIZE*CELL), 1)
        pygame.draw.line(surface, (200,200,200),
                        (start_x, start_y + i*CELL),
                        (start_x + BOARD_SIZE*CELL, start_y + i*CELL), 1)

def draw_labels(surface, start_x, start_y, font):
    """Satır/kolon etiketlerini (A..J / 0..9) çizer."""
    for r in range(BOARD_SIZE):
        txt = font.render(string.ascii_uppercase[r], True, (230,230,230))
        surface.blit(txt, (start_x - 20, start_y + r*CELL + CELL/3))
    for c in range(BOARD_SIZE):
        txt = font.render(str(c), True, (230,230,230))
        surface.blit(txt, (start_x + c*CELL + CELL/3, start_y - 24))

def cell_at(pos, start_x, start_y) -> Optional[Coord]:
    """Mouse pozisyonunu hücre koordinatına çevir."""
    x, y = pos
    if x < start_x or y < start_y:
        return None
    dx = x - start_x
    dy = y - start_y
    c = dx // CELL
    r = dy // CELL
    if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
        return int(r), int(c)
    return None

def draw_board(surface, board: Board, start_x, start_y, show_ships: bool, font_small):
    """Hücreleri ve durumlarını (gemi/ıska/vuruş) çizer."""
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            rect = pygame.Rect(start_x + c*CELL + 1, start_y + r*CELL + 1,
                             CELL-2, CELL-2)
            val = board.grid[r][c]
            # Zemin
            pygame.draw.rect(surface, (25, 40, 60), rect)
            # Gemi göster (sadece kendi panosunda)
            if show_ships and val == SHIP:
                pygame.draw.rect(surface, (80, 160, 220), rect)
            # Vuruş/Iska izleri
            if val == HIT:
                pygame.draw.line(surface, (220,70,70), rect.topleft, rect.bottomright, 3)
                pygame.draw.line(surface, (220,70,70), rect.topright, rect.bottomleft, 3)
            elif val == MISS:
                pygame.draw.circle(surface, (180,180,220), rect.center, 5, 0)

def draw_ship_preview(surface, start_x, start_y, cell_rc: Optional[Coord],
                     length: int, horizontal: bool):
    """Yerleştirme sırasında yeşil önizleme çerçevesi çizer."""
    if cell_rc is None:
        return
    r, c = cell_rc
    for i in range(length):
        rr = r + (0 if horizontal else i)
        cc = c + (i if horizontal else 0)
        if not in_bounds(rr, cc):
            continue
        rect = pygame.Rect(start_x + cc*CELL + 1, start_y + rr*CELL + 1,
                         CELL-2, CELL-2)
        pygame.draw.rect(surface, (120, 200, 120), rect, 3)

# -------------- Ana Oyun Döngüsü --------------
def main():
    """Pygame penceresini açar ve oyunu çalıştırır."""
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Battleship - Deep Q-Learning AI")
    clock = pygame.time.Clock()
    
    font = pygame.font.SysFont(FONT_NAME, 24)
    font_small = pygame.font.SysFont(FONT_NAME, 18)
    font_big = pygame.font.SysFont(FONT_NAME, 36, bold=True)
    font_tiny = pygame.font.SysFont(FONT_NAME, 14)
    
    # Model yükle (varsa)
    model_path = "battleship_dqn_model.pth"
    game = Game(ai_model_path=model_path if os.path.exists(model_path) else None)
    
    # Panoların sol üst köşe koordinatları
    left_x = PADDING
    left_y = PADDING + 40
    right_x = PADDING + BOARD_SIZE*CELL + GAP
    right_y = left_y
    
    running = True
    while running:
        clock.tick(FPS)
        mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Klavye kontrolleri
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and game.state == "placing":
                    game.horizontal = not game.horizontal      # Yerleştirme yönünü değiştir
                if event.key == pygame.K_n and game.state == "gameover":
                    game.reset()                               # Yeni oyun
                if event.key == pygame.K_t:  # Training modunu aç/kapa
                    game.training_mode = not game.training_mode
                    print(f"Training modu: {'AÇIK' if game.training_mode else 'KAPALI'}")
                if event.key == pygame.K_s:  # Modeli kaydet
                    game.ai.save_model(model_path)
            
            # Fare tıklamaları
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Gemi yerleştirme
                if game.state == "placing":
                    cell = cell_at(mouse_pos, left_x, left_y)
                    if cell:
                        length = SHIP_SPECS[game.placing_index]
                        if game.player.place_ship(cell[0], cell[1], length, game.horizontal):
                            game.placing_index += 1
                            if game.placing_index >= len(SHIP_SPECS):
                                game.state = "playing"
                                game.msg = "Oyun başladı! Sağ panoya tıklayarak atış yap."
                            else:
                                nxt = SHIP_SPECS[game.placing_index]
                                game.msg = f"Gemileri yerleştir: uzunluk {nxt} (R ile yön)"
                # Oyuncu atışı
                elif game.state == "playing" and game.turn == "player":
                    cell = cell_at(mouse_pos, right_x, right_y)
                    if cell:
                        rc = (cell[0], cell[1])
                        if rc not in game.player_shots:
                            game.player_shots.add(rc)
                            hit, sunk, sunk_len = game.enemy.receive_shot(rc)
                            game.msg = "Vurdun! " + (f"Gemi battı ({sunk_len})." if sunk else "") if hit else "Iska."
                            if game.enemy.all_sunk():
                                game.state = "gameover"
                                game.result = "KAZANDIN! 🎉"
                            else:
                                game.turn = "ai"
        
        # AI hamlesi
        if game.state == "playing" and game.turn == "ai":
            # Tahta durumunu al ve özellik çıkar
            board_state = game.player.get_state_for_ai()
            state = game.ai.get_state_features(board_state)
            # Geçerli hamleler
            valid_actions = game.ai.get_valid_actions()
            # Aksiyon seç
            action = game.ai.choose_action(state, valid_actions, training=game.training_mode)
            shot = game.ai.action_to_coord(action)
            game.ai.shots_taken.add(shot)
            # Uygula
            hit, sunk, sunk_len = game.player.receive_shot(shot)
            game.ai.feedback(shot, hit, sunk, sunk_len)
            # Eğitim modundaysa ödüllendir ve replay'e ekle
            if game.training_mode:
                reward = 10 + (50 if sunk else 0) + (sunk_len * 10 if sunk else 0) if hit else -1
                new_state = game.ai.get_state_features(game.player.get_state_for_ai())
                done = game.player.all_sunk()
                game.ai.memory.push(state, action, reward, new_state, done)
                game.ai.update_model()
            # UI mesajı
            lab = rc_to_label(shot)
            game.msg = (f"AI {lab} vurdu!" + (f" Gemi battı ({sunk_len})." if sunk else "")) if hit else f"AI {lab} ıska."
            # Bitti mi?
            if game.player.all_sunk():
                game.state = "gameover"
                game.result = "KAYBETTİN 😞"
                if game.training_mode:
                    game.ai.episodes_done += 1
                    if game.ai.episodes_done % SAVE_INTERVAL == 0:
                        game.ai.save_model(model_path)
            else:
                game.turn = "player"
        
        # ---- Çizim ----
        screen.fill((15, 20, 30))
        # Başlık
        title = font_big.render("Battleship - DQN AI", True, (240,240,255))
        screen.blit(title, (PADDING, 10))
        # Panel başlıkları
        lh = font.render("Senin Panon", True, (220,230,255))
        rh = font.render("AI Panosu", True, (220,230,255))
        screen.blit(lh, (left_x, left_y - 34))
        screen.blit(rh, (right_x, right_y - 34))
        # Grid ve etiketler
        draw_grid(screen, left_x, left_y)
        draw_grid(screen, right_x, right_y)
        draw_labels(screen, left_x, left_y, font_small)
        draw_labels(screen, right_x, right_y, font_small)
        # Tahtalar
        draw_board(screen, game.player, left_x, left_y, show_ships=True,  font_small=font_small)
        draw_board(screen, game.enemy, right_x, right_y, show_ships=False, font_small=font_small)
        # Yerleştirme önizleme
        if game.state == "placing":
            cell = cell_at(mouse_pos, left_x, left_y)
            length = SHIP_SPECS[game.placing_index]
            draw_ship_preview(screen, left_x, left_y, cell, length, game.horizontal)
        # Durum mesajı
        msg = font.render(game.msg, True, (230,230,230))
        screen.blit(msg, (PADDING, PADDING + BOARD_SIZE*CELL + 60))
        # AI bilgileri
        ai_info1 = font_tiny.render(f"Epsilon: {game.ai.epsilon:.3f} | Steps: {game.ai.steps_done} | Episodes: {game.ai.episodes_done}", 
                                   True, (150,150,180))
        screen.blit(ai_info1, (PADDING, HEIGHT - 25))
        mode_text = "Training: AÇIK (T)" if game.training_mode else "Training: KAPALI (T)"
        ai_info2 = font_tiny.render(f"{mode_text} | Model Kaydet (S)", True, (150,150,180))
        screen.blit(ai_info2, (PADDING, HEIGHT - 45))
        # Sıra/Sonuç
        if game.state == "playing":
            turn_txt = "Sıra: SENDE" if game.turn == "player" else "Sıra: AI"
            t = font.render(turn_txt, True, (240,220,150))
            screen.blit(t, (PADDING, PADDING + BOARD_SIZE*CELL + 92))
        elif game.state == "gameover":
            res = font_big.render(game.result, True, (255,220,160))
            screen.blit(res, (PADDING, PADDING + BOARD_SIZE*CELL + 92))
            hint = font_small.render("Yeni oyun için N'ye bas.", True, (200,200,200))
            screen.blit(hint, (PADDING, PADDING + BOARD_SIZE*CELL + 126))
        
        pygame.display.flip()
    
    pygame.quit()

if __name__ == "__main__":
    main()
