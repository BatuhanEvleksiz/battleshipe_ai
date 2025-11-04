# Battleship (Amiral Battı) - Deep Q-Learning AI

Bu proje, Deep Q-Learning (DQN) kullanarak kendi kendini eğitebilen bir Amiral Battı AI'sı içerir. AI, self-play yöntemiyle kendine karşı oynayarak strateji öğrenir.

## 🎮 Özellikler

### AI Yetenekleri
- **Deep Q-Network (DQN)**: 5 katmanlı derin sinir ağı
- **Stratejik Karar Verme**: Olasılık haritaları ve pattern tanıma
- **Self-Play Eğitim**: Kendine karşı oynayarak öğrenme
- **Experience Replay**: Geçmiş deneyimlerden öğrenme
- **Target Network**: Stabilize edilmiş öğrenme

### AI'nın Öğrendiği Stratejiler
1. **Akıllı Atış Seçimi**: Gemi olasılığı yüksek bölgeleri hedefleme
2. **Pattern Tanıma**: Ardışık vuruşları takip etme ve hat oluşturma
3. **Olasılık Hesaplama**: Her kare için gemi bulunma olasılığını hesaplama
4. **Çevresel Farkındalık**: Vuruş sonrası komşu kareleri önceliklendirme
5. **Boyut Tahmini**: Kalan gemi boyutlarına göre strateji belirleme

## 📦 Kurulum

### Gereksinimler
```bash
pip install -r requirements.txt
```

Veya manuel kurulum:
```bash
pip install pygame torch numpy matplotlib
```

## 🚀 Kullanım

### 1. Ana Oyunu Çalıştırma
```bash
python battleship_ai_dqn.py
```

#### Oyun Kontrolleri
- **Mouse**: Gemi yerleştirme ve atış yapma
- **R**: Gemi yönünü değiştir (yatay/dikey)
- **N**: Yeni oyun başlat
- **T**: Training modunu aç/kapa
- **S**: Modeli kaydet

### 2. AI'yı Eğitme (Self-Play)

#### Hızlı Eğitim (500 oyun)
```bash
python train_self_play.py
```

#### Özel Sayıda Oyunla Eğitim
```bash
python train_self_play.py train 1000  # 1000 oyun
python train_self_play.py train 5000  # 5000 oyun
```

#### Sadece Test Etme
```bash
python train_self_play.py test
```

## 🧠 AI Mimarisi

### Neural Network Yapısı
```
Input Layer (320 features)
    ↓
Hidden Layer 1 (256 neurons) + BatchNorm + ReLU + Dropout
    ↓
Hidden Layer 2 (512 neurons) + BatchNorm + ReLU + Dropout
    ↓
Hidden Layer 3 (512 neurons) + BatchNorm + ReLU + Dropout
    ↓
Hidden Layer 4 (256 neurons) + BatchNorm + ReLU + Dropout
    ↓
Output Layer (100 Q-values)
```

### State Representation (320 features)
1. **Tahta Kanalları (300)**: 
   - Hit channel (100)
   - Miss channel (100)
   - Unknown channel (100)

2. **Ekstra Özellikler (20)**:
   - Heat map (ısı haritası) istatistikleri
   - Toplam vuruş/iska/bilinmeyen sayıları
   - Ardışık vuruş pattern'leri
   - Olasılık haritası istatistikleri
   - Kenar ve köşe analizleri

### Reward System
```python
# Vuruş: +10
# Gemi batırma: +50 + (gemi_boyu * 20)
# Iska: -2
# Ardışık vuruş bonusu: +5
# Oyun kazanma: +100
# Oyun kaybetme: -50
```

## 📊 Training İstatistikleri

Eğitim sırasında şu metrikler takip edilir:
- **Win Rate**: Kazanma oranı
- **Hit Rate**: İsabet oranı
- **Game Length**: Ortalama oyun uzunluğu
- **Epsilon**: Keşif oranı (exploration rate)

Grafikler otomatik olarak `training_progress_[timestamp].png` olarak kaydedilir.

## 🎯 Performans Beklentileri

### Eğitim Aşamaları

1. **0-100 Oyun**: Rastgele atışlar, temel pattern öğrenme
2. **100-500 Oyun**: Vuruş takibi, basit stratejiler
3. **500-1000 Oyun**: Olasılık hesaplama, gelişmiş stratejiler
4. **1000+ Oyun**: Optimizasyon, ince ayar

### Beklenen Sonuçlar (1000 oyun sonrası)
- Random AI'ya karşı kazanma oranı: >%85
- Ortalama isabet oranı: >%40
- Ortalama oyun süresi: <60 hamle

## 🔧 Hyperparameter Tuning

`battleship_ai_dqn.py` dosyasında değiştirilebilir parametreler:

```python
LEARNING_RATE = 0.001      # Öğrenme hızı
GAMMA = 0.95               # Discount factor
EPSILON_START = 1.0        # Başlangıç keşif oranı
EPSILON_END = 0.01         # Minimum keşif oranı
EPSILON_DECAY = 0.995      # Keşif azalma oranı
BATCH_SIZE = 32            # Batch boyutu
MEMORY_SIZE = 10000        # Experience replay bellek boyutu
TARGET_UPDATE = 100        # Target network güncelleme sıklığı
```

## 📝 Model Dosyaları

- `battleship_dqn_model.pth`: Eğitilmiş model weights
- `training_stats_[episode].json`: Eğitim istatistikleri
- `training_progress_[timestamp].png`: İlerleme grafikleri

## 🚦 Training İpuçları

1. **İlk Eğitim**: En az 500-1000 oyunla başlayın
2. **İteratif Eğitim**: Modeli yükleyip üzerine eğitmeye devam edebilirsiniz
3. **Overfitting Kontrolü**: Hit rate çok yüksekse (%60+) overfitting olabilir
4. **Exploration**: Epsilon değeri çok hızlı düşüyorsa EPSILON_DECAY'i artırın

## 🎮 Oyun İçi AI Davranışları

AI şu davranışları sergiler:

1. **Hunt Mode**: Sistematik arama (checkerboard pattern)
2. **Target Mode**: Vuruş sonrası komşu karelere odaklanma
3. **Line Extension**: Ardışık vuruşları hatta devam ettirme
4. **Probability Mapping**: Olası gemi konumlarını hesaplama
5. **Smart Recovery**: Başarısız pattern sonrası yeni strateji

## 🐛 Bilinen Sorunlar ve Çözümler

1. **CUDA Hatası**: CPU kullanmak için `device = torch.device("cpu")` yapın
2. **Bellek Tükenmesi**: MEMORY_SIZE'ı düşürün
3. **Yavaş Eğitim**: BATCH_SIZE'ı düşürün veya TARGET_UPDATE'i artırın

## 📈 Gelecek İyileştirmeler

- [ ] Multi-agent training (3+ AI)
- [ ] Prioritized experience replay
- [ ] Dueling DQN architecture
- [ ] Noisy networks for exploration
- [ ] Curriculum learning (kolay → zor)
- [ ] Transfer learning for different board sizes
- [ ] Real-time visualization of AI thinking

## 📜 Lisans

Bu proje eğitim amaçlıdır. Özgürce kullanabilir ve değiştirebilirsiniz.

## 🤝 Katkıda Bulunma

İyileştirme önerileriniz varsa, lütfen pull request gönderin veya issue açın!

---

**Not**: İlk eğitim biraz zaman alabilir. Sabırlı olun, AI zamanla gelişecektir! 🚀
