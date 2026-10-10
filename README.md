# Arabica x Youreye Hackathon 2026

## 👥 Takım / Katılımcı Bilgileri
- **Takım Adı:** [Takım Adı Buraya]
- **Takım Üyeleri:**
  - Üye 1: [Ad Soyad - Rol / E-posta / GitHub]
  - Üye 2: [Ad Soyad - Rol / E-posta / GitHub]

---

## 📌 Proje Genel Bakışı
Bu proje, Arabica x Youreye Hackathon 2026 kapsamında geliştirilen görüntü işleme, nesne tespiti, takip, plaka tanıma ve analiz platformudur.

### Görev Dağılımı ve Mimarisi
1. **Görev 1 (Task 1): Kişi Tespiti, Sayım ve Cinsiyet Tahmini** (`task1_people.py`)
   - Kişi tespiti (Object Detection & Tracking)
   - Giriş / Çıkış ve bölge bazlı kişi sayımı
   - Demografik/Cinsiyet analizi

2. **Görev 2 (Task 2): Araç Tespiti, Sayım, Plaka Kırpma ve OCR** (`task2_vehicles.py`, `plate_utils.py`)
   - Araç sınıflarının tespiti ve takibi
   - Araç sayımı
   - Araç plakası lokalizasyonu ve kırpılması (crops)
   - TR formatına uygun plaka regex doğrulaması ve OCR okuması

3. **Görselleştirme & Dashboard** (`app.py`)
   - Streamlit tabanlı interaktif kontrol paneli ve canlı analiz ekranı

---

## 🏗️ Proje Mimarisi & Dizin Yapısı

```
hackathon/
├── data/                  # Ham veya test video/görüntü verileri (Git tarafından takip edilmez)
├── models/                # Eğitilmiş veya indirilen ağırlık dosyaları (.pt, .onnx vb.)
├── outputs/               # Analiz çıktıları, video ve raporlar
│   └── crops/             # Kırpılan plaka veya nesne görselleri
├── task1_people.py        # Görev 1: Kişi Tespiti, Sayım ve Cinsiyet
├── task2_vehicles.py      # Görev 2: Araç Tespiti, Sayım, Plaka Kırpma ve OCR
├── plate_utils.py         # TR Plaka regex ve ön işleme yardımcı modülleri
├── app.py                 # Streamlit analiz dashboard'u
├── requirements.txt       # Proje bağımlılıkları
├── README.md              # Proje dokümantasyonu
└── .gitignore             # Versiyon kontrol kural dosyası
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Sanal Ortam Oluşturma ve Aktifleştirme
```bash
# Sanal ortam oluşturma
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate
```

### 2. Bağımlılıkların Yüklenmesi
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Uygulamayı Başlatma
```bash
streamlit run app.py
```

---

## ⚠️ Yasal ve Teknik Çekince (Disclaimer)

> [!WARNING]
> **Önemli Not:** Hackathon kuralları ve etik yapay zeka standartları gereği; **bu model çıktısı yalnızca tahmindir, kesin kimlik doğrulama niteliği taşımaz**. Cinsiyet ve demografik analiz tahminleri istatistiksel çıkarımlara dayanmakta olup hiçbir resmi veya yasal doğruluğu temsil etmez.
