# Arabica x Youreye Hackathon 2026

## 👥 Katılımcı Bilgisi
- **Geliştirici:** Osman Melih Gümüş

---

## 📌 Proje Genel Bakışı
Bu proje, Arabica x Youreye Hackathon 2026 kapsamında geliştirilen nesne tespiti, nesne takibi, cinsiyet tahmini, plaka tespiti/OCR ve canlı analiz platformudur.

### 🎯 Proje Milestoneları & Durumu

- [x] **Milestone 1: Proje Kurulumu ve Altyapı**
  - [x] Modüler proje mimarisinin ve dizin yapısının oluşturulması
  - [x] Bağımlılıkların tanımlanması (`requirements.txt`) ve `plate_utils.py` yardımcı modülü
  - [x] Git versiyon kontrol altyapısının kurulması

- [x] **Milestone 2: Görev 1 - Kişi Tespiti, Sayım ve Cinsiyet Tahmini (`task1_people.py`)**
  - [x] YOLO11 (Ultralytics) ile kişilerin tespiti
  - [x] ByteTrack ve Supervision LineZone entegrasyonu ile hassas giriş/çıkış sayımı
  - [x] CLIP (Zero-Shot) tabanlı cinsiyet tahmini ve zaman aşımı oylama sistemi
  - [x] Detaylı veri kayıt altyapısı (CSV log & ROI kişi görselleri kırpma)

- [x] **Milestone 3: Görev 2 - Araç Tespiti, Araç Türü, Plaka Kırpma ve OCR (`task2_vehicles.py`)**
  - [x] Araç sınıflarının (Otomobil, Motosiklet, Otobüs, Kamyon) tespiti ve takibi
  - [x] Supervision LineZone ile çift yönlü araç sayımı, `--line` ve `--flip` desteği, HUD bilgi paneli
  - [x] YOLO Plaka tespiti ve 4K orijinal kareden yüksek çözünürlüklü plaka kırpma (High-Res Crop)
  - [x] Polonya / AB plaka regex doğrulaması (`PL_RE`), karakter düzeltme (`fix_pl`) ve formatlama
  - [x] EasyOCR `recognize` ile tek geçişli, mavi şerit kesmeli yüksek hızlı OCR okuması
  - [x] Karakter bazlı konumsal çoğunluk oylaması (`PlateVoteTracker`) ve bütçe/kapı kontrolleri
  - [x] Geçiş loglarının CSV ve JSON olarak kaydı (`outputs/task2_arac_sayimi-revX.X/`)

- [ ] **Milestone 4: Görselleştirme ve Dashboard (`app.py`)**
  - [ ] Streamlit interaktif kontrol paneli ve sonuç analiz ekranı

---

## 🏗️ Proje Dizin Yapısı & Revizyon Dosyalaması

Çıktılar revizyon ve görev bazlı olarak `outputs/` dizininde versiyonlanmaktadır:

```
hackathon/
├── data/                            # Ham video verileri
├── models/                          # Eğitilmiş/indirilen model ağırlıkları (.pt)
├── outputs/                         # Revizyon çıktı klasörleri
│   ├── task1_kisi_sayimi-rev1.0/   # Görev 1 revizyon çıktıları
│   ├── task2_arac_sayimi-rev1.0/   # Görev 2 revizyon çıktıları (video, CSV, JSON, crops)
│   └── crops/                       # Genel kırpıntı dizini
├── task1_people.py                  # Görev 1: Kişi Tespiti, Sayım ve Cinsiyet
├── task2_vehicles.py                # Görev 2: Araç Tespiti, Sayım, Plaka Kırpma ve OCR
├── plate_utils.py                   # Polonya/TR Plaka regex, görüntü ön işleme ve OCR yardımcıları
├── app.py                           # Streamlit analiz dashboard'u
├── requirements.txt                 # Bağımlılık listesi
└── README.md                        # Dokümantasyon
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Bağımlılıkların Yüklenmesi
```bash
pip install -r requirements.txt
```

### 2. Görev 1 (Kişi Tespiti & Cinsiyet Tahmini)
```bash
python task1_people.py --video data/kisi_sayimi.mp4
```

### 3. Görev 2 (Araç Tespiti, Sayım ve Plaka OCR)
```bash
# Varsayılan Çalıştırma (Senaryo A - Orta Koridor)
python task2_vehicles.py --scenario A

# Tüm Yolu Kapsayan Geniş Çizgi (Senaryo B)
python task2_vehicles.py --scenario B

# Özel Çizgi ve Yön Çevirme İle Çalıştırma
python task2_vehicles.py --line "150,800,1750,800" --flip
```

---

## ⚠️ Yasal ve Teknik Çekince (Disclaimer)

> [!WARNING]
> **Önemli Not:** Hackathon kuralları ve etik yapay zeka standartları gereği; **bu model çıktısı yalnızca tahmindir, kesin kimlik doğrulama niteliği taşımaz**. Cinsiyet ve demografik analiz tahminleri istatistiksel çıkarımlara dayanmaktadır.
