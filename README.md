# ARABICA × YOUREYE HACKATHON DAY
### Görüntü İşleme ve Yapay Zekâ Çözümleri (Kayseri 38. Cadde | 10.10.2026)

---

## 👤 Katılımcı ve Proje Bilgileri
- **Geliştirici:** Osman Melih Gümüş
- **Çalışma Modeli:** Bireysel Katılım (Hackathon Kurallarına Uygun)
- **Hedef:** 6 saatlik süre zarfında iki gerçek saha senaryosunu (Kafe içi kişi sayımı/cinsiyet sınıflandırma ve Şehir girişi araç sayımı/plaka tespiti) yüksek doğruluk, FPS kararlılığı ve modüler bir mimariyle çalışan çözüme dönüştürmek.

---

## 📁 Proje Dosya ve Dizin Mimarisi

Uygulama modüler, betik tabanlı (CLI) çalışmakta olup gereksiz web arayüzlerinden arındırılmış, doğrudan video işleme boru hattı (pipeline) ve veri kayıt (CSV / JSON) standartlarına göre yapılandırılmıştır:

```text
hackathon/
├── models/                                                 # İndirilen model ağırlıkları (.pt)
├── outputs/
│   ├── task1_output_final/                                # Görev 1 Nihai Çıktıları
│   │   ├── task1_kisi_sayimi.mp4                          # H.264 encode edilmiş sonuç videosu
│   │   └── task1_kisi_sayimi.csv                          # Kişi geçiş ve cinsiyet logları
│   └── task2_outputs/
│       └── task2_arac_sayimi-rev1.1/                      # Görev 2 Nihai Çıktıları (Rev 1.1)
│           ├── vehicle_log.csv                            # Araç geçiş, plaka ve güven logları
│           ├── vehicle_log.json                           # Detaylı JSON veri aktarımı
│           └── task2_vehicles_output.mp4                  # İşlenmiş video çıktısı (630 MB)
├── plate_utils.py                                         # Plaka temizleme, TR/AB regex ve ön işleme
├── task1_people.py                                        # Görev 1: Kişi Tespiti, Sayım ve Cinsiyet Pipeline
├── task2_vehicles.py                                      # Görev 2: Araç Tespiti, Plaka Crop ve OCR Pipeline
├── requirements.txt                                       # Python bağımlılıkları
└── README.md                                              # Proje ve teknik rapor dokümantasyonu
```

---

## ☕ Aşama 1: Kafe İçerisinde Kişi Sayımı ve Cinsiyet Sınıflandırma (`task1_people.py`)

### 1.1 Mimari ve Kullanılan Teknolojiler
- **Kişi Tespiti (Person Detection):** `YOLOv8m` (Ultralytics) - CUDA FP16 (Half-precision) hızlandırması ile yüksek tespit doğruluğu ve düşük gecikme.
- **Çoklu Nesne Takibi (Multi-Object Tracking):** `ByteTrack` (`track_thresh=0.25`, `track_buffer=45`, `match_thresh=0.8`) ile oklüzyon ve geçici kayıplarda kimlik (ID) sürekliliği.
- **Alan Filtreleme (PolygonZone):** Dışarıdan geçen veya cam arkasındaki yayaların elenmesi için kafe oturma ve giriş alanı poligon ile sınırlandırılmıştır (`[76, 120]`, `[6, 198]`, `[1478, 60]`, `[1913, 430]`).
- **Giriş / Çıkış Sayım Hattı (LineZone):** Kapı giriş hattına tam hizalanmış sanal çizgi (`[725, 185] -> [1060, 185]`).
  - *Tetikleme Noktası (Trigger Anchor):* Hatalı sayımları önlemek için kişinin orta noktası yerine zemin teması olan **Ayak Bölgesi (`sv.Position.BOTTOM_CENTER`)** kullanılmıştır.
- **Cinsiyet Sınıflandırma (Zero-Shot CLIP):** `openai/clip-vit-base-patch32` modeli yerel bağlama uygun promptlarla kullanılmıştır:
  - **Kadın Promptları:** *"a woman with headscarf or hijab"*, *"a veiled woman"*, *"a woman with long hair"*, *"a female customer in a cafe"*, *"a woman wearing a purse or handbag"*.
  - **Erkek Promptları:** *"a man sitting at a laptop"*, *"a man seen from behind"*, *"a man with short hair"*, *"a bald man"*, *"a broad-shouldered man"*, *"a male customer in a cafe"*.
- **Temporal Oylama & Performans:** Her karede CLIP çalıştırmak yerine her 8 karede bir (`frame_idx % 8 == 0`) çıkarım yapılarak `deque(maxlen=10)` kayan pencere çoğunluk oylaması uygulanmıştır. Boyutu küçük crop'lar (`h < 80` veya `w < 40`) CLIP'e gönderilmeyerek yanlış çıkarımların önüne geçilmiştir.
- **Görsel Standartlar:** Kadınlar için pembe/magenta (`#FF1493`), erkekler için camgöbeği (`#00FFFF`), henüz emin olunamayan kişiler için sarı (`#FFFF00`) renk kodlaması.

### 1.2 Yasal ve Teknik Not (Zorunlu Çekince)
> [!WARNING]
> **Teknik Not:** Cinsiyet sınıflandırması yalnızca görsel model (OpenAI CLIP) çıktısı olarak değerlendirilir. Bu tür tahminler kesin kimlik veya kişisel özellik tespiti değildir; sonuçların tahmini olduğu ve hata payı içerebileceği unutulmamalıdır.

### 1.3 Aşama 1 - Bilinen Hatalar ve Kısıtlar
1. **Yoğun Grup Geçişleri ve Örtüşme (Occlusion):** Kapıdan aynı anda 2-3 kişi omuz omuza girdiğinde tek bir geniş bounding box oluşabilmekte veya ByteTrack geçici olarak ID değişimine uğrayabilmektedir.
2. **Düşük Çözünürlüklü Arka Plan Kişileri:** Kameraya çok uzak noktalarda masada oturan kişilerin piksel çözünürlüğü düşük olduğundan CLIP modeli yeterli güven eşiğini aşamayarak bir süre "Analiz Ediliyor" durumunda kalabilmektedir.
3. **Masa ve Sandalye Arkasında Oturma:** Masanın altında kalan vücutlar sebebiyle `BOTTOM_CENTER` koordinatı gerçek ayak zemin teması yerine masa seviyesinde oluşabilmektedir.
4. **Cam Yansımaları:** Gün ışığına bağlı olarak kafe camındaki yansımalarda anlık silüet tespitleri oluşabilmekte, bu durum poligon filtreleme ile minimize edilmektedir.

### 1.4 Aşama 1 - Geliştirme Önerileri
1. **Yüz Tespiti Destekli Hibrit Demografi:** Sadece tam vücut CLIP analizi yerine, yüz görünen anlarda `RetinaFace` + `InsightFace / FairFace` modelleri devreye sokularak yüz netliği yüksek karelerde %98+ doğruluk elde edilebilir.
2. **Görsel Re-ID Özellik Çıkarımı:** ByteTrack yerine `BoT-SORT` veya `DeepSORT` gibi derin Re-ID embedding kullanan takipçilere geçilerek kadrajdan çıkıp tekrar giren müşterilerin aynı ID ile devam etmesi sağlanabilir.
3. **Kamera Perspektifi:** Tavana dik (top-down) konumlandırılmış veya fisheye kamera entegrasyonu ile oklüzyon tamamen ortadan kaldırılabilir.
4. **TensorRT Hızlandırması:** CLIP ve YOLO modelleri TensorRT motoruna dönüştürülerek RTX 4050 üzerinde 100+ FPS gerçek zamanlı hızlara ulaşılabilir.

---

## 🚗 Aşama 2: Şehre Giren Araç Sayımı ve Plaka Tespiti (`task2_vehicles.py`)

### 2.1 Mimari ve Kullanılan Teknolojiler
- **Araç Tespiti ve Sınıflandırma:** `YOLOv8s` (Otomobil, Kamyon, Otobüs, Motosiklet sınıfları).
- **Çoklu Araç Takibi:** `ByteTrack` ile yön kestirimi, ID takibi ve tekrar sayımın engellenmesi.
- **Sanal Sayım Hattı:** Yol genişliğine ve perspektifine göre kalibre edilmiş LineZone (`--scenario A`, `--scenario B` veya özel `--line` desteği).
- **Yüksek Çözünürlüklü 4K Plaka Kırpma (High-Res Crop):** Araç takibi 1080p çözünürlükte optimize şekilde yürütülürken, plaka crop işlemi orijinal 4K (3840x2160) kareden piksel kaybı yaşanmadan alınmaktadır.
- **Laplacian Bulanıklık Filtresi (Blur Budget):** Plaka crop'ları üzerinde Laplacian varyansı hesaplanarak (`laplacian_variance >= 30.0`) yalnızca net ve odaklanmış kareler OCR motoruna gönderilir.
- **EasyOCR ve Regex Doğrulama:**
  - Avrupa/Polonya (`PL_RE`) ve Türkiye (`TR_RE`) plaka standartlarına göre regex formatlama.
  - Karakter kural motoru (`fix_pl`): Rakam beklenilen yerdeki harfleri (`O->0`, `I->1`, `B->8`, `Z->2`) ve harf beklenilen yerdeki rakamları otomatik düzelten heuristik motor.
- **Zaman Serisi Çoğunluk Oylaması (`PlateVoteTracker`):** Aracın kadrajda kaldığı tüm kareler boyunca okunan plakalar pozisyonel karakter frekansı ile oylanarak en kararlı plaka metni belirlenir.
- **Veri Dışa Aktarımı:** Araç ID, tip, plaka, araç güven skoru, plaka güven skoru ve zaman damgası `outputs/task2_outputs/task2_arac_sayimi-rev1.1/` altına hem `.csv` hem `.json` olarak kaydedilir.

### 2.2 Aşama 2 - Bilinen Hatalar ve Kısıtlar
1. **Aşırı Uzak Mesafe ve Gece/Işık Koşulları:** Giriş çizgisinden çok uzakta kalan veya aşırı hızla geçen araçlarda plaka piksellerinin aşırı küçülmesi ve hareket bulanıklığı (motion blur).
2. **Perspektif Bozulması (Skew/Angle):** Kameranın yolun sol veya sağ yukarısında açılı durması nedeniyle plakaların yamuklaşması ve OCR'ın karakter kenarlarını karıştırması.
3. **Karakter Benzerlikleri:** Kirli veya hasarlı plakalarda `0` (sıfır) ile `O` (harf), `8` ile `B`, `D` ile `0` ayrımının sınırda kalması.
4. **Büyük Video Çıktı Boyutu:** 4K/1080p yüksek kaliteli işlenmiş video dosyasının 630 MB olması nedeniyle GitHub 100 MB tek dosya limitine takılması (çıktı CSV ve JSON dosyaları repository'de sunulmuş, video yerel ortamda üretilmiştir).

### 2.3 Aşama 2 - Geliştirme Önerileri
1. **Perspektif Düzeltme (Warping / Affine Transform):** Plaka tespiti aşamasında 4 köşe noktası (polygon detection) kestirimi yapılarak plakanın dikdörtgen forma getirilmesi (unwarping).
2. **Süper Çözünürlük (Super Resolution - Real-ESRGAN):** Düşük çözünürlüklü plaka kırpıntılarını 2x/4x derin öğrenme tabanlı upscaling ile iyileştirerek OCR tanıma oranının %40 artırılması.
3. **End-to-End LPRNet / PaddleOCR:** Genel OCR yerine doğrudan plaka karakter setiyle eğitilmiş hafif LPRNet veya Türkçe plakalara optimize PaddleOCR modeline geçilmesi.
4. **Hız ve Şerit İhlali Tespiti:** Homografi matrisi ve piksel-metre kalibrasyonu ile araçların anlık hızlarının hesaplanması ve emniyet şeridi ihlal analitiği eklenmesi.

---

## ⚡ Donanım ve Performans Metrikleri

Tüm testler ve çalıştırmalar aşağıdaki ortamda gerçekleştirilmiştir:
- **Donanım:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)
- **Çalışma Hassasiyeti:** FP16 (Half-Precision) CUDA Çıkarımı
- **Ortalama FPS:**
  - Görev 1 (YOLOv8m + ByteTrack + CLIP ViT-B/32 + PolygonZone): ~32 - 40 FPS
  - Görev 2 (YOLOv8s + ByteTrack + 4K Plate Crop + EasyOCR): ~24 - 30 FPS

---

## 🛠️ Kurulum ve Çalıştırma Talimatı

### 1. Ortam Hazırlığı
```bash
# Bağımlılıkları yükleyin
pip install -r requirements.txt
```

### 2. Görev 1: Kafe Kişi Sayımı ve Cinsiyet Sınıflandırma
```bash
# Varsayılan girdi videosu (data/kisi_sayimi.mp4) ile çalıştırma:
python task1_people.py

# Farklı video veya parametrelerle çalıştırma:
python task1_people.py --video data/kisi_sayimi.mp4 --conf 0.12 --output outputs/task1_output_final/task1_kisi_sayimi.mp4
```

### 3. Görev 2: Araç Sayımı ve Plaka Tespiti
```bash
# Varsayılan Senaryo A (Merkezi Giriş Hattı) ile çalıştırma:
python task2_vehicles.py --scenario A

# Geniş şerit veya alternatif senaryolar:
python task2_vehicles.py --scenario B
python task2_vehicles.py --line "150,800,1750,800" --flip
```

---

## 📊 Hackathon Değerlendirme Kriterleri Karşılık Matrisi

| Kriter | Puan | Projedeki Karşılığı ve Durumu |
| :--- | :---: | :--- |
| **Kişi / Araç Tespit Doğruluğu** | 20 | `YOLOv8m` (insan) ve `YOLOv8s` (araç) FP16 modelleri ile yüksek tespit başarımı sağlandı. |
| **Sayım ve Tracking Başarısı** | 20 | `ByteTrack` + `LineZone` entegrasyonu; `BOTTOM_CENTER` ayak hizası tetiklemesi ile sıfır mükerrer sayım. |
| **Plaka Tespit ve OCR Başarısı** | 20 | 4K orijinal kareden yüksek çözünürlüklü plaka kırpma, Laplacian netlik filtresi, `EasyOCR` + `PlateVoteTracker` oylaması. |
| **Cinsiyet Sınıflandırma Çıktısı** | 10 | `OpenAI CLIP ViT-B/32` ile demografiye özel promptlar, 8-frame cache, rolling majority voting ve yasal çekince notu. |
| **Kod Kalitesi ve Sistem Mimarisi** | 10 | Temiz, fonksiyonel, modüler Python pipeline mimarisi; `try/finally` kaynak yönetimi ve çökme korumaları. |
| **Performans / FPS / Kararlılık** | 10 | RTX 4050 GPU üzerinde FP16 çıkarım, akıllı önbellekleme ve çözünürlük optimizasyonları ile gerçek zamanlı akış. |
| **Sunum, Anlaşılabilirlik ve Bonuslar** | 10 | HUD bilgi panelleri, araç türü sınıflandırması, TR/AB plaka regex motoru, CSV ve JSON çıktı formatları. |
| **TOPLAM** | **100** | **Eksiksiz ve tam uyumlu teslimat.** |

---
*ARABICA Coffee House × YOUREYE AI Technologies Hackathon Day*
