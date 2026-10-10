"""Plaka görüntü işleme, OCR ve regex doğrulama yardımcı modülü.

Arabica x Youreye Hackathon 2026 - Görev 2
"""

import re
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Polonya Plaka Regex ve Düzeltme Kuralları
# ---------------------------------------------------------------------------
PL_RE = re.compile(r'^([A-Z]{2,3})(\d[A-Z0-9]{3,4})$')   # örn WI1145K, WPN12345
D2L = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '2': 'Z'}
L2D = {'O': '0', 'I': '1', 'S': '5', 'B': '8', 'Z': '2'}


def fix_pl(t: str) -> Optional[str]:
    """Polonya plaka karakter karışıklıklarını düzeltir ve doğrular."""
    if not t or len(t) < 5:
        return None
    # İlk 2 karakter harf olmalıdır (0->O, 1->I vb.)
    t_fixed = ''.join(D2L.get(c, c) for c in t[:2]) + t[2:]
    for L in (2, 3):
        cand = t_fixed[:L] + L2D.get(t_fixed[L:L+1], t_fixed[L:L+1]) + t_fixed[L+1:]
        if 6 <= len(cand) <= 8 and PL_RE.match(cand):
            return cand
    return None


def format_pl_display(cand: str) -> str:
    """Polonya plakasını 'ÖNEK GÖVDE' formatına getirir (Örn: WI 1145K, WPI 12345)."""
    m = PL_RE.match(cand)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    return cand


# ---------------------------------------------------------------------------
# Temizleme ve Eski Format Fonksiyonları (Geriye Uyumluluk İçin Bırakılmıştır)
# ---------------------------------------------------------------------------
TR_PLATE_PATTERNS = [
    r'^(0[1-9]|[1-7][0-9]|8[01])[A-Z]{1}[0-9]{4,5}$',
    r'^(0[1-9]|[1-7][0-9]|8[01])[A-Z]{2}[0-9]{3,4}$',
    r'^(0[1-9]|[1-7][0-9]|8[01])[A-Z]{3}[0-9]{2,3}$',
]
EU_INTL_PATTERN = r'^[A-Z]{1,3}[0-9A-Z]{4,6}$'


def clean_plate_text(raw_text: str) -> str:
    """OCR ham metninden boşlukları, özel karakterleri temizler ve büyük harfe çevirir."""
    if not raw_text:
        return ""
    tr_map = str.maketrans("çğıöşüİĞÜŞÖÇ", "CGIOSUIGUSOC")
    cleaned = raw_text.translate(tr_map).upper()
    cleaned = re.sub(r'[^A-Z0-9]', '', cleaned)
    return cleaned


def is_valid_tr_plate(plate_text: str) -> Tuple[bool, str]:
    """TR plaka kontrolü (eski boru hattı)."""
    cleaned = clean_plate_text(plate_text)
    if not (6 <= len(cleaned) <= 8):
        return False, cleaned
    for pattern in TR_PLATE_PATTERNS:
        if re.match(pattern, cleaned):
            il = cleaned[:2]
            rest = cleaned[2:]
            letters = "".join([c for c in rest if c.isalpha()])
            numbers = "".join([c for c in rest[len(letters):] if c.isdigit()])
            return True, f"{il} {letters} {numbers}".strip()
    return False, cleaned


def is_valid_eu_or_intl_plate(plate_text: str) -> Tuple[bool, str]:
    """AB / Uluslararası plaka kontrolü (eski boru hattı)."""
    cleaned = clean_plate_text(plate_text)
    if not (5 <= len(cleaned) <= 9):
        return False, cleaned
    if re.match(EU_INTL_PATTERN, cleaned):
        m = re.match(r'^([A-Z]{1,3})([0-9A-Z]{4,6})$', cleaned)
        if m:
            return True, f"{m.group(1)} {m.group(2)}"
        return True, cleaned
    return False, cleaned


# ---------------------------------------------------------------------------
# EasyOCR Okuma ve Recognize Entegrasyonu
# ---------------------------------------------------------------------------
class LicensePlateOCR:
    """EasyOCR tabanlı, dedektörsüz 'recognize' kullanan yüksek hızlı plaka okuyucu."""

    def __init__(self, gpu: bool = True):
        import easyocr
        self.reader = easyocr.Reader(['en'], gpu=gpu, verbose=False)

    def read_plate(self, plate_crop: np.ndarray) -> Tuple[str, float, str]:
        """Kırpılan plakadan hızlı ve tek geçişli OCR okuması yapar.
        
        Dönüş:
            (display_text, ham_conf, plate_type)
            plate_type: 'PL' veya 'RAW'
        """
        if plate_crop is None or plate_crop.size == 0:
            return "", 0.0, "UNKNOWN"

        h, w = plate_crop.shape[:2]
        if h < 10 or w < 20:
            return "", 0.0, "UNKNOWN"

        crop_to_proc = plate_crop

        # 1. Sol %12'lik mavi şerit kontrolü (AB Polonya bandı)
        strip_w = max(1, int(w * 0.12))
        strip = crop_to_proc[:, :strip_w]
        b_mean = float(strip[:, :, 0].mean())
        r_mean = float(strip[:, :, 2].mean())
        if b_mean > r_mean + 25:
            crop_to_proc = crop_to_proc[:, strip_w:]

        # 2. Gri tonlama, en-boy oranını koruyarak yükseklik = 80 px (INTER_CUBIC) + CLAHE
        gray = cv2.cvtColor(crop_to_proc, cv2.COLOR_BGR2GRAY)
        target_h = 80
        scale = target_h / max(gray.shape[0], 1)
        new_w = max(1, int(gray.shape[1] * scale))
        resized = cv2.resize(gray, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        proc = clahe.apply(resized)

        # 3. Metin dedektörsüz doğrudan recognize
        gh, gw = proc.shape
        try:
            res = self.reader.recognize(
                proc,
                horizontal_list=[[0, gw, 0, gh]],
                free_list=[],
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                decoder='greedy'
            )
        except Exception:
            return "", 0.0, "UNKNOWN"

        if not res:
            return "", 0.0, "UNKNOWN"

        # Tek okuma sonucunu kullan, parçaları birleştirme
        raw_text = res[0][1]
        ham_conf = float(res[0][2])
        cleaned = clean_plate_text(raw_text)

        # 4 karakterden kısa okumaları at
        if len(cleaned) < 4:
            return "", 0.0, "UNKNOWN"

        # Polonya doğrulaması
        fixed = fix_pl(cleaned)
        if fixed:
            display_text = format_pl_display(fixed)
            return display_text, ham_conf, "PL"

        return cleaned, ham_conf, "RAW"


# ---------------------------------------------------------------------------
# Çoklu Kare Oylama ve Plaka Takipçisi (PlateVoteTracker)
# ---------------------------------------------------------------------------
class PlateVoteTracker:
    """Takip edilen her araç ID'si için okumaları biriktirir ve konumsal çoğunluk oylaması yapar."""

    def __init__(self):
        # track_id -> List[Tuple[text, ham_conf, p_type, quality_score]]
        self.history: Dict[int, List[Tuple[str, float, str, float]]] = {}
        # track_id -> en iyi plaka kırpıntı görseli (skor, np.ndarray)
        self.best_plate_crops: Dict[int, Tuple[float, np.ndarray]] = {}

    def add_prediction(
        self,
        track_id: int,
        plate_text: str,
        conf: float,
        p_type: str,
        plate_crop: Optional[np.ndarray] = None
    ):
        """Tahmin ve kırpıntı kalitesini kaydeder."""
        if not plate_text or conf <= 0.0 or p_type == "UNKNOWN":
            return

        # Plaka kırpıntı kalite skoru: alan * Laplacian varyansı
        quality_score = 0.0
        if plate_crop is not None and plate_crop.size > 0:
            h, w = plate_crop.shape[:2]
            gray_crop = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY) if len(plate_crop.shape) == 3 else plate_crop
            lap_var = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
            quality_score = (h * w) * lap_var

            # En kaliteli plaka crop'unu sakla
            if track_id not in self.best_plate_crops or quality_score > self.best_plate_crops[track_id][0]:
                self.best_plate_crops[track_id] = (quality_score, plate_crop.copy())

        if track_id not in self.history:
            self.history[track_id] = []
        self.history[track_id].append((plate_text, conf, p_type, quality_score))

    def get_best_plate(self, track_id: int) -> Tuple[str, float, str]:
        """Aracın oylanmış nihai plakasını döndürür.
        
        - 'PL' okumalarını uzunluğa göre grupla, en yüksek conf ağırlıklı grubu seç.
        - Grup içinde her karakter pozisyonu için conf-ağırlıklı çoğunluk oyu uygula.
        - PL yoksa en yüksek ham conf'lu RAW okumayı döndür.
        - Hiç okuma yoksa ('PLAKA YOK', 0.0, 'UNKNOWN').
        """
        records = self.history.get(track_id, [])
        if not records:
            return "PLAKA YOK", 0.0, "UNKNOWN"

        pl_records = [(t, c, q) for t, c, pt, q in records if pt == "PL"]

        if pl_records:
            # Temizlenmiş (boşluksuz) halleriyle uzunluğa göre grupla
            by_len: Dict[int, List[Tuple[str, float]]] = {}
            len_weights: Dict[int, float] = {}
            for t, c, _ in pl_records:
                raw_c = clean_plate_text(t)
                L = len(raw_c)
                by_len.setdefault(L, []).append((raw_c, c))
                len_weights[L] = len_weights.get(L, 0.0) + c

            # En yüksek conf ağırlıklı uzunluk grubunu seç
            best_L = max(len_weights, key=len_weights.get)
            group = by_len[best_L]

            # Karakter bazlı konumsal çoğunluk oylaması (conf ağırlıklı)
            voted_chars = []
            for pos in range(best_L):
                char_scores: Dict[str, float] = {}
                for text, conf in group:
                    ch = text[pos]
                    char_scores[ch] = char_scores.get(ch, 0.0) + conf
                best_ch = max(char_scores, key=char_scores.get)
                voted_chars.append(best_ch)

            voted_clean = "".join(voted_chars)
            # Polonya formatına yeniden uygunlaştır ve formatla
            fixed = fix_pl(voted_clean)
            if fixed:
                final_text = format_pl_display(fixed)
            else:
                final_text = format_pl_display(voted_clean)

            # Nihai güven = gruptaki okumaların ortalama ham conf değeri
            avg_conf = sum(c for _, c in group) / len(group)
            return final_text, avg_conf, "PL"

        # PL okuma yoksa en yüksek conf'lu RAW okuma
        raw_records = [(t, c) for t, c, pt, _ in records if pt == "RAW"]
        if raw_records:
            best_raw = max(raw_records, key=lambda x: x[1])
            return best_raw[0], best_raw[1], "RAW"

        return "PLAKA YOK", 0.0, "UNKNOWN"

    def get_best_plate_crop(self, track_id: int) -> Optional[np.ndarray]:
        """Aracın bellekte tutulan en kaliteli plaka kırpıntısını döndürür."""
        if track_id in self.best_plate_crops:
            return self.best_plate_crops[track_id][1]
        return None
