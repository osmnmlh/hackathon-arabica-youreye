"""Plaka görüntü işleme, OCR ve regex doğrulama yardımcı modülü.

Arabica x Youreye Hackathon 2026 - Görev 2
"""

import re
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# TR ve Uluslararası Plaka Regex Kuralları
# ---------------------------------------------------------------------------
# Türkiye standart plaka formatları:
# 1. 2 rakam (01-81), 2 harf, 3-4 rakam: 34 AB 123, 34 AB 1234
# 2. 2 rakam (01-81), 3 harf, 2-3 rakam: 06 ABC 12, 06 ABC 123
# 3. 2 rakam (01-81), 1 harf, 4-5 rakam: 34 A 1234, 34 A 12345
# 4. İl plaka kodları: 01 - 81
TR_PLATE_PATTERNS = [
    r'^(0[1-9]|[1-7][0-9]|8[01])[A-Z]{1}[0-9]{4,5}$',     # 34 A 1234
    r'^(0[1-9]|[1-7][0-9]|8[01])[A-Z]{2}[0-9]{3,4}$',     # 34 AB 123 / 1234
    r'^(0[1-9]|[1-7][0-9]|8[01])[A-Z]{3}[0-9]{2,3}$',     # 34 ABC 12 / 123
]

# Genel AB / Uluslararası Format (Polonya vb. dahil):
# Örn: WI 1145K, WI 4619C, WPI 12345, B 1234 CD
# 1-3 Harf + boşluk/tire opsiyonel + 4-6 Alfanümerik karakter (Toplam 5-9 karakter)
EU_INTL_PATTERN = r'^[A-Z]{1,3}[0-9A-Z]{4,6}$'


# OCR Karakter Karışıklığı Düzeltme Haritaları
CHAR_TO_NUM = {
    'O': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'D': '0', 'Q': '0'
}
NUM_TO_CHAR = {
    '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G', '4': 'A'
}


def clean_plate_text(raw_text: str) -> str:
    """OCR ham metninden boşlukları, özel karakterleri temizler ve büyük harfe çevirir."""
    if not raw_text:
        return ""
    # Türkçe karakterleri İngilizce eşleniklerine çevir
    tr_map = str.maketrans("çğıöşüİĞÜŞÖÇ", "CGIOSUIGUSOC")
    cleaned = raw_text.translate(tr_map).upper()
    # Sadece A-Z ve 0-9 tut
    cleaned = re.sub(r'[^A-Z0-9]', '', cleaned)
    return cleaned


def is_valid_tr_plate(plate_text: str) -> Tuple[bool, str]:
    """Plakanın resmi Türkiye plaka standartlarına uyup uymadığını doğrular.
    
    Dönüş: (Geçerli mi, formatlanmış metin)
    """
    cleaned = clean_plate_text(plate_text)
    if not (6 <= len(cleaned) <= 8):
        return False, cleaned

    for pattern in TR_PLATE_PATTERNS:
        if re.match(pattern, cleaned):
            # İl kodu ve gövdeyi estetik formatla (Örn: 34 ABC 123)
            il = cleaned[:2]
            rest = cleaned[2:]
            letters = "".join([c for c in rest if c.isalpha()])
            numbers = "".join([c for c in rest[len(letters):] if c.isdigit()])
            formatted = f"{il} {letters} {numbers}".strip()
            return True, formatted
            
    return False, cleaned


def is_valid_eu_or_intl_plate(plate_text: str) -> Tuple[bool, str]:
    """Avrupa / Uluslararası plaka standartlarına uygunluğu denetler.
    
    Örn: WI 1145K, WI 4619C
    """
    cleaned = clean_plate_text(plate_text)
    if not (5 <= len(cleaned) <= 9):
        return False, cleaned

    if re.match(EU_INTL_PATTERN, cleaned):
        # İlk harf grubu ile devamını ayır (Örn: WI 1145K)
        m = re.match(r'^([A-Z]{1,3})([0-9A-Z]{4,6})$', cleaned)
        if m:
            return True, f"{m.group(1)} {m.group(2)}"
        return True, cleaned

    return False, cleaned


def validate_and_format_plate(raw_text: str) -> Tuple[str, str, bool]:
    """Akıllı İkili Doğrulama (Hybrid Validation).
    
    Önce TR formatını dener (Bonus kriteri).
    Uymazsa Genel AB/Uluslararası formatı dener (Video uyumu).
    
    Dönüş:
        (formatted_text, plate_type, is_valid)
        plate_type: 'TR', 'EU_INTL', 'UNKNOWN'
    """
    cleaned = clean_plate_text(raw_text)
    if not cleaned or len(cleaned) < 4:
        return cleaned, "UNKNOWN", False

    # 1. TR Kontrolü
    tr_valid, tr_formatted = is_valid_tr_plate(cleaned)
    if tr_valid:
        return tr_formatted, "TR", True

    # 2. Uluslararası / AB Kontrolü
    eu_valid, eu_formatted = is_valid_eu_or_intl_plate(cleaned)
    if eu_valid:
        return eu_formatted, "EU_INTL", True

    # 3. Kısmi eşleşme / format dışı ama okunabilmiş plaka
    return cleaned, "RAW", False


# ---------------------------------------------------------------------------
# Görüntü Ön İşleme (Pre-processing) Fonksiyonları
# ---------------------------------------------------------------------------
def preprocess_plate_image(plate_crop: np.ndarray, target_height: int = 120) -> List[np.ndarray]:
    """Kırpılan plaka görüntüsünü OCR için çoklu varyasyonlarla hazırlar.
    
    Döndürülen liste, EasyOCR'ın en iyi sonucu yakalaması için farklı
    kontrast ve filtreleme aşamalarını içerir.
    """
    if plate_crop is None or plate_crop.size == 0:
        return []

    h, w = plate_crop.shape[:2]
    if h == 0 or w == 0:
        return []

    # 1. En-boy oranını koruyarak yeniden boyutlandır (Super-resolution yerine 2x-3x akıllı resize)
    scale = target_height / max(h, 1)
    new_w = int(w * scale)
    resized = cv2.resize(plate_crop, (new_w, target_height), interpolation=cv2.INTER_CUBIC)

    variations = []

    # Varyasyon A: Renkli Büyütülmüş (EasyOCR renkli görüntülerde de çok iyidir)
    variations.append(resized)

    # Varyasyon B: Gri Seviye + CLAHE + Keskinleştirme
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    
    # Hafif Unsharp Masking (Keskinleştirme)
    gaussian = cv2.GaussianBlur(enhanced_gray, (0, 0), 2.0)
    sharpened = cv2.addWeighted(enhanced_gray, 1.5, gaussian, -0.5, 0)
    variations.append(sharpened)

    # Varyasyon C: Otsu Eşikleme (Binarization) - Yansımaları ve parlamaları kırar
    _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variations.append(thresh)

    return variations


# ---------------------------------------------------------------------------
# EasyOCR Okuma ve Oylama (Voting) Mekanizması
# ---------------------------------------------------------------------------
class LicensePlateOCR:
    """EasyOCR tabanlı, GPU hızlandırmalı ve oylama mekanizmalı plaka okuyucu."""

    def __init__(self, gpu: bool = True):
        import easyocr
        # Plakalar sadece İngilizce karakter ve rakamlardan oluşur
        self.reader = easyocr.Reader(['en'], gpu=gpu, verbose=False)

    def read_plate(self, plate_crop: np.ndarray) -> Tuple[str, float, str]:
        """Tek bir plaka crop'undan en iyi okumayı çıkarır.
        
        Dönüş: (plate_text, confidence, plate_type)
        """
        if plate_crop is None or plate_crop.size == 0:
            return "", 0.0, "UNKNOWN"

        candidates = preprocess_plate_image(plate_crop)
        best_text = ""
        best_conf = 0.0
        best_type = "UNKNOWN"

        for img in candidates:
            try:
                results = self.reader.readtext(
                    img,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    paragraph=False,
                    decoder='greedy'
                )
                
                # Birden fazla metin parçası varsa birleştir (Örn: "WI" ve "1145K")
                combined_text = ""
                total_conf = 0.0
                count = 0

                for bbox, text, conf in results:
                    cleaned = clean_plate_text(text)
                    if cleaned:
                        combined_text += cleaned
                        total_conf += conf
                        count += 1

                if combined_text and count > 0:
                    avg_conf = total_conf / count
                    formatted_text, p_type, is_valid = validate_and_format_plate(combined_text)
                    
                    # Geçerli bir plaka formatı ise güven skoruna bonus ekle
                    effective_score = avg_conf + (0.25 if is_valid else 0.0)
                    
                    if effective_score > best_conf:
                        best_conf = effective_score
                        best_text = formatted_text
                        best_type = p_type

            except Exception:
                continue

        # Normalleştirilmiş güven skoru (0-1 arasında tut)
        best_conf = min(best_conf, 1.0)
        return best_text, best_conf, best_type


class PlateVoteTracker:
    """Takip edilen her araç ID'si için çoklu kare plaka okumalarını biriktirir ve oylar."""

    def __init__(self):
        # track_id -> List[Tuple[plate_text, conf, plate_type]]
        self.history: Dict[int, List[Tuple[str, float, str]]] = {}
        # track_id -> (final_plate, final_conf, final_type)
        self.finalized: Dict[int, Tuple[str, float, str]] = {}

    def add_prediction(self, track_id: int, plate_text: str, conf: float, p_type: str):
        # Tek harfli veya çok kısa gürültüleri engelle
        clean_text = clean_plate_text(plate_text)
        if not clean_text or len(clean_text) < 3 or conf < 0.15:
            return

        if track_id not in self.history:
            self.history[track_id] = []
        self.history[track_id].append((plate_text, conf, p_type))

    def get_best_plate(self, track_id: int) -> Tuple[str, float, str]:
        """Araç için biriken tahminlerden en tutarlı ve yüksek skorlu olanı seçer."""
        if track_id in self.finalized:
            return self.finalized[track_id]

        records = self.history.get(track_id, [])
        if not records:
            return "PLAKA YOK", 0.0, "UNKNOWN"

        # 1. Metin sıklığı ve kümülatif skor hesabı
        scores: Dict[str, float] = {}
        type_map: Dict[str, str] = {}
        for text, conf, p_type in records:
            # Geçerli TR veya AB formatında olanlara yüksek ağırlık ver
            weight = 2.5 if p_type in ["TR", "EU_INTL"] else 0.5
            # Karakter sayısı arttıkça (plaka standartlarına yaklaştıkça) ekstra güven
            length_bonus = 1.0 + (min(len(clean_plate_text(text)), 8) / 10.0)
            score = conf * weight * length_bonus

            scores[text] = scores.get(text, 0.0) + score
            type_map[text] = p_type

        # En yüksek toplam puana sahip plaka
        best_text = max(scores, key=scores.get)
        # Ortalama güven skoru
        matching_confs = [c for t, c, _ in records if t == best_text]
        avg_conf = sum(matching_confs) / len(matching_confs)

        return best_text, avg_conf, type_map.get(best_text, "UNKNOWN")

    def finalize(self, track_id: int) -> Tuple[str, float, str]:
        """Aracın plakasını kesinleştirir (Sayım anında çağrılır)."""
        best = self.get_best_plate(track_id)
        self.finalized[track_id] = best
        return best

