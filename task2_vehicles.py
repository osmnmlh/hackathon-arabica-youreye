"""Görev 2: Araç Tespiti, Sayım, Araç Türü Sınıflandırma, Plaka Tespiti ve OCR.

Arabica x Youreye Hackathon 2026.
Donanım Optimizasyonu: NVIDIA RTX 4050 CUDA / FP16, Temporal Class Voting, High-Res Plate OCR.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import supervision as sv
from ultralytics import YOLO

import plate_utils


# ---------------------------------------------------------------------------
# Varsayılan Yapılandırma ve Sabitler
# ---------------------------------------------------------------------------
COCO_VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Genişletilmiş Sınıf Sözlüğü (Bonus Minibüs/Ticari ayrımı dahil)
CLASS_NAMES_EN = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
    8: "Van/Commercial"
}

# 1920x1080 Referans Tek, Temiz ve Kararlı Sayım Çizgisi
DEFAULT_LINE_START = (100, 820)
DEFAULT_LINE_END = (1750, 820)


# ---------------------------------------------------------------------------
# Minibüs / Ticari Araç Heuristic ve Sınıf Belirleme
# ---------------------------------------------------------------------------
def classify_vehicle_type(class_id: int, bbox_w: int, bbox_h: int, history_classes: List[int]) -> int:
    """Ford Transit / Minibüs / Kamyonet araçlarını COCO sınıflarından ayırır.
    
    COCO modelleri van/minibüs tipi araçları genellikle 'Otobüs' (5), 'Kamyon' (7)
    veya büyük 'Car' (2) olarak etiketler.
    Kural:
    - Sınıf 5 (Otobüs) gelmişse ancak piksel alanı devasa bir otobüs boyutunda (< 120.000 px)
      değilse veya en-boy oranı van profiline uyuyorsa Minibüs/Ticari (8) yapılır.
    - Sınıf geçmişinde hem otomobil hem kamyon/otobüs geçişi olan orta boy araçlar da van/ticari kabul edilir.
    """
    area = bbox_w * bbox_h
    aspect_ratio = bbox_w / max(bbox_h, 1)

    # 1. Otobüs etiketi almış ama otobüs kadar devasa olmayan araçlar
    if class_id == 5:
        if area < 125000:
            return 8  # Minibüs / Ticari
        return 5      # Gerçek Otobüs

    # 2. Kamyonet / Van tipi araçlar (Kamyon etiketi almış ama küçük/orta boyutlu kasa)
    if class_id == 7:
        if area < 100000 and 0.4 < aspect_ratio < 1.3:
            return 8  # Minibüs / Kamyonet
        return 7      # Gerçek Kamyon / Tır

    # 3. Geçmişte sınıf kararsızlığı yaşayan orta hacimli araçlar
    if len(history_classes) >= 5:
        unique_classes = set(history_classes)
        if 2 in unique_classes and (5 in unique_classes or 7 in unique_classes):
            if 40000 < area < 120000:
                return 8

    return class_id


def resolve_temporal_class_and_conf(
    history: List[Tuple[int, float]],
    bbox_w: int,
    bbox_h: int
) -> Tuple[int, str, float]:
    """Her track_id için ağırlıklı çoğunluk oylaması (Majority Voting) ile kararlı sınıfı belirler."""
    if not history:
        return 2, "Car", 0.0

    raw_classes = [c for c, _ in history]
    # Her tahmine van/minibüs heuristic'i uygula
    adjusted_history = [
        (classify_vehicle_type(c, bbox_w, bbox_h, raw_classes), conf)
        for c, conf in history
    ]

    # Sınıf bazında toplam güven ağırlıklarını topla
    class_weights: Dict[int, float] = {}
    class_confs: Dict[int, List[float]] = {}
    for cid, conf in adjusted_history:
        class_weights[cid] = class_weights.get(cid, 0.0) + conf
        class_confs.setdefault(cid, []).append(conf)

    # En yüksek ağırlıklı sınıfı seç
    best_cid = max(class_weights, key=class_weights.get)
    best_conf = float(np.mean(class_confs[best_cid])) if class_confs[best_cid] else 0.0
    v_name = CLASS_NAMES_EN.get(best_cid, "Car")

    return best_cid, v_name, best_conf


# ---------------------------------------------------------------------------
# UI ve Bilgi Paneli Çizim Yardımcıları (Optimize Edilmiş ROI Harmanlama)
# ---------------------------------------------------------------------------
def draw_dashboard(
    frame: np.ndarray,
    in_count: int,
    out_count: int,
    active_count: int,
    class_counts: Dict[str, int],
    fps: float,
    recent_passes: List[Dict]
) -> np.ndarray:
    """Ekranın üst sol ve sağ kısımlarına estetik bilgi paneli çizer."""
    h, w = frame.shape[:2]

    # 1. Sol Üst: Sayım ve İstatistik Paneli
    panel_x1, panel_y1, panel_w, panel_h = 20, 20, 380, 250
    roi_left = frame[panel_y1:panel_y1 + panel_h, panel_x1:panel_x1 + panel_w]
    overlay_left = roi_left.copy()
    overlay_left[:] = (20, 20, 25)
    cv2.addWeighted(overlay_left, 0.75, roi_left, 0.25, 0, roi_left)

    # Panel Başlığı
    cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x1 + panel_w, panel_y1 + 40), (45, 45, 55), -1)
    cv2.putText(frame, "TRAFFIC CONTROL PANEL", (panel_x1 + 15, panel_y1 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2, cv2.LINE_AA)

    # Sayım Rakamları
    cv2.putText(frame, f"VEHICLES IN: {in_count}", (panel_x1 + 15, panel_y1 + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 128), 2, cv2.LINE_AA)
    cv2.putText(frame, f"VEHICLES OUT: {out_count}", (panel_x1 + 15, panel_y1 + 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"ACTIVE TRACKS:   {active_count}", (panel_x1 + 15, panel_y1 + 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 100), 1, cv2.LINE_AA)
    cv2.putText(frame, f"PROCESSING SPEED:      {fps:.1f} FPS", (panel_x1 + 15, panel_y1 + 165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    # Araç Tipi Dağılımı Özeti
    type_str = f"Car:{class_counts.get('Car', 0)} | Van:{class_counts.get('Van/Commercial', 0)} | Truck:{class_counts.get('Truck', 0)}"
    cv2.putText(frame, type_str, (panel_x1 + 15, panel_y1 + 205),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    sub_str = f"Bus:{class_counts.get('Bus', 0)} | Moto:{class_counts.get('Motorcycle', 0)}"
    cv2.putText(frame, sub_str, (panel_x1 + 15, panel_y1 + 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1, cv2.LINE_AA)

    # 2. Sağ Üst: Son Okunan Plakalar Akışı
    if recent_passes:
        r_w, r_h = 350, 220
        rx1, ry1 = w - r_w - 20, 20
        roi_right = frame[ry1:ry1 + r_h, rx1:rx1 + r_w]
        overlay_right = roi_right.copy()
        overlay_right[:] = (20, 20, 25)
        cv2.addWeighted(overlay_right, 0.75, roi_right, 0.25, 0, roi_right)

        cv2.rectangle(frame, (rx1, ry1), (rx1 + r_w, ry1 + 35), (45, 45, 55), -1)
        cv2.putText(frame, "RECENT PASSES", (rx1 + 15, ry1 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2, cv2.LINE_AA)

        y_offset = ry1 + 60
        for ev in recent_passes:
            tid = ev["arac_id"]
            mm_ss = ev["gecis_zamani"]
            p_text = ev.get("plaka", "-")
            p_conf = ev.get("plaka_guven", 0.0)
            
            if p_text == "-" or p_text == "PLAKA YOK":
                line_str = f"#{tid} {mm_ss} - %0"
                color = (180, 180, 180)
            else:
                line_str = f"#{tid} {mm_ss} {p_text} %{int(p_conf*100)}"
                color = (0, 255, 128) if ev.get("plaka_turu") == "PL" else (0, 200, 255)
            
            cv2.putText(frame, line_str, (rx1 + 15, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
            y_offset += 25

    return frame


# ---------------------------------------------------------------------------
# Ana İşleme Hattı (Pipeline)
# ---------------------------------------------------------------------------
def process_video(
    video_path: str,
    output_video_path: str,
    output_csv_path: str,
    crops_dir: str,
    vehicle_model_path: str = "models/yolov8s.pt",
    plate_model_path: str = "models/license_plate_detector.pt",
    custom_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    flip: bool = False,
    conf_threshold: float = 0.35,
    save_crops: bool = True,
    max_frames: Optional[int] = None,
    show_preview: bool = False
):
    """Araç tespit, sayım, plaka kırpma ve OCR hattını uçtan uca çalıştırır."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_half = (device == "cuda")
    print(f"[*] İşlemci Aygıtı: {device.upper()} (FP16/Half: {use_half})")
    if device == "cuda":
        print(f"[*] GPU: {torch.cuda.get_device_name(0)}")

    # Çıktı dizinlerini hazırla
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    if save_crops:
        os.makedirs(crops_dir, exist_ok=True)

    # 1. Modelleri Yükle
    print(f"[*] Araç tespit modeli yükleniyor: {vehicle_model_path}...")
    v_model = YOLO(vehicle_model_path)
    print(f"[*] Plaka tespit modeli yükleniyor: {plate_model_path}...")
    p_model = YOLO(plate_model_path)
    print("[*] EasyOCR okuyucu başlatılıyor (GPU)...")
    ocr_reader = plate_utils.LicensePlateOCR(gpu=(device == "cuda"))
    vote_tracker = plate_utils.PlateVoteTracker()

    # 2. Video Aç ve Bilgileri Al
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video dosyası açılamadı: {video_path}")

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Performans ve kararlılık için işlem çözünürlüğü: 1920x1080
    proc_width, proc_height = 1920, 1080
    scale_x = proc_width / orig_width
    scale_y = proc_height / orig_height

    print(f"[*] Video Boyutu: {orig_width}x{orig_height} -> İşlem Boyutu: {proc_width}x{proc_height} @ {input_fps:.1f} FPS")
    print(f"[*] Toplam Kare: {total_frames}")

    # Video Yazıcı
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_video_path, fourcc, input_fps, (proc_width, proc_height))

    # 3. Sayım Çizgisi Tanımlama
    if custom_line:
        p1, p2 = custom_line
        print(f"[*] Özel Çizgi Belirlendi: {p1} -> {p2}")
    else:
        p1, p2 = DEFAULT_LINE_START, DEFAULT_LINE_END
        print(f"[*] Varsayılan Tüm Yol Koridor Çizgisi: {p1} -> {p2}")

    if flip:
        p1, p2 = p2, p1
        print(f"[*] Çizgi Yönü Çevrildi (--flip): {p1} -> {p2}")

    try:
        line_zone = sv.LineZone(
            start=sv.Point(x=p1[0], y=p1[1]),
            end=sv.Point(x=p2[0], y=p2[1]),
            triggering_anchors=[sv.Position.BOTTOM_CENTER]
        )
    except TypeError as e:
        print(f"[!] sv.LineZone triggering_anchors desteklenmedi ({e}), parametresiz oluşturuluyor.")
        line_zone = sv.LineZone(
            start=sv.Point(x=p1[0], y=p1[1]),
            end=sv.Point(x=p2[0], y=p2[1])
        )

    line_annotator = sv.LineZoneAnnotator(
        thickness=3,
        text_thickness=2,
        text_scale=0.7,
        color=sv.Color(r=0, g=255, b=128)
    )

    # Takipçi (ByteTrack)
    tracker = sv.ByteTrack(
        track_activation_threshold=conf_threshold,
        lost_track_buffer=45,
        minimum_matching_threshold=0.8,
        frame_rate=int(input_fps)
    )

    # Anotatörler
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=4)
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=30)

    # Kayıt ve Durum Veri Yapıları
    counting_events: List[Dict] = []
    vehicle_crops_cache: Dict[int, np.ndarray] = {}  # track_id -> RAM kopya araç görseli
    
    class_counter: Dict[str, int] = {
        "Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0, "Van/Commercial": 0
    }

    # Sınıf Atlama ve Titremeyi Önleyen Zamansal Sınıf Geçmişi (Temporal History)
    track_class_history: Dict[int, List[Tuple[int, float]]] = {}  # track_id -> [(class_id, conf), ...]

    # OCR Bütçesi ve Kapıları Takip Verileri
    track_ocr_counts: Dict[int, int] = {}             # track_id -> yapılan OCR sayısı (max 8)
    track_best_q: Dict[int, float] = {}               # track_id -> en iyi kırpıntı kalitesi
    track_completed: set = set()                      # 2 adet aynı 'PL' ve conf >= 0.6 alanlar
    track_pl_highconf: Dict[int, List[str]] = {}      # track_id -> List[plate_text]
    track_pass_count: Dict[int, int] = {}             # track_id -> gecis sayisi
    track_plate_det_conf: Dict[int, float] = {}       # track_id -> plaka detektor guveni
    last_plate_boxes: Dict[int, Tuple[Tuple[int, int], Tuple[int, int], float]] = {} # track_id -> (pt1, pt2, conf)

    # Performans Ölçüm Sayaçları (ms toplamları)
    perf_timers = {
        "read_resize": 0.0,
        "yolo_track": 0.0,
        "plate_ocr": 0.0,
        "drawing": 0.0,
        "write": 0.0
    }

    frame_idx = 0
    start_time = time.time()
    last_fps_calc = time.time()
    fps_display = input_fps

    print("[*] Video işleme başladı...")

    try:
        while True:
            # Aşama 1: [okuma+resize]
            t0 = time.perf_counter()
            ret, frame_orig = cap.read()
            if not ret:
                break
    
            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break
    
            if (orig_width, orig_height) != (proc_width, proc_height):
                frame = cv2.resize(frame_orig, (proc_width, proc_height), interpolation=cv2.INTER_AREA)
            else:
                frame = frame_orig.copy()
            t1 = time.perf_counter()
            perf_timers["read_resize"] += (t1 - t0) * 1000.0
    
            # Aşama 2: [araç YOLO+tracker]
            v_results = v_model(
                frame,
                classes=COCO_VEHICLE_CLASSES,
                conf=0.15,
                verbose=False,
                device=device,
                half=use_half
            )[0]
    
            detections = sv.Detections.from_ultralytics(v_results)
            detections = tracker.update_with_detections(detections)
            t2 = time.perf_counter()
            perf_timers["yolo_track"] += (t2 - t1) * 1000.0
    
            # Sınıf geçmişini güncelle
            ids = detections.tracker_id if detections.tracker_id is not None else []
            for i, (xyxy, track_id) in enumerate(zip(detections.xyxy, ids)):
                track_id = int(track_id)
                cls_id = int(detections.class_id[i]) if detections.class_id is not None else 2
                det_conf = float(detections.confidence[i]) if detections.confidence is not None else 0.5
                track_class_history.setdefault(track_id, []).append((cls_id, det_conf))
    
            # Aşama 3: [plaka modeli+OCR]
            # Y Sadece [550, 950] piksel aralığında ve her 2 karede bir çalıştır
            if frame_idx % 2 == 0 and len(ids) > 0:
                candidate_tracks = []
                candidate_vcrops = []
                candidate_meta = []
    
                for i, (xyxy, track_id) in enumerate(zip(detections.xyxy, ids)):
                    track_id = int(track_id)
                    if track_id in track_completed or track_ocr_counts.get(track_id, 0) >= 8:
                        continue
    
                    x1, y1, x2, y2 = map(int, xyxy)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(proc_width, x2), min(proc_height, y2)
    
                    # Plaka modelini sadece netlik aralığı olan Y in [550, 950] içinde tetikle
                    if 550 <= y2 <= 950 and (y2 - y1) > 50 and (x2 - x1) > 60:
                        v_crop = frame[y1:y2, x1:x2]
                        if v_crop.size > 0:
                            candidate_tracks.append(track_id)
                            candidate_vcrops.append(v_crop)
                            candidate_meta.append((x1, y1, x2, y2))
    
                # Plaka modelini tek batch olarak çağır (FP16 / Half)
                if len(candidate_vcrops) > 0:
                    p_results = p_model(
                        candidate_vcrops,
                        conf=0.15,
                        verbose=False,
                        device=device,
                        half=use_half
                    )
    
                    ocr_candidates = []
                    for tr_id, v_crop, (x1, y1, x2, y2), p_res in zip(candidate_tracks, candidate_vcrops, candidate_meta, p_results):
                        if len(p_res.boxes) > 0:
                            best_pbox = max(p_res.boxes, key=lambda b: float(b.conf[0]))
                            px1, py1, px2, py2 = best_pbox.xyxy[0].cpu().numpy().astype(int)
                            px1, py1 = max(0, px1), max(0, py1)
                            px2, py2 = min(v_crop.shape[1], px2), min(v_crop.shape[0], py2)
    
                            # Orijinal 4K koordinatları
                            orig_px1 = max(0, int((x1 + px1) / scale_x))
                            orig_py1 = max(0, int((y1 + py1) / scale_y))
                            orig_px2 = min(orig_width, int((x1 + px2) / scale_x))
                            orig_py2 = min(orig_height, int((y1 + py2) / scale_y))
    
                            crop_w = orig_px2 - orig_px1
                            crop_h = orig_py2 - orig_py1
    
                            # Kırpıntı alanı kontrolleri
                            if crop_w >= 80 and crop_h >= 20:
                                plate_crop_highres = frame_orig[orig_py1:orig_py2, orig_px1:orig_px2]
                                if plate_crop_highres.size > 0:
                                    ocr_candidates.append((crop_w * crop_h, tr_id, plate_crop_highres))
    
                    # Kare başına en fazla 2 OCR çağrısı
                    ocr_candidates.sort(key=lambda x: x[0], reverse=True)
                    for _, tr_id, plate_crop_hr in ocr_candidates[:2]:
                        p_text, p_conf, p_type = ocr_reader.read_plate(plate_crop_hr)
                        track_ocr_counts[tr_id] = track_ocr_counts.get(tr_id, 0) + 1
    
                        if p_text and p_type != "UNKNOWN":
                            vote_tracker.add_prediction(tr_id, p_text, p_conf, p_type, plate_crop=plate_crop_hr)
    
                            if p_type == "PL" and p_conf >= 0.60:
                                high_confs = track_pl_highconf.setdefault(tr_id, [])
                                high_confs.append(p_text)
                                if high_confs.count(p_text) >= 2:
                                    track_completed.add(tr_id)
    
            t3 = time.perf_counter()
            perf_timers["plate_ocr"] += (t3 - t2) * 1000.0
    
            # Çizgi Sayımı ve Tetiklenme (LineZone)
            crossed_in, crossed_out = line_zone.trigger(detections)
    
            for i, (is_in, is_out) in enumerate(zip(crossed_in, crossed_out)):
                if is_in or is_out:
                    track_id = int(ids[i])
                    if True:
                        # counted_vehicle_ids.add(track_id)
    
                        xyxy = detections.xyxy[i]
                        x1, y1 = max(0, int(xyxy[0])), max(0, int(xyxy[1]))
                        x2, y2 = min(proc_width, int(xyxy[2])), min(proc_height, int(xyxy[3]))
                        bw, bh = x2 - x1, y2 - y1
    
                        # Zamansal sınıf oylaması ile kararlı araç türünü ve güvenini belirle
                        hist = track_class_history.get(track_id, [])
                        best_cid, v_type_name, v_conf = resolve_temporal_class_and_conf(hist, bw, bh)
                        class_counter[v_type_name] = class_counter.get(v_type_name, 0) + 1
    
                        direction = "In" if is_in else "Out"
                        timestamp_sec = frame_idx / input_fps
                        mm = int(timestamp_sec // 60)
                        ss = timestamp_sec % 60
                        gecis_zamani = f"{mm:02d}:{ss:04.1f}"
                        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
                        track_pass_count[track_id] = track_pass_count.get(track_id, 0) + 1
                        pass_c = track_pass_count[track_id]
                        tekrar_gecis = True if pass_c > 1 else False
    
                        # Araç görselini sayım anında RAM'e kopyalayarak sakla
                        v_crop = frame[y1:y2, x1:x2].copy()
                        vehicle_crops_cache[track_id] = v_crop
    
                        # Sayım olayını kaydet
                        counting_events.append({
                            "arac_id": track_id,
                            "arac_tipi": v_type_name,
                            "arac_guven": round(v_conf, 2),
                            "yon": direction,
                            "video_saniyesi": round(timestamp_sec, 2),
                            "gecis_zamani": gecis_zamani,
                            "kare_no": frame_idx,
                            "zaman": time_str,
                            "tekrar_gecis": tekrar_gecis,
                            "gecis_sayisi": pass_c,
                            "plaka_tespit_guven": round(track_plate_det_conf.get(track_id, 0.0), 2)
                        })
    
                        print(f"[*] [SAYIM #{len(counting_events)}] ID:{track_id} | Tür:{v_type_name} (%{int(v_conf*100)}) | Yön:{direction}")
    
            # Aşama 4: [çizim]
            labels = []
            for i, track_id in enumerate(ids):
                track_id = int(track_id)
                xyxy = detections.xyxy[i]
                bw = int(xyxy[2] - xyxy[0])
                bh = int(xyxy[3] - xyxy[1])
    
                hist = track_class_history.get(track_id, [])
                _, v_name, v_conf = resolve_temporal_class_and_conf(hist, bw, bh)
    
                cur_plate, p_conf, p_type = vote_tracker.get_best_plate(track_id)
                det_conf = track_plate_det_conf.get(track_id, 0.0)
    
                if cur_plate and cur_plate != "PLAKA YOK":
                    plate_label = cur_plate if p_type != "RAW" else f"{cur_plate}?"
                    labels.append(f"#{track_id} {v_name} | {plate_label} (P:%{int(p_conf*100)} D:%{int(det_conf*100)})")
                else:
                    labels.append(f"#{track_id} {v_name} (%{int(v_conf*100)})")
                    
                # Draw last plate box
                if track_id in last_plate_boxes:
                    p1_proc, p2_proc, pd_conf = last_plate_boxes[track_id]
                    cv2.rectangle(frame, p1_proc, p2_proc, (0, 200, 255), 2)
    
            frame = trace_annotator.annotate(scene=frame, detections=detections)
            frame = box_annotator.annotate(scene=frame, detections=detections)
            frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
            line_annotator.annotate(frame=frame, line_counter=line_zone)
    
            # FPS Güncellemesi
            if frame_idx % 15 == 0:
                now = time.time()
                fps_display = 15.0 / max(now - last_fps_calc, 0.001)
                last_fps_calc = now
    
            in_count = sum(1 for ev in counting_events if ev["yon"] == "In")
            out_count = sum(1 for ev in counting_events if ev["yon"] == "Out")

            # Son geçişleri dinamik güncelle
            recent_passes_display = []
            for ev in counting_events[-6:]:
                tid = ev["arac_id"]
                p_text, p_conf, p_type = vote_tracker.get_best_plate(tid)
                disp_ev = ev.copy()
                if p_text and p_text != "PLAKA YOK":
                    disp_ev["plaka"] = p_text
                    disp_ev["plaka_guven"] = p_conf
                    disp_ev["plaka_turu"] = p_type
                else:
                    disp_ev["plaka"] = "-"
                    disp_ev["plaka_guven"] = 0.0
                    disp_ev["plaka_turu"] = "UNKNOWN"
                recent_passes_display.append(disp_ev)

            frame = draw_dashboard(
                frame=frame,
                in_count=in_count,
                out_count=out_count,
                active_count=len(detections),
                class_counts=class_counter,
                fps=fps_display,
                recent_passes=recent_passes_display
            )
            t4 = time.perf_counter()
            perf_timers["drawing"] += (t4 - t3) * 1000.0
    
            # Aşama 5: [video yazma]
            out_writer.write(frame)
            t5 = time.perf_counter()
            perf_timers["write"] += (t5 - t4) * 1000.0
    
            if show_preview:
                cv2.imshow("Gorev 2 - Arac Takip & Plaka Tanima", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
            # Her 60 karede aşama başına ortalama ms ve FPS raporu
            if frame_idx % 60 == 0:
                n = 60
                ms_read = perf_timers["read_resize"] / n
                ms_yolo = perf_timers["yolo_track"] / n
                ms_ocr = perf_timers["plate_ocr"] / n
                ms_draw = perf_timers["drawing"] / n
                ms_write = perf_timers["write"] / n
                elapsed = time.time() - start_time
                cur_fps = frame_idx / max(elapsed, 0.001)
    
                print(f"[{frame_idx}/{total_frames}] Aşama Ort (ms): "
                      f"[okuma+resize]: {ms_read:.1f}ms | [araç YOLO+tracker]: {ms_yolo:.1f}ms | "
                      f"[plaka+OCR]: {ms_ocr:.1f}ms | [çizim]: {ms_draw:.1f}ms | [video yazma]: {ms_write:.1f}ms || "
                      f"Ort FPS: {cur_fps:.1f} | Toplam Sayım: {len(counting_events)}")
    
                for k in perf_timers:
                    perf_timers[k] = 0.0
    
    finally:
        # Temizlik
        cap.release()
        out_writer.release()
        if show_preview:
            cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    avg_fps = frame_idx / max(total_time, 0.001)
    print(f"\n[+] Video işleme tamamlandı!")
    print(f"[+] Toplam Süre: {total_time:.2f} sn | Ortalama FPS: {avg_fps:.1f}")
    print(f"[+] Çıktı Videosu: {output_video_path}")

    # -------------------------------------------------------------------
    # 5. Video Sonu Kesinleştirme ve Raporlama (CSV + JSON + Standart Görseller)
    # -------------------------------------------------------------------
    final_records: List[Dict] = []
    for ev in counting_events:
        tid = ev["arac_id"]
        final_plate, final_conf, final_type = vote_tracker.get_best_plate(tid)

        plate_filename = ""
        veh_filename = ""

        if save_crops:
            # Standart dosya isimlendirmesi: plate_{id}.jpg ve veh_{id}.jpg
            best_plate_img = vote_tracker.get_best_plate_crop(tid)
            if best_plate_img is not None and best_plate_img.size > 0:
                plate_filename = f"plate_{tid}.jpg"
                cv2.imwrite(os.path.join(crops_dir, plate_filename), best_plate_img)

            if tid in vehicle_crops_cache:
                veh_filename = f"veh_{tid}.jpg"
                cv2.imwrite(os.path.join(crops_dir, veh_filename), vehicle_crops_cache[tid])

        rec = {
            "arac_id": tid,
            "arac_tipi": ev["arac_tipi"],
            "arac_guven": ev.get("arac_guven", 0.0),
            "plaka": final_plate,
            "plaka_turu": final_type,
            "plaka_guven": round(final_conf, 2),
            "plaka_tespit_guven": round(track_plate_det_conf.get(tid, 0.0), 2),
            "yon": ev["yon"],
            "gecis_zamani": ev.get("gecis_zamani", ""),
            "tekrar_gecis": ev.get("tekrar_gecis", False),
            "gecis_sayisi": ev.get("gecis_sayisi", 1),
            "video_saniyesi": ev["video_saniyesi"],
            "kare_no": ev["kare_no"],
            "plaka_gorsel": plate_filename,
            "arac_gorsel": veh_filename
        }
        final_records.append(rec)

    # CSV Kaydet
    df = pd.DataFrame(final_records)
    if not df.empty:
        df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
        print(f"[+] CSV Raporu Kaydedildi: {output_csv_path} ({len(df)} kayıt)")
    else:
        cols = ["arac_id", "arac_tipi", "arac_guven", "plaka", "plaka_turu", "plaka_guven", "plaka_tespit_guven", "yon", "gecis_zamani", "tekrar_gecis", "gecis_sayisi", "video_saniyesi", "kare_no", "plaka_gorsel", "arac_gorsel"]
        pd.DataFrame(columns=cols).to_csv(output_csv_path, index=False, encoding="utf-8-sig")
        print(f"[!] Belirtilen çizgiden araç geçmedi. Boş CSV şablonu oluşturuldu: {output_csv_path}")

    # JSON Kaydet
    json_path = os.path.splitext(output_csv_path)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(final_records, jf, ensure_ascii=False, indent=2)
    print(f"[+] JSON Raporu Kaydedildi: {json_path}")

    return {
        "total_in": sum(1 for ev in counting_events if ev["yon"] == "In"),
        "total_out": sum(1 for ev in counting_events if ev["yon"] == "Out"),
        "class_counts": class_counter,
        "records_count": len(final_records),
        "avg_fps": round(avg_fps, 1)
    }


# ---------------------------------------------------------------------------
# CLI Arayüzü (Argparse)
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Görev 2: Araç Tespiti, Sayım, Araç Türü ve Plaka OCR")
    parser.add_argument(
        "--video",
        type=str,
        default="data/traffic_1080p.mp4",
        help="Girdi video dosyası yolu"
    )
    parser.add_argument(
        "--output_video",
        type=str,
        default="outputs/task2_vehicles_output.mp4",
        help="Çıktı video dosyası yolu"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="outputs/vehicle_log.csv",
        help="Çıktı CSV log dosyası yolu"
    )
    parser.add_argument(
        "--crops_dir",
        type=str,
        default="outputs/crops",
        help="Kırpılan plaka ve araç görsellerinin kaydedileceği dizin"
    )
    parser.add_argument(
        "--vehicle_model",
        type=str,
        default="models/yolov8s.pt",
        help="Araç tespit YOLO ağırlık yolu"
    )
    parser.add_argument(
        "--plate_model",
        type=str,
        default="models/license_plate_detector.pt",
        help="Plaka tespit YOLO ağırlık yolu"
    )
    parser.add_argument(
        "--line",
        type=str,
        default=None,
        help="Özel çizgi koordinatları 'x1,y1,x2,y2' (Örn: '100,820,1750,820')"
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        help="Sayım çizgisi yönünü tersine çevir (Giriş/Çıkış yönünü değiştirir)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Araç tespit güven eşiği"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Test için işlenecek maksimum kare sayısı"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="İşleme sırasında OpenCV penceresinde önizleme göster"
    )

    args = parser.parse_args()

    custom_line = None
    if args.line:
        try:
            parts = [int(p.strip()) for p in args.line.split(",")]
            if len(parts) == 4:
                custom_line = ((parts[0], parts[1]), (parts[2], parts[3]))
            else:
                print(f"[!] Hatalı --line formatı ({args.line}), varsayılan çizgi kullanılacak.")
        except Exception as e:
            print(f"[!] --line parse hatası ({e}), varsayılan çizgi kullanılacak.")

    process_video(
        video_path=args.video,
        output_video_path=args.output_video,
        output_csv_path=args.output_csv,
        crops_dir=args.crops_dir,
        vehicle_model_path=args.vehicle_model,
        plate_model_path=args.plate_model,
        custom_line=custom_line,
        flip=args.flip,
        conf_threshold=args.conf,
        max_frames=args.max_frames,
        show_preview=args.preview
    )


if __name__ == "__main__":
    main()
