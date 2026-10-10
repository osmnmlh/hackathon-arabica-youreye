"""Görev 2: Araç Tespiti, Sayım, Araç Türü Sınıflandırma, Plaka Tespiti ve OCR

Arabica x Youreye Hackathon 2026.
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
CLASS_NAMES_TR = {
    2: "Otomobil",
    3: "Motosiklet",
    5: "Otobüs",
    7: "Kamyon"
}

# 1920x1080 Referans Koordinatları
DEFAULT_LINE_START_A = (800, 780)
DEFAULT_LINE_END_A = (1120, 780)

DEFAULT_LINE_START_B = (150, 800)
DEFAULT_LINE_END_B = (1750, 800)


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
    recent_plates: List[Tuple[str, str, float]]
) -> np.ndarray:
    """Ekranın üst sol ve sağ kısımlarına estetik bilgi paneli çizer (Sadece ROI kopyalayarak optimize edildi)."""
    h, w = frame.shape[:2]

    # 1. Sol Üst: Sayım ve İstatistik Paneli
    panel_x1, panel_y1, panel_w, panel_h = 20, 20, 360, 240
    roi_left = frame[panel_y1:panel_y1 + panel_h, panel_x1:panel_x1 + panel_w]
    overlay_left = roi_left.copy()
    overlay_left[:] = (20, 20, 25)
    cv2.addWeighted(overlay_left, 0.75, roi_left, 0.25, 0, roi_left)

    # Panel Başlığı
    cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x1 + panel_w, panel_y1 + 40), (45, 45, 55), -1)
    cv2.putText(frame, "TRAFIK DENETIM PANELI", (panel_x1 + 15, panel_y1 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2, cv2.LINE_AA)

    # Sayım Rakamları
    cv2.putText(frame, f"GIRIS YAPAN ARAC: {in_count}", (panel_x1 + 15, panel_y1 + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 128), 2, cv2.LINE_AA)
    cv2.putText(frame, f"CIKIS YAPAN ARAC: {out_count}", (panel_x1 + 15, panel_y1 + 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"AKTIF TAKIPTE:   {active_count}", (panel_x1 + 15, panel_y1 + 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 100), 1, cv2.LINE_AA)
    cv2.putText(frame, f"ISLEM HIZI:      {fps:.1f} FPS", (panel_x1 + 15, panel_y1 + 165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    # Araç Tipi Dağılımı Özeti
    type_str = f"Oto:{class_counts.get('Otomobil', 0)} | Kamyon:{class_counts.get('Kamyon', 0)} | OtoB:{class_counts.get('Otobüs', 0)}"
    cv2.putText(frame, type_str, (panel_x1 + 15, panel_y1 + 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    # 2. Sağ Üst: Son Okunan Plakalar Akışı
    if recent_plates:
        r_w, r_h = 320, 140
        rx1, ry1 = w - r_w - 20, 20
        roi_right = frame[ry1:ry1 + r_h, rx1:rx1 + r_w]
        overlay_right = roi_right.copy()
        overlay_right[:] = (20, 20, 25)
        cv2.addWeighted(overlay_right, 0.75, roi_right, 0.25, 0, roi_right)

        cv2.rectangle(frame, (rx1, ry1), (rx1 + r_w, ry1 + 35), (45, 45, 55), -1)
        cv2.putText(frame, "SON OKUNAN PLAKALAR", (rx1 + 15, ry1 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2, cv2.LINE_AA)

        y_offset = ry1 + 60
        for p_text, p_type, conf in recent_plates[-3:]:
            color = (0, 255, 128) if p_type == "PL" else (0, 200, 255)
            line_str = f"[{p_type}] {p_text} (%{conf*100:.0f})"
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
    scenario: str = "A",
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
    print(f"[*] İşlemci Aygıtı: {device.upper()} (Half precision: {use_half})")
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

    # Performans ve stabilite için işlem çözünürlüğü: 1920x1080
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
    elif scenario.upper() == "A":
        p1, p2 = DEFAULT_LINE_START_A, DEFAULT_LINE_END_A
        print(f"[*] Senaryo A Seçildi (Orta Şehir Giriş Koridoru: {p1} -> {p2})")
    else:
        p1, p2 = DEFAULT_LINE_START_B, DEFAULT_LINE_END_B
        print(f"[*] Senaryo B Seçildi (Tüm Koridor: {p1} -> {p2})")

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

    # Kayıt Veri Yapıları
    counting_events: List[Dict] = []
    vehicle_crops_cache: Dict[int, np.ndarray] = {}  # track_id -> kopya araç görseli
    counted_vehicle_ids = set()
    recent_plates_display: List[Tuple[str, str, float]] = []
    class_counter: Dict[str, int] = {"Otomobil": 0, "Motosiklet": 0, "Otobüs": 0, "Kamyon": 0}
    track_classes: Dict[int, int] = {}

    # OCR Bütçesi ve Kapıları Takip Verileri
    track_ocr_counts: Dict[int, int] = {}             # track_id -> toplam yapılan OCR sayısı (max 8)
    track_completed: set = set()                      # 2 adet aynı 'PL' ve conf >= 0.6 alanlar
    track_pl_highconf: Dict[int, List[str]] = {}      # track_id -> List[plate_text]

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
            conf=conf_threshold,
            verbose=False,
            device=device,
            half=use_half
        )[0]

        detections = sv.Detections.from_ultralytics(v_results)
        detections = tracker.update_with_detections(detections)
        t2 = time.perf_counter()
        perf_timers["yolo_track"] += (t2 - t1) * 1000.0

        # Aşama 3: [plaka modeli+OCR]
        ids = detections.tracker_id if detections.tracker_id is not None else []
        for i, (xyxy, track_id) in enumerate(zip(detections.xyxy, ids)):
            track_id = int(track_id)
            cls_id = int(detections.class_id[i]) if detections.class_id is not None else 2
            track_classes[track_id] = cls_id

        # Plaka modelini her 2 karede bir (frame_idx % 2 == 0) çalıştır
        if frame_idx % 2 == 0 and len(ids) > 0:
            candidate_tracks = []
            candidate_vcrops = []
            candidate_meta = []

            for i, (xyxy, track_id) in enumerate(zip(detections.xyxy, ids)):
                track_id = int(track_id)
                # Tamamlanmış veya bütçesi (8 OCR) dolmuş olanları atla
                if track_id in track_completed or track_ocr_counts.get(track_id, 0) >= 8:
                    continue

                x1, y1, x2, y2 = map(int, xyxy)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(proc_width, x2), min(proc_height, y2)

                # Sweet spot bölgesi kontrolü
                if y2 > 400 and (y2 - y1) > 50 and (x2 - x1) > 60:
                    v_crop = frame[y1:y2, x1:x2]
                    if v_crop.size > 0:
                        candidate_tracks.append(track_id)
                        candidate_vcrops.append(v_crop)
                        candidate_meta.append((x1, y1, x2, y2))

            # Plaka modelini tek batch olarak çağır
            if len(candidate_vcrops) > 0:
                p_results = p_model(
                    candidate_vcrops,
                    conf=0.15,
                    verbose=False,
                    device=device,
                    half=use_half
                )

                ocr_candidates = []  # List of (area, track_id, plate_crop_highres)
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

                        # Plaka kırpıntısı 4K'da genişlik < 80 px veya yükseklik < 20 px ise OCR yapma
                        if crop_w >= 80 and crop_h >= 20:
                            plate_crop_highres = frame_orig[orig_py1:orig_py2, orig_px1:orig_px2]
                            if plate_crop_highres.size > 0:
                                ocr_candidates.append((crop_w * crop_h, tr_id, plate_crop_highres))

                # Kare başına en fazla 2 OCR çağrısı (Kırpıntı alanı en büyük olanlara öncelik ver)
                ocr_candidates.sort(key=lambda x: x[0], reverse=True)
                for _, tr_id, plate_crop_hr in ocr_candidates[:2]:
                    p_text, p_conf, p_type = ocr_reader.read_plate(plate_crop_hr)
                    track_ocr_counts[tr_id] = track_ocr_counts.get(tr_id, 0) + 1

                    if p_text and p_type != "UNKNOWN":
                        vote_tracker.add_prediction(tr_id, p_text, p_conf, p_type, plate_crop=plate_crop_hr)

                        # Erken tamamlama kapısı kontrolü: 2 adet 'PL' tipi, aynı ve ham conf >= 0.6
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
                if track_id not in counted_vehicle_ids:
                    counted_vehicle_ids.add(track_id)

                    cls_id = track_classes.get(track_id, 2)
                    v_type_name = CLASS_NAMES_TR.get(cls_id, "Otomobil")
                    class_counter[v_type_name] = class_counter.get(v_type_name, 0) + 1

                    direction = "Giris" if is_in else "Cikis"
                    timestamp_sec = frame_idx / input_fps
                    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Araç görselini sayım anında RAM'e kopyalayarak sakla (Büyük kareyi tutma)
                    xyxy = detections.xyxy[i]
                    x1, y1 = max(0, int(xyxy[0])), max(0, int(xyxy[1]))
                    x2, y2 = min(proc_width, int(xyxy[2])), min(proc_height, int(xyxy[3]))
                    v_crop = frame[y1:y2, x1:x2].copy()
                    vehicle_crops_cache[track_id] = v_crop

                    # Olayı kaydet (Plaka video bitiminde vote_tracker'dan kesinleşecek)
                    counting_events.append({
                        "arac_id": track_id,
                        "arac_tipi": v_type_name,
                        "yon": direction,
                        "video_saniyesi": round(timestamp_sec, 2),
                        "kare_no": frame_idx,
                        "zaman": time_str
                    })

                    print(f"[*] [SAYIM #{len(counting_events)}] ID:{track_id} | Tür:{v_type_name} | Yön:{direction}")

        # Aşama 4: [çizim]
        labels = []
        for i, track_id in enumerate(ids):
            track_id = int(track_id)
            cls_id = track_classes.get(track_id, 2)
            v_name = CLASS_NAMES_TR.get(cls_id, "Otomobil")
            cur_plate, p_conf, p_type = vote_tracker.get_best_plate(track_id)

            if cur_plate and cur_plate != "PLAKA YOK":
                plate_label = cur_plate if p_type != "RAW" else f"{cur_plate}?"
                labels.append(f"#{track_id} {v_name} | {plate_label}")
            else:
                labels.append(f"#{track_id} {v_name}")

        frame = trace_annotator.annotate(scene=frame, detections=detections)
        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
        line_annotator.annotate(frame=frame, line_counter=line_zone)

        # FPS Güncellemesi
        if frame_idx % 15 == 0:
            now = time.time()
            fps_display = 15.0 / max(now - last_fps_calc, 0.001)
            last_fps_calc = now

        # Son okunan plaka listesini güncelle
        recent_plates_display = []
        for ev in counting_events[-3:]:
            tid = ev["arac_id"]
            p_text, p_conf, p_type = vote_tracker.get_best_plate(tid)
            if p_text and p_text != "PLAKA YOK":
                recent_plates_display.append((p_text, p_type, p_conf))

        frame = draw_dashboard(
            frame=frame,
            in_count=line_zone.in_count,
            out_count=line_zone.out_count,
            active_count=len(detections),
            class_counts=class_counter,
            fps=fps_display,
            recent_plates=recent_plates_display
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
                  f"Ort FPS: {cur_fps:.1f} | Toplam Sayım: {line_zone.in_count + line_zone.out_count}")

            # Sayaçları sıfırla
            for k in perf_timers:
                perf_timers[k] = 0.0

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
    # 5. Video Sonu Kesinleştirme ve Raporlama (CSV + JSON + Görseller)
    # -------------------------------------------------------------------
    final_records: List[Dict] = []
    for ev in counting_events:
        tid = ev["arac_id"]
        final_plate, final_conf, final_type = vote_tracker.get_best_plate(tid)

        plate_filename = ""
        veh_filename = ""

        if save_crops:
            # 1. En kaliteli plaka görselini kaydet
            best_plate_img = vote_tracker.get_best_plate_crop(tid)
            if best_plate_img is not None and best_plate_img.size > 0:
                clean_p_str = final_plate.replace(" ", "_").replace("?", "")
                plate_filename = f"plate_id{tid}_{clean_p_str}.jpg"
                cv2.imwrite(os.path.join(crops_dir, plate_filename), best_plate_img)

            # 2. Araç görselini kaydet
            if tid in vehicle_crops_cache:
                veh_filename = f"veh_id{tid}.jpg"
                cv2.imwrite(os.path.join(crops_dir, veh_filename), vehicle_crops_cache[tid])

        rec = {
            "arac_id": tid,
            "arac_tipi": ev["arac_tipi"],
            "plaka": final_plate,
            "plaka_turu": final_type,
            "plaka_guven": round(final_conf, 2),
            "yon": ev["yon"],
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
        cols = ["arac_id", "arac_tipi", "plaka", "plaka_turu", "plaka_guven", "yon", "video_saniyesi", "kare_no", "plaka_gorsel", "arac_gorsel"]
        pd.DataFrame(columns=cols).to_csv(output_csv_path, index=False, encoding="utf-8-sig")
        print(f"[!] Belirtilen çizgiden araç geçmedi. Boş CSV şablonu oluşturuldu: {output_csv_path}")

    # JSON Kaydet (CSV ile aynı klasör ve isimle)
    json_path = os.path.splitext(output_csv_path)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(final_records, jf, ensure_ascii=False, indent=2)
    print(f"[+] JSON Raporu Kaydedildi: {json_path}")

    return {
        "total_in": line_zone.in_count,
        "total_out": line_zone.out_count,
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
        default="data/4K Road traffic video for object detection and tracking - free download now!.mp4",
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
        "--scenario",
        type=str,
        choices=["A", "B"],
        default="A",
        help="Sayım çizgisi senaryosu: A (Orta koridor), B (Tüm yol)"
    )
    parser.add_argument(
        "--line",
        type=str,
        default=None,
        help="Özel çizgi koordinatları 'x1,y1,x2,y2' (Örn: '150,800,1750,800')"
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
                print(f"[!] Hatalı --line formatı ({args.line}), varsayılan senaryo kullanılacak.")
        except Exception as e:
            print(f"[!] --line parse hatası ({e}), varsayılan senaryo kullanılacak.")

    process_video(
        video_path=args.video,
        output_video_path=args.output_video,
        output_csv_path=args.output_csv,
        crops_dir=args.crops_dir,
        vehicle_model_path=args.vehicle_model,
        plate_model_path=args.plate_model,
        scenario=args.scenario,
        custom_line=custom_line,
        flip=args.flip,
        conf_threshold=args.conf,
        max_frames=args.max_frames,
        show_preview=args.preview
    )


if __name__ == "__main__":
    main()
