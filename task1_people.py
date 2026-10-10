"""Görev 1: Kişi Tespiti, Sayım ve Cinsiyet Tahmini — Rev 1.1

Arabica x Youreye Hackathon 2026.
YASAL UYARI: Bu model çıktısı yalnızca tahmindir, kesin kimlik doğrulama niteliği taşımaz.
"""

import argparse
import os
import sys
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
import supervision as sv
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Ön-işleme: Karanlık karelerde kontrast iyileştirme (CLAHE)
# ---------------------------------------------------------------------------
def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """Karanlık/düşük kontrastlı kareleri CLAHE ile iyileştirir.

    Modelin gece veya loş aydınlatma koşullarında daha iyi tespit
    yapabilmesi için parlaklık ve kontrast normalleştirmesi uygular.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    mean_l = float(l_channel.mean())

    # Sadece karanlık karelere uygula (parlak karelere dokunma)
    if mean_l < 100:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(l_channel)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return frame


# ---------------------------------------------------------------------------
# Cinsiyet Tahmini (CLIP Zero-Shot)
# ---------------------------------------------------------------------------
class GenderEstimator:
    """CLIP tabanlı sıfır atışlı (zero-shot) cinsiyet tahminleyicisi.

    Yasal/Teknik Çekince:
    Bu model çıktısı yalnızca tahmindir, kesin kimlik doğrulama niteliği taşımaz.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self.processor = None
        self.labels = ["Erkek", "Kadin"]
        self.prompts = [
            "a photo of a man walking",
            "a photo of a woman walking",
            "a photo of a woman wearing a headscarf",
            "a photo of a woman wearing a hijab",
        ]
        self.text_inputs = None
        self.cache: Dict[int, Tuple[str, float, int]] = {}  # track_id -> (gender, conf, eval_count)
        self.is_ready = False
        self._load_model()

    def _load_model(self):
        try:
            from transformers import CLIPModel, CLIPProcessor

            model_id = "openai/clip-vit-base-patch32"
            try:
                self.model = CLIPModel.from_pretrained(model_id, local_files_only=True).to(self.device)
                self.processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
            except Exception:
                self.model = CLIPModel.from_pretrained(model_id).to(self.device)
                self.processor = CLIPProcessor.from_pretrained(model_id)

            self.model.eval()
            self.text_inputs = self.processor(
                text=self.prompts,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            self.is_ready = True
            print("[INFO] GenderEstimator (CLIP) başarıyla yüklendi.")
        except Exception as e:
            print(f"[UYARI] Cinsiyet modeli yüklenemedi ({e}). Varsayılan mod kullanılacak.")
            self.is_ready = False

    def predict(self, crop_bgr: np.ndarray, track_id: Optional[int] = None) -> Tuple[str, float]:
        """Kırpılan kişi görüntüsünden cinsiyet tahmini yapar.

        Her track_id için birden fazla değerlendirme yaparak en yüksek
        güvenilirlik skorunu (majority vote benzeri) kaydeder.
        """
        # Kesinleşmiş önbellekteki sonucu döndür
        if track_id is not None and track_id in self.cache:
            cached = self.cache[track_id]
            if cached[1] >= 0.65 and cached[2] >= 2:
                return (cached[0], cached[1])

        if not self.is_ready or crop_bgr is None or crop_bgr.size == 0:
            return ("Bilinmiyor", 0.5)

        h, w = crop_bgr.shape[:2]
        if h < 30 or w < 15:
            return ("Bilinmiyor", 0.5)

        try:
            rgb_img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)

            img_inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=self.text_inputs["input_ids"],
                    attention_mask=self.text_inputs["attention_mask"],
                    pixel_values=img_inputs["pixel_values"],
                )
                probs = outputs.logits_per_image.softmax(dim=1)[0].cpu().numpy()

            male_prob = float(probs[0])
            female_prob = float(probs[1] + probs[2] + probs[3])

            if male_prob > female_prob:
                pred_gender = "Erkek"
                conf = male_prob
            else:
                pred_gender = "Kadin"
                conf = female_prob

            # Önbelleğe güncelle (en yüksek conf kazanır)
            if track_id is not None:
                if track_id in self.cache:
                    old_gender, old_conf, old_count = self.cache[track_id]
                    if conf > old_conf:
                        self.cache[track_id] = (pred_gender, conf, old_count + 1)
                    else:
                        self.cache[track_id] = (old_gender, old_conf, old_count + 1)
                else:
                    self.cache[track_id] = (pred_gender, conf, 1)

            return (pred_gender, conf)
        except Exception:
            return ("Bilinmiyor", 0.5)

    def get_cached(self, track_id: int) -> Tuple[str, float]:
        """Önbellekteki sonucu döndür."""
        if track_id in self.cache:
            return (self.cache[track_id][0], self.cache[track_id][1])
        return ("Bilinmiyor", 0.5)


# ---------------------------------------------------------------------------
# HUD Çizimi
# ---------------------------------------------------------------------------
def draw_hud(
    frame: np.ndarray,
    active_count: int,
    total_unique: int,
    in_count: int,
    out_count: int,
    male_count: int,
    female_count: int,
    fps: float,
    current_frame: int,
    total_frames: int,
):
    """Görüntü üzerine modern ve şık bir HUD bilgi paneli çizer."""
    overlay = frame.copy()
    hud_w, hud_h = 360, 220
    x0, y0 = 20, 20
    cv2.rectangle(overlay, (x0, y0), (x0 + hud_w, y0 + hud_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + hud_w, y0 + hud_h), (70, 160, 245), 2)

    # Başlık
    cv2.putText(frame, "ARABICA x YOUREYE - TASK 1", (x0 + 12, y0 + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (x0 + 10, y0 + 36), (x0 + hud_w - 10, y0 + 36), (100, 100, 100), 1)

    # Metrikler
    cv2.putText(frame, f"Aktif Kisi: {active_count}", (x0 + 15, y0 + 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Toplam Tekil: {total_unique}", (x0 + 180, y0 + 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    cv2.putText(frame, f"Giris (IN): {in_count}", (x0 + 15, y0 + 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 235, 120), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Cikis (OUT): {out_count}", (x0 + 180, y0 + 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 130, 245), 2, cv2.LINE_AA)

    cv2.putText(frame, f"Erkek: {male_count}", (x0 + 15, y0 + 128),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 180, 70), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Kadin: {female_count}", (x0 + 180, y0 + 128),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 120, 245), 1, cv2.LINE_AA)

    progress_str = f"Kare: {current_frame}/{total_frames}" if total_frames > 0 else f"Kare: {current_frame}"
    cv2.putText(frame, f"{progress_str} | FPS: {fps:.1f}", (x0 + 15, y0 + 162),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)

    # Yasal Not
    cv2.putText(frame, "* Cinsiyet tahmini yalnizca istatistiki kestirimdir.",
                (x0 + 12, y0 + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Ana İşlem Hattı
# ---------------------------------------------------------------------------
def run_pipeline(
    video_path: str,
    output_dir: str,
    model_path: str,
    max_frames: Optional[int] = None,
    conf_threshold: float = 0.15,
    iou_threshold: float = 0.5,
    device: str = "cpu",
    line_start_xy: Tuple[int, int] = (1097, 200),
    line_end_xy: Tuple[int, int] = (800, 220),
    enable_gender: bool = True,
    save_video: bool = True,
    show_window: bool = False,
):
    print("=" * 70)
    print("Arabica x Youreye Hackathon 2026 - Görev 1: Kişi Tespiti & Sayım")
    print("Rev 1.1 — IN/OUT yönü tersine çevrildi, başörtülü kadın tespiti iyileştirildi")
    print("Yasal Uyarı: Bu model çıktısı yalnızca tahmindir, kesin kimlik doğrulama niteliği taşımaz.")
    print("=" * 70)

    # 1. Video Kontrolü
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Girdi video dosyası bulunamadı: {video_path}")

    # 2. Model Kontrolü ve Hazırlık
    if not os.path.exists(model_path):
        # Daha güçlü yolov8s modeline otomatik geç
        alt_paths = ["models/yolov8s.pt", "yolov8s.pt", "models/yolov8n.pt", "yolov8n.pt"]
        found = False
        for alt in alt_paths:
            if os.path.exists(alt):
                model_path = alt
                found = True
                break
        if not found:
            print(f"[INFO] Model bulunamadı, yolov8s.pt indiriliyor...")

    print(f"[INFO] YOLO Modeli yükleniyor: {model_path} (Cihaz: {device})")
    model = YOLO(model_path)

    # 3. Cinsiyet Modeli
    gender_estimator = None
    if enable_gender:
        print(f"[INFO] Cinsiyet tahminleyicisi başlatılıyor (Cihaz: {device})...")
        gender_estimator = GenderEstimator(device=device)

    # 4. Video Okuma Hazırlığı
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    effective_total = min(total_video_frames, max_frames) if max_frames else total_video_frames
    print(f"[INFO] Video: {width}x{height}, FPS: {fps:.1f}, Toplam: {total_video_frames} (İşlenecek: {effective_total})")

    # Çıktı Dizini: outputs/task1_kisi_sayimi-rev1.1/  → .mp4 ve .csv birlikte
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, "task1_kisi_sayimi-rev1.1.mp4")
    csv_output_path = os.path.join(output_dir, "task1_kisi_sayimi-rev1.1.csv")

    # Video Yazıcı
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        print(f"[INFO] Çıktı videosu: {output_video_path}")

    # 5. Sayım Hattı — Kapı önü koordinatları
    # start ve end'in yönü IN/OUT semantiğini belirler:
    #   Çizginin solundan sağına (start→end vektörünün sağ tarafına) geçiş = IN
    #   Çizginin sağından soluna (start→end vektörünün sol tarafına) geçiş = OUT
    line_start = sv.Point(x=line_start_xy[0], y=line_start_xy[1])
    line_end = sv.Point(x=line_end_xy[0], y=line_end_xy[1])

    line_zone = sv.LineZone(
        start=line_start,
        end=line_end,
        triggering_anchors=[sv.Position.BOTTOM_CENTER],
    )
    print(f"[INFO] Sayım hattı: ({line_start_xy[0]},{line_start_xy[1]}) → ({line_end_xy[0]},{line_end_xy[1]})")

    line_annotator = sv.LineZoneAnnotator(
        thickness=3,
        color=sv.Color(r=0, g=255, b=200),
        text_thickness=2,
        text_scale=0.7,
    )
    box_annotator = sv.BoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.TRACK)
    label_annotator = sv.LabelAnnotator(
        text_scale=0.45,
        text_thickness=1,
        color_lookup=sv.ColorLookup.TRACK,
    )

    # İstatistik Takibi
    all_seen_track_ids = set()
    track_history: Dict[int, dict] = {}
    frame_idx = 0
    start_time = time.time()
    last_fps_time = time.time()
    calc_fps = fps

    # Cinsiyet değerlendirme hız sınırlama sayacı
    gender_eval_interval = 5  # Her 5 karede bir yeni cinsiyet tahmini yap

    print(f"\n[INFO] İşlem başlatılıyor (conf={conf_threshold}, iou={iou_threshold})...")
    while cap.isOpened():
        if max_frames and frame_idx >= max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Karanlık kare iyileştirme
        enhanced_frame = enhance_frame(frame)

        # YOLO ile Takip (Person class = 0)
        results = model.track(
            enhanced_frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=conf_threshold,
            iou=iou_threshold,
            classes=[0],
            device=device,
            verbose=False,
            imgsz=1280,
        )[0]

        detections = sv.Detections.from_ultralytics(results)
        has_tracks = detections.tracker_id is not None and len(detections) > 0
        if has_tracks:
            valid = detections.tracker_id != None  # noqa: E711
            detections = detections[valid]
            has_tracks = len(detections) > 0

        # Sayım Tetiklemesi
        if has_tracks:
            crossed_in, crossed_out = line_zone.trigger(detections)

        # Cinsiyet ve Kimlik Analizi
        labels = []
        active_count = len(detections) if has_tracks else 0

        if has_tracks:
            for i, track_id in enumerate(detections.tracker_id):
                track_id = int(track_id)
                all_seen_track_ids.add(track_id)

                xyxy = detections.xyxy[i].astype(int)
                x1 = max(0, xyxy[0])
                y1 = max(0, xyxy[1])
                x2 = min(width, xyxy[2])
                y2 = min(height, xyxy[3])
                crop = frame[y1:y2, x1:x2]

                if track_id not in track_history:
                    track_history[track_id] = {
                        "track_id": track_id,
                        "first_frame": frame_idx,
                        "last_frame": frame_idx,
                        "gender": "Bilinmiyor",
                        "gender_conf": 0.5,
                        "crops_evaluated": 0,
                    }
                else:
                    track_history[track_id]["last_frame"] = frame_idx

                current_info = track_history[track_id]

                # Cinsiyet tahmini: ilk 3 değerlendirmede veya düşük güvende yeniden dene
                should_eval_gender = (
                    gender_estimator
                    and frame_idx % gender_eval_interval == 0
                    and (
                        current_info["crops_evaluated"] < 3
                        or current_info["gender_conf"] < 0.60
                    )
                )
                if should_eval_gender and crop.size > 0 and (y2 - y1) >= 40 and (x2 - x1) >= 20:
                    pred_gender, conf = gender_estimator.predict(crop, track_id=track_id)
                    cached = gender_estimator.get_cached(track_id)
                    current_info["gender"] = cached[0]
                    current_info["gender_conf"] = cached[1]
                    current_info["crops_evaluated"] += 1

                gender_tag = current_info["gender"]
                gender_conf = current_info["gender_conf"]
                labels.append(f"#{track_id} {gender_tag} ({gender_conf:.0%})")

        # Görselleştirme
        annotated_frame = frame.copy()
        if len(detections) > 0:
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            if labels and len(labels) == len(detections):
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame, detections=detections, labels=labels
                )

        annotated_frame = line_annotator.annotate(frame=annotated_frame, line_counter=line_zone)

        # Cinsiyet Dağılımını Hesapla
        male_count = sum(1 for v in track_history.values() if v["gender"] == "Erkek")
        female_count = sum(1 for v in track_history.values() if v["gender"] == "Kadin")

        # FPS hesabı
        if frame_idx % 10 == 0:
            now = time.time()
            calc_fps = 10.0 / max(1e-4, (now - last_fps_time))
            last_fps_time = now

        draw_hud(
            frame=annotated_frame,
            active_count=active_count,
            total_unique=len(all_seen_track_ids),
            in_count=line_zone.in_count,
            out_count=line_zone.out_count,
            male_count=male_count,
            female_count=female_count,
            fps=calc_fps,
            current_frame=frame_idx,
            total_frames=effective_total,
        )

        if writer:
            writer.write(annotated_frame)

        if show_window:
            cv2.imshow("Arabica x Youreye - Task 1", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Kullanıcı tarafından durduruldu.")
                break

        if frame_idx % 50 == 0 or frame_idx == effective_total:
            elapsed = time.time() - start_time
            proc_fps = frame_idx / max(1e-4, elapsed)
            print(
                f"[{frame_idx:04d}/{effective_total}] "
                f"FPS: {proc_fps:.1f} | "
                f"Aktif: {active_count:02d} | "
                f"Tekil: {len(all_seen_track_ids):02d} | "
                f"IN: {line_zone.in_count} OUT: {line_zone.out_count} | "
                f"E: {male_count} K: {female_count}"
            )

    cap.release()
    if writer:
        writer.release()
    if show_window:
        cv2.destroyAllWindows()

    total_time = time.time() - start_time
    print("-" * 70)
    print(f"[SONUÇ] {frame_idx} kare / {total_time:.2f}s ({frame_idx / max(1e-4, total_time):.1f} FPS)")
    print(f"[SONUÇ] Toplam Tekil Kişi: {len(all_seen_track_ids)}")
    print(f"[SONUÇ] Hat Geçişleri → Giriş (IN): {line_zone.in_count}, Çıkış (OUT): {line_zone.out_count}")
    print(f"[SONUÇ] Cinsiyet → Erkek: {male_count}, Kadın: {female_count}, Bilinmeyen: {len(all_seen_track_ids) - male_count - female_count}")

    # CSV Raporu
    if track_history:
        df = pd.DataFrame(list(track_history.values()))
        df.to_csv(csv_output_path, index=False, encoding="utf-8-sig")
        print(f"[SONUÇ] CSV raporu: {csv_output_path}")

    if save_video and os.path.exists(output_video_path):
        print(f"[SONUÇ] Video: {output_video_path}")

    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Arabica x Youreye Hackathon 2026 — Task 1 Rev 1.1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", type=str, default="data/kisi_sayimi.mp4",
                        help="Girdi video dosya yolu")
    parser.add_argument("--output_dir", type=str, default="outputs/task1_kisi_sayimi-rev1.1",
                        help="Çıktı klasörü (mp4 + csv birlikte)")
    parser.add_argument("--model", type=str, default="models/yolov8s.pt",
                        help="YOLO model ağırlık dosyası (yolov8s önerilir)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="İşlenecek maksimum kare sayısı")
    parser.add_argument("--conf", type=float, default=0.15,
                        help="Tespit güven eşiği (düşük = karanlıkta daha iyi)")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="NMS / Takip IoU eşiği")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Hesaplama cihazı")
    parser.add_argument("--line_start", type=int, nargs=2, default=[1097, 200],
                        help="Sayım hattı başlangıç noktası (x y)")
    parser.add_argument("--line_end", type=int, nargs=2, default=[800, 220],
                        help="Sayım hattı bitiş noktası (x y)")
    parser.add_argument("--no_gender", action="store_true",
                        help="Cinsiyet tahminini devre dışı bırak")
    parser.add_argument("--no_video", action="store_true",
                        help="Çıktı videosunu kaydetme")
    parser.add_argument("--show", action="store_true",
                        help="Canlı önizleme penceresini göster")

    args = parser.parse_args()

    run_pipeline(
        video_path=args.video,
        output_dir=args.output_dir,
        model_path=args.model,
        max_frames=args.max_frames,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        line_start_xy=tuple(args.line_start),
        line_end_xy=tuple(args.line_end),
        enable_gender=not args.no_gender,
        save_video=not args.no_video,
        show_window=args.show,
    )


if __name__ == "__main__":
    main()
