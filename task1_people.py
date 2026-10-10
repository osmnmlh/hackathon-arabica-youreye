"""Görev 1: Kişi Tespiti, Sayım ve Cinsiyet Tahmini — Rev 1.2"""

import argparse
import os
import sys
import time
import collections
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
import supervision as sv
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Ön-işleme: Karanlık karelerde kontrast iyileştirme
# ---------------------------------------------------------------------------
def enhance_frame(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    if float(l_channel.mean()) < 100:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(l_channel)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return frame

# ---------------------------------------------------------------------------
# Cinsiyet Tahmini (CLIP Zero-Shot)
# ---------------------------------------------------------------------------
class GenderEstimator:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self.processor = None
        self.prompts = [
            "a photo of a man walking",
            "a photo of a woman walking",
            "a photo of a woman wearing a headscarf",
            "a photo of a woman wearing a hijab",
        ]
        self.text_inputs = None
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
            self.text_inputs = self.processor(text=self.prompts, return_tensors="pt", padding=True).to(self.device)
            self.is_ready = True
            print("[INFO] GenderEstimator (CLIP) başarıyla yüklendi.")
        except Exception as e:
            print(f"[UYARI] Cinsiyet modeli yüklenemedi ({e}).")
            self.is_ready = False

    def predict(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        if not self.is_ready or crop_bgr is None or crop_bgr.size == 0:
            return "Bilinmiyor", 0.5
            
        h, w = crop_bgr.shape[:2]
        # Çok küçük crop'larda analiz yapma
        if h < 80 or w < 40:
            return "Bilinmiyor", 0.5

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
                return "Erkek", male_prob
            else:
                return "Kadin", female_prob
        except Exception as e:
            return "Bilinmiyor", 0.5

# ---------------------------------------------------------------------------
# HUD Çizimi
# ---------------------------------------------------------------------------
def draw_hud(
    frame: np.ndarray, active_count: int, total_unique: int, 
    in_count: int, out_count: int, male_count: int, female_count: int, 
    fps: float, current_frame: int, total_frames: int
):
    overlay = frame.copy()
    hud_w, hud_h = 400, 240
    x0, y0 = 20, 20
    cv2.rectangle(overlay, (x0, y0), (x0 + hud_w, y0 + hud_h), (15, 20, 25), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + hud_w, y0 + hud_h), (70, 160, 245), 2)

    cv2.putText(frame, "AI OPERATIONS - TASK 1 (PEOPLE)", (x0 + 15, y0 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (x0 + 10, y0 + 45), (x0 + hud_w - 10, y0 + 45), (100, 100, 100), 1)

    total_count = in_count + out_count
    
    cv2.putText(frame, f"Aktif Kisi: {active_count}", (x0 + 15, y0 + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Tekil Görülen: {total_unique}", (x0 + 200, y0 + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

    cv2.putText(frame, f"Giris (IN): {in_count}", (x0 + 15, y0 + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 235, 120), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Cikis (OUT): {out_count}", (x0 + 200, y0 + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 130, 245), 2, cv2.LINE_AA)

    cv2.putText(frame, f"Erkek: {male_count}", (x0 + 15, y0 + 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 180, 70), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Kadin: {female_count}", (x0 + 200, y0 + 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 120, 245), 1, cv2.LINE_AA)

    cv2.putText(frame, f"Toplam Gecis: {total_count}", (x0 + 15, y0 + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    progress_str = f"Kare: {current_frame}/{total_frames}" if total_frames > 0 else f"Kare: {current_frame}"
    cv2.putText(frame, f"{progress_str} | FPS: {fps:.1f}", (x0 + 15, y0 + 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

# ---------------------------------------------------------------------------
# Ana İşlem Hattı
# ---------------------------------------------------------------------------
def run_pipeline(
    video_path: str, output_dir: str, model_path: str,
    max_frames: Optional[int] = None, conf_threshold: float = 0.15,
    iou_threshold: float = 0.5, device: str = "cpu",
    line_start_xy: Tuple[int, int] = (1097, 200), line_end_xy: Tuple[int, int] = (800, 220),
    enable_gender: bool = True, save_video: bool = True, show_window: bool = False,
):
    print("=" * 70)
    print("Arabica x Youreye - Görev 1: Kişi Tespiti & Sayım (Rev 1.2)")
    print("=" * 70)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Girdi video bulunamadı: {video_path}")

    model = YOLO(model_path)
    gender_estimator = GenderEstimator(device=device) if enable_gender else None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")

    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    effective_total = min(total_video_frames, max_frames) if max_frames else total_video_frames

    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, "task1_kisi_sayimi.mp4")
    csv_output_path = os.path.join(output_dir, "task1_kisi_sayimi.csv")

    writer = None
    if save_video:
        writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # LineZone
    line_start, line_end = sv.Point(*line_start_xy), sv.Point(*line_end_xy)
    try:
        line_zone = sv.LineZone(start=line_start, end=line_end, triggering_anchors=[sv.Position.CENTER])
    except Exception:
        line_zone = sv.LineZone(start=line_start, end=line_end)  # Fallback for different sv versions

    line_annotator = sv.LineZoneAnnotator(thickness=3, color=sv.Color(0, 255, 200), text_thickness=2, text_scale=0.7)
    
    # Specific color annotators for gender
    color_male = sv.Color(255, 120, 30)    # Blue for male (BGR->RGB order depending on sv.Color, actually sv.Color is RGB)
    color_female = sv.Color(200, 30, 255)  # Pink for female
    color_unk = sv.Color(128, 128, 128)    # Gray for unknown
    
    box_annotators = {
        "Erkek": sv.BoxAnnotator(thickness=2, color=color_male),
        "Kadin": sv.BoxAnnotator(thickness=2, color=color_female),
        "Bilinmiyor": sv.BoxAnnotator(thickness=2, color=color_unk)
    }
    label_annotators = {
        "Erkek": sv.LabelAnnotator(text_scale=0.5, text_thickness=1, color=color_male),
        "Kadin": sv.LabelAnnotator(text_scale=0.5, text_thickness=1, color=color_female),
        "Bilinmiyor": sv.LabelAnnotator(text_scale=0.5, text_thickness=1, color=color_unk)
    }

    # İstatistik Takibi
    all_seen_track_ids = set()
    track_history: Dict[int, dict] = {}
    gender_history: Dict[int, collections.deque] = {}  # track_id -> deque of (gender, conf)
    
    frame_idx = 0
    start_time, last_fps_time = time.time(), time.time()
    calc_fps = fps
    gender_eval_interval = 8  # Her 8 karede bir analiz

    use_half = (device != "cpu")
    print(f"[INFO] Video işleniyor (FP16: {use_half})...")
    
    while cap.isOpened():
        if max_frames and frame_idx >= max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        enhanced_frame = enhance_frame(frame)

        # YOLO FP16
        try:
            results = model.track(
                enhanced_frame, persist=True, tracker="bytetrack.yaml",
                conf=conf_threshold, iou=iou_threshold, classes=[0],
                device=device, verbose=False, imgsz=1280, half=use_half
            )[0]
        except Exception:
            results = model.track(
                enhanced_frame, persist=True, tracker="bytetrack.yaml",
                conf=conf_threshold, iou=iou_threshold, classes=[0],
                device=device, verbose=False, imgsz=1280
            )[0]

        detections = sv.Detections.from_ultralytics(results)
        if detections.tracker_id is not None:
            valid = detections.tracker_id != None
            detections = detections[valid]
            
        has_tracks = len(detections) > 0

        # Sayım Tetiklemesi
        if has_tracks:
            try:
                line_zone.trigger(detections)
            except Exception:
                pass

        labels = []
        active_count = len(detections) if has_tracks else 0

        if has_tracks:
            for i, track_id in enumerate(detections.tracker_id):
                track_id = int(track_id)
                all_seen_track_ids.add(track_id)

                xyxy = detections.xyxy[i].astype(int)
                conf = float(detections.confidence[i])
                x1, y1, x2, y2 = max(0, xyxy[0]), max(0, xyxy[1]), min(width, xyxy[2]), min(height, xyxy[3])
                crop = frame[y1:y2, x1:x2]

                if track_id not in track_history:
                    track_history[track_id] = {
                        "track_id": track_id, "first_frame": frame_idx, "last_frame": frame_idx,
                        "gender": "Bilinmiyor", "gender_conf": 0.0, "det_conf": conf
                    }
                    gender_history[track_id] = collections.deque(maxlen=10)
                else:
                    track_history[track_id]["last_frame"] = frame_idx
                    track_history[track_id]["det_conf"] = max(track_history[track_id]["det_conf"], conf)

                current_info = track_history[track_id]

                # Cinsiyet değerlendirmesi
                if gender_estimator and (frame_idx % gender_eval_interval == 0 or len(gender_history[track_id]) == 0):
                    pred, g_conf = gender_estimator.predict(crop)
                    if pred != "Bilinmiyor":
                        gender_history[track_id].append((pred, g_conf))
                
                # Çoğunluk oylaması (Majority Voting)
                if len(gender_history[track_id]) > 0:
                    votes = [g[0] for g in gender_history[track_id]]
                    confs = [g[1] for g in gender_history[track_id]]
                    
                    male_votes = votes.count("Erkek")
                    female_votes = votes.count("Kadin")
                    
                    if male_votes > female_votes:
                        current_info["gender"] = "Erkek"
                        current_info["gender_conf"] = float(np.mean([confs[i] for i, v in enumerate(votes) if v == "Erkek"]))
                    else:
                        current_info["gender"] = "Kadin"
                        current_info["gender_conf"] = float(np.mean([confs[i] for i, v in enumerate(votes) if v == "Kadin"]))

                g_str, g_conf = current_info["gender"], current_info["gender_conf"]
                if g_str != "Bilinmiyor":
                    labels.append(f"#{track_id} {g_str} ({g_conf:.0%}) | D:{conf:.2f}")
                else:
                    labels.append(f"#{track_id} Analiz ediliyor... | D:{conf:.2f}")

        # Görselleştirme ve Kafe Sınırları
        annotated = frame.copy()
        
        # Kafe sınırlarını çiz
        cv2.line(annotated, (76, 120), (6, 198), (0, 0, 255), 2)
        cv2.line(annotated, (60, 111), (1478, 60), (0, 0, 255), 2)
        cv2.line(annotated, (1481, 68), (1913, 430), (0, 0, 255), 2)

        if len(detections) > 0:
            g_indices = {"Erkek": [], "Kadin": [], "Bilinmiyor": []}
            g_labels = {"Erkek": [], "Kadin": [], "Bilinmiyor": []}
            
            for i, (tid, lbl) in enumerate(zip(detections.tracker_id, labels)):
                gen = track_history[int(tid)]["gender"]
                g_indices[gen].append(i)
                g_labels[gen].append(lbl)
                
            for gen in ["Erkek", "Kadin", "Bilinmiyor"]:
                if len(g_indices[gen]) > 0:
                    sub_dets = detections[g_indices[gen]]
                    annotated = box_annotators[gen].annotate(scene=annotated, detections=sub_dets)
                    annotated = label_annotators[gen].annotate(scene=annotated, detections=sub_dets, labels=g_labels[gen])
                    
        annotated = line_annotator.annotate(frame=annotated, line_counter=line_zone)

        # Cinsiyet Dağılımını Hesapla
        male_count = sum(1 for v in track_history.values() if v["gender"] == "Erkek")
        female_count = sum(1 for v in track_history.values() if v["gender"] == "Kadin")

        if frame_idx % 15 == 0:
            now = time.time()
            calc_fps = 15.0 / max(1e-4, (now - last_fps_time))
            last_fps_time = now

        draw_hud(annotated, active_count, len(all_seen_track_ids), line_zone.in_count, line_zone.out_count, male_count, female_count, calc_fps, frame_idx, effective_total)

        if writer:
            writer.write(annotated)

        if show_window:
            cv2.imshow("Task 1", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    if show_window:
        cv2.destroyAllWindows()

    total_time = time.time() - start_time
    print("-" * 70)
    print(f"[SONUÇ] Özet Bilgiler:")
    print(f" - Toplam Süre: {total_time:.2f} saniye")
    print(f" - Ortalama FPS: {frame_idx / max(1e-4, total_time):.1f}")
    print(f" - Toplam Tekil Kişi: {len(all_seen_track_ids)}")
    print(f" - Hat Geçişleri → IN: {line_zone.in_count}, OUT: {line_zone.out_count} (Toplam: {line_zone.in_count + line_zone.out_count})")
    print(f" - Cinsiyet → Erkek: {male_count}, Kadın: {female_count}")

    if track_history:
        df = pd.DataFrame(list(track_history.values()))
        df.to_csv(csv_output_path, index=False, encoding="utf-8-sig")
        print(f"[SONUÇ] CSV raporu: {csv_output_path}")
    if save_video and os.path.exists(output_video_path):
        print(f"[SONUÇ] Çıktı videosu: {output_video_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Task 1: Kişi Sayımı ve Cinsiyet Sınıflandırma", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--video", type=str, default="data/kisi_sayimi.mp4", help="Girdi video yolu")
    parser.add_argument("--output_dir", type=str, default="outputs/task1_outputs/task1_kisi_sayimi-rev1.2", help="Çıktı klasörü")
    parser.add_argument("--model", type=str, default="models/yolov8m.pt", help="YOLO modeli")
    parser.add_argument("--max_frames", type=int, default=None, help="Maksimum işlenecek kare")
    parser.add_argument("--conf", type=float, default=0.15, help="Tespit güven eşiği")
    parser.add_argument("--iou", type=float, default=0.4, help="NMS IoU eşiği")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Cihaz (cuda/cpu)")
    parser.add_argument("--line_start", type=int, nargs=2, default=[1100, 180], help="Çizgi başı (x y)")
    parser.add_argument("--line_end", type=int, nargs=2, default=[600, 180], help="Çizgi sonu (x y)")
    parser.add_argument("--no_gender", action="store_true", help="Cinsiyeti devre dışı bırak")
    parser.add_argument("--no_video", action="store_true", help="Videoyu kaydetme")
    parser.add_argument("--show", action="store_true", help="Önizlemeyi göster")
    args = parser.parse_args()

    run_pipeline(
        video_path=args.video, output_dir=args.output_dir, model_path=args.model,
        max_frames=args.max_frames, conf_threshold=args.conf, iou_threshold=args.iou, device=args.device,
        line_start_xy=tuple(args.line_start), line_end_xy=tuple(args.line_end),
        enable_gender=not args.no_gender, save_video=not args.no_video, show_window=args.show,
    )


if __name__ == "__main__":
    main()
