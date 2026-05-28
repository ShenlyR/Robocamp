import os
import cv2
import math
from ultralytics import YOLO

# ---------- Konfigurasi ----------
MODEL_PATH  = os.path.join("models", "bola.pt")
VIDEO_PATH  = os.path.join("video", "Edit Perjalanan Kapal.mp4")
CONF_THRESH = 0.35

# Warna BGR
COLOR_RED      = (0, 0, 255)
COLOR_GREEN    = (0, 255, 0)
COLOR_CYAN     = (255, 255, 0)
COLOR_YELLOW   = (0, 255, 255)
COLOR_WHITE    = (255, 255, 255)
COLOR_BLACK    = (0, 0, 0)
COLOR_ORANGE   = (0, 165, 255)

# Class ID dari model bola.pt
# Sesuaikan jika urutan class berbeda
CLASS_GREEN    = 0   # Buoy Hijau
CLASS_RED      = 1   # Buoy Merah
CLASS_OBSTACLE = 2   # Obstacle (jika ada di model); jika tidak ada, diabaikan

# Navigasi
FRAME_CENTER_RATIO = 0.5   # titik tengah frame secara horizontal
DEAD_ZONE_PX       = 40    # toleransi dead zone (pixel) agar tidak goyang terus

# ---------- Helper ----------
def box_center(x1, y1, x2, y2):
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def draw_hud(frame, arah, koreksi_m, n_red, n_green, n_obstacle, fps):
    """Gambar HUD info di pojok kiri atas seperti referensi."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 145), COLOR_BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Arah
    arah_color = COLOR_CYAN if arah == "LURUS" else COLOR_YELLOW
    cv2.putText(frame, f"ARAH : {arah}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, arah_color, 2)

    # Info count
    cv2.putText(frame, f"Buoy Merah  : {n_red}",      (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_RED, 2)
    cv2.putText(frame, f"Buoy Hijau  : {n_green}",    (10, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GREEN, 2)
    cv2.putText(frame, f"Obstacle    : {n_obstacle}", (10, 106),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
    cv2.putText(frame, f"FPS : {fps:.1f}",            (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1)

    # Koreksi arah
    if arah != "LURUS":
        simbol = "<=" if arah == "LEFT" else "=>"
        ket = f"{simbol} Koreksi {'KIRI' if arah == 'LEFT' else 'KANAN'}  {koreksi_m:.2f}m"
        cv2.putText(frame, ket, (10, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_YELLOW, 1)

def draw_bottom_banner(frame, arah, koreksi_m, frame_w, frame_h):
    """Banner bawah besar seperti referensi."""
    if arah == "LURUS":
        text  = "==>  LURUS  <=="
        color = COLOR_CYAN
    elif arah == "LEFT":
        text  = f"<=  ARAHKAN KAPAL KE KIRI  {koreksi_m:.2f} m"
        color = COLOR_YELLOW
    else:
        text  = f"ARAHKAN KAPAL KE KANAN  =>  {koreksi_m:.2f} m"
        color = COLOR_YELLOW

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, frame_h - 55), (frame_w, frame_h), COLOR_BLACK, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, text, (frame_w // 2 - 280, frame_h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# ---------- Main ----------
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model tidak ditemukan: {MODEL_PATH}")
        return

    model = YOLO(MODEL_PATH)

    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] Video tidak ditemukan: {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("[ERROR] Tidak dapat membuka video.")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    center_x = int(frame_w * FRAME_CENTER_RATIO)

    print(f"Video: {frame_w}x{frame_h} @ {fps_src:.1f} FPS")
    print("Tekan 'q' untuk keluar | 'p' untuk pause | '+'/'-' untuk kecepatan")

    paused   = False
    speed    = 1          # multiplier delay
    prev_t   = cv2.getTickCount()

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Video selesai.")
                break

            # ---- Inference ----
            results = model(frame, conf=CONF_THRESH, verbose=False)

            buoy_red    = []   # list of (cx, cy, x1,y1,x2,y2, conf)
            buoy_green  = []
            obstacles   = []

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cx, cy = box_center(x1, y1, x2, y2)

                    if cls_id == CLASS_RED:
                        buoy_red.append((cx, cy, x1, y1, x2, y2, conf))
                    elif cls_id == CLASS_GREEN:
                        buoy_green.append((cx, cy, x1, y1, x2, y2, conf))
                    elif cls_id == CLASS_OBSTACLE:
                        obstacles.append((cx, cy, x1, y1, x2, y2, conf))

            # ---- Gambar bounding box ----
            for (cx, cy, x1, y1, x2, y2, conf) in buoy_red:
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_RED, 2)
                cv2.putText(frame, f"red {conf:.0%}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_RED, 2)
                cv2.circle(frame, (cx, cy), 5, COLOR_RED, -1)

            for (cx, cy, x1, y1, x2, y2, conf) in buoy_green:
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_GREEN, 2)
                cv2.putText(frame, f"green {conf:.0%}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_GREEN, 2)
                cv2.circle(frame, (cx, cy), 5, COLOR_GREEN, -1)

            for (cx, cy, x1, y1, x2, y2, conf) in obstacles:
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ORANGE, 2)
                cv2.putText(frame, f"obstacle {conf:.0%}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_ORANGE, 2)

            # ---- Hitung Midpoint ----
            midpoint_x = None
            midpoint_y = None
            arah       = "LURUS"
            koreksi_m  = 0.0

            if buoy_red and buoy_green:
                # Ambil buoy terbawah (paling dekat kapal) dari masing-masing sisi
                nearest_red   = max(buoy_red,   key=lambda b: b[1])  # cy terbesar
                nearest_green = max(buoy_green, key=lambda b: b[1])

                rx, ry = nearest_red[0],   nearest_red[1]
                gx, gy = nearest_green[0], nearest_green[1]

                midpoint_x = (rx + gx) // 2
                midpoint_y = (ry + gy) // 2

                # Garis lintasan
                cv2.line(frame, (rx, ry), (gx, gy), COLOR_CYAN, 2)

                # Titik midpoint
                cv2.circle(frame, (midpoint_x, midpoint_y), 10, COLOR_CYAN, -1)
                cv2.putText(frame, "MIDPOINT", (midpoint_x + 12, midpoint_y + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_CYAN, 2)

                # Offset dari tengah frame
                offset_px = midpoint_x - center_x

                # Konversi px → meter (estimasi kasar, sesuaikan konstanta)
                PX_PER_METER = 60
                koreksi_m    = abs(offset_px) / PX_PER_METER

                if offset_px < -DEAD_ZONE_PX:
                    arah = "LEFT"
                elif offset_px > DEAD_ZONE_PX:
                    arah = "RIGHT"
                else:
                    arah = "LURUS"

            elif buoy_red:
                # Hanya ada buoy merah → belok kanan (hindari)
                arah = "RIGHT"
            elif buoy_green:
                # Hanya ada buoy hijau → belok kiri (hindari)
                arah = "LEFT"

            # ---- Garis tengah frame (referensi) ----
            cv2.line(frame, (center_x, 0), (center_x, frame_h), (80, 80, 80), 1)

            # ---- FPS ----
            cur_t = cv2.getTickCount()
            fps   = cv2.getTickFrequency() / (cur_t - prev_t)
            prev_t = cur_t

            # ---- HUD & Banner ----
            draw_hud(frame, arah, koreksi_m,
                     len(buoy_red), len(buoy_green), len(obstacles), fps)
            draw_bottom_banner(frame, arah, koreksi_m, frame_w, frame_h)

            # ---- Offset text bawah ----
            if midpoint_x is not None:
                offset_px = midpoint_x - center_x
                cv2.putText(frame, f"Offset: {offset_px:+d}px",
                            (frame_w // 2 - 60, frame_h - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)

        # ---- Tampilkan ----
        cv2.imshow("ASV Navigation System - RoboCamp 2026 (q=quit, p=pause, +/-=speed)", frame)

        delay = max(1, int((1000 / fps_src) / speed))
        key   = cv2.waitKey(delay) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("PAUSE" if paused else "LANJUT")
        elif key == ord('+') or key == ord('='):
            speed = min(speed * 2, 8)
            print(f"Speed: {speed}x")
        elif key == ord('-'):
            speed = max(speed / 2, 0.25)
            print(f"Speed: {speed}x")

    cap.release()
    cv2.destroyAllWindows()
    print("Selesai.")

if __name__ == "__main__":
    main()