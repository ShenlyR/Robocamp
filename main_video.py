#import library 
import os
import cv2
import datetime
from ultralytics import YOLO

#konfigurasi
MODEL_PATH = os.path.join('model_asv_2024', 'bola.pt')
VIDEO_PATH = os.path.join('video', 'Edit Perjalanan Kapal.mp4')

#navigasi
CONF_THRESH = 0.25
CONF_BUOY_MIN = 0.45 
PX_PER_METER = 60
OBSTACLE_MARGIN = 30
DEAD_ZONE_PX = 40
FRAME_CENTER_RATIO = 0.5

#ukuran display
DISPLAY_W = 1200
DISPLAY_H = 620

#warna 
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_CYAN = (255, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_ORANGE = (0, 165, 255) 
COLOR_GRAY = (80, 80, 80)
COLOR_DEADZONE = (160, 220, 255)

#class model
CLASS_GREEN = 0
CLASS_RED = 1

def box_center(x1, y1, x2, y2):
    return ((x1 + x2) // 2, (y1 + y2) // 2) 

def buoy_utama(buoy_list):
    if not buoy_list:
        return None
    
    filtered = [b for b in buoy_list if b[6]] 
    pool = filtered if filtered else buoy_list
    return max(pool, key=lambda b: b[1]) if pool else None

def cek_obstacle(nearest_red, nearest_green, buoy_red, buoy_green):
    if nearest_red is None or nearest_green is None:
        return False, None
    
    batas_kiri = min(nearest_red[0], nearest_green[0])
    batas_kanan = max(nearest_red[0], nearest_green[0])
    mid_lintasan = (batas_kiri + batas_kanan) // 2

    obstacle_count = 0
    left_score = 0
    right_score = 0

    for obj in buoy_red + buoy_green:
        if obj in (nearest_red, nearest_green):
            continue
        
        ox = obj[0] 

        if not (batas_kiri < ox < batas_kanan):
            continue

        obstacle_count += 1

        if ox < mid_lintasan: 
            left_score += 1
        else:
            right_score += 1
        
    if obstacle_count == 0:
        return False, None
        
    if right_score > left_score:
            return True, 'Right'
    elif left_score > right_score:
            return True, 'Left'
    else:
            return True, None

#deteksi buoy
def gambar_deteksi(frame, buoy_red, buoy_green):
    for cx, cy, x1, y1, x2, y2, conf in buoy_red:
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_RED, 2)
        cv2.putText(frame, f'red {conf:.0%}', 
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_RED, 2)
        cv2.circle(frame, (cx, cy), 5, COLOR_GREEN, -1) 
    
    for cx, cy, x1, y1, x2, y2, conf in buoy_green:
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_GREEN, 2)
        cv2.putText(frame, f'green {conf:.0%}',
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_GREEN, 2)
        cv2.circle(frame, (cx, cy), 5, COLOR_GREEN, -1)

#menentukan posisi kapal
def midpoint(frame, rx, ry, gx, gy, mid_x, mid_y):
    cv2.line(frame, (rx, ry), (gx, gy), COLOR_CYAN, 2)
    cv2.circle(frame, (mid_x, mid_y), 10, COLOR_CYAN, -1)
    cv2.putText(frame, 'MidPoint', 
                (mid_x + 12, mid_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_CYAN, 2)
#deadzone
def draw_deadzone_line(frame, x, y_start, y_end, color, seg=10, gap=5):
    y = y_start
    while y < y_end:
        y_end_seg = min(y + seg, y_end)
        cv2.line(frame, (x, y), (x, y_end_seg), color, 1)
        y += seg + gap 

def deadzone(frame, center_x, frame_h): 
    cv2.line(frame, (center_x, 0), (center_x, frame_h), COLOR_GRAY, 1)
    draw_deadzone_line(frame, center_x - DEAD_ZONE_PX, 0, frame_h, COLOR_DEADZONE)
    draw_deadzone_line(frame, center_x + DEAD_ZONE_PX, 0, frame_h, COLOR_DEADZONE)

#informasi panel 
def hud(frame, arah, koreksi_m, n_red, n_green, n_obstacle, fps, elapsed_sec):
    overlay = frame.copy() 
    cv2.rectangle(overlay, (0, 0), (450, 180), COLOR_BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame) 
    arah_warna = COLOR_CYAN if arah == 'Lurus' else COLOR_YELLOW
    cv2.putText(frame, f'Arah: {arah}', 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, arah_warna, 2)
    cv2.putText(frame, f'Buoy Merah: {n_red}', 
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_RED, 2)
    cv2.putText(frame, f"Buoy Hijau: {n_green}",
                (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                COLOR_ORANGE if n_obstacle > 0 else COLOR_WHITE, 2)
    cv2.putText(frame, f'Obstacle: {n_obstacle}',
                (10, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                COLOR_ORANGE if n_obstacle > 0 else  COLOR_WHITE, 2)
    
    #datetime 
    h = int(elapsed_sec // 3600)
    m = int(elapsed_sec % 3600) // 60 
    s = int(elapsed_sec % 60)
    ts = f'{h:02d}:{m:02d}:{s:02d}'
    cv2.putText(frame, f'Video: {ts}',
                (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1)
    
    now = datetime.datetime.now().strftime('%H:%M:%S')
    cv2.putText(frame, f'Waktu Saat Ini: {now}', 
        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1) 

#Keputusan navigasi kapal
def draw_bottom_banner(frame, arah, koreksi_m, frame_w, frame_h):
    teks = {
        'Lurus': ('==> Lurus <==', COLOR_CYAN), 
        'Left': (f'<= Arahkan Kapal ke Kiri {koreksi_m:.2f} m', COLOR_YELLOW),
        'Right': (f'Arahkan Kapal ke Kanan {koreksi_m:.2f} m', COLOR_YELLOW),
    }
    text, color = teks.get(arah, ('', COLOR_WHITE))

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, frame_h - 55), (frame_w, frame_h), COLOR_BLACK, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, text, 
                (frame_w // 2 - 280, frame_h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
#main 
def main():
    if not os.path.exists(MODEL_PATH):
        print(f'[ERROR] Model tidak ditemukan: {MODEL_PATH}')
        return
    if not os.path.exists(VIDEO_PATH):
        print(f'[ERROR] Video tidak ditemukan: {VIDEO_PATH}')
        return
    
    model = YOLO(MODEL_PATH)
    capture = cv2.VideoCapture(VIDEO_PATH)

    if not capture.isOpened():
        print('[ERROR] Tidak dapat membuka video')
        return
    
    frame_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src =  capture.get(cv2.CAP_PROP_FPS) or 30 
    center_x = int(frame_w * FRAME_CENTER_RATIO) 

    print(f'Video: {frame_w}x{frame_h} @ {fps_src:.1f} FPS')
    print('Tekan q keluar | p pause | +/- kecepatan')

    paused = False 
    speed = 1
    prev_t = cv2.getTickCount()

    #datetime
    frame_count = 0

    while True:
        if not paused:
            ret, frame = capture.read()
            if not ret:
                print('Video selesai')
                break

            frame_count += 1
            elapsed_sec = frame_count / fps_src

            results = model(frame, conf=CONF_THRESH, verbose=False)
            buoy_red = []
            buoy_green = []

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cx, cy = box_center(x1, y1, x2, y2) 
                    
                    if conf < CONF_BUOY_MIN:
                        continue

                    if cls_id ==  CLASS_RED: 
                        buoy_red.append((cx, cy, x1, y1, x2, y2, conf))
                    elif cls_id == CLASS_GREEN:
                        buoy_green.append((cx, cy, x1, y1, x2, y2, conf))

            gambar_deteksi(frame, buoy_red, buoy_green)

            midpoint_x = None
            arah = 'Lurus'
            koreksi_m = 0.0 
            n_obs = 0
            nearest_red = buoy_utama(buoy_red)
            nearest_green = buoy_utama(buoy_green)

            if nearest_red and nearest_green:
                rx, ry = nearest_red[0], nearest_red[1]
                gx, gy = nearest_green[0], nearest_green[1] 
                midpoint_x = (rx + gx) // 2 
                midpoint_y = (ry + gy) // 2

                midpoint(frame, rx, ry, gx, gy, midpoint_x, midpoint_y)

                offset_px = midpoint_x - center_x 
                koreksi_m = abs(offset_px) / PX_PER_METER 

                ada_obstacle, arah_hindaran = cek_obstacle(
                    nearest_red, nearest_green, buoy_red, buoy_green
                )

                if ada_obstacle:
                    arah = arah_hindaran
                    n_obs = 1
                    cv2.putText(frame, 'Obstacled Terdeteksi!', 
                                (frame_w // 2 - 150, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_ORANGE, 2) 
                else:
                    if offset_px < - DEAD_ZONE_PX:
                        arah = 'Left'
                    elif offset_px > DEAD_ZONE_PX:
                        arah = 'Right'
                    else:
                        arah = 'Lurus'
            elif nearest_red:
                arah = 'Right'
            elif nearest_green:
                arah = 'Left'
            else: 
                cv2.putText(frame, 'Peringatan: Buoy Tidak Terdeteksi!',
                            (frame_w // 2 - 150, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_RED, 2)
            
            deadzone(frame, center_x, frame_h)

            cur_t = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (cur_t - prev_t)
            prev_t = cur_t

            hud(frame, arah, koreksi_m,
                len(buoy_red), len(buoy_green), n_obs, fps, elapsed_sec)
            draw_bottom_banner(frame, arah, koreksi_m, frame_w, frame_h)

            if midpoint_x is not None:
                offset_color = COLOR_YELLOW if arah != 'Lurus' else COLOR_CYAN
                cv2.putText(frame, f'Offset: {offset_px:+d} px', 
                            (frame_w - 175, frame_h - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, offset_color, 1)
            
        display = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
        cv2.imshow(
            'ASV Navigation System'
            '(q=quit, p=pause, +/-=speed)', display
        )
        delay = max(1, int((1000 / fps_src) / speed))
        key = cv2.waitKey(delay) & 0xFF

        if key == ord('q'):
            break 
        elif key == ord('p'):
            paused = not paused
            print('Pause' if paused else "Lanjut") 
        elif key in (ord('+'), ord('=')):
            speed = min(speed * 2, 8)
            print( f'Speed: {speed} x')
        elif key == ord('-'):
            speed = max(speed / 2, 0.25)
            print(f'Speed: {speed} x')

    capture.release()
    cv2.destroyAllWindows()
    print('Load Model Selesai')

if __name__ == '__main__':
    main()
