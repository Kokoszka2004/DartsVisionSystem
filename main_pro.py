import cv2
import numpy as np
import math
from ultralytics import YOLO

# ================= USTAWIENIA =================
MODEL_PATH = 'DART_MASTER_4K.pt'
WEBCAM_ID = 1
CONF_THRESHOLD = 0.18
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# Wymiary standardowej tarczy w mm (do wirtualnego modelu)
REAL_RADIUS_DOUBLE = 170.0

SECTORS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

# ================= ZMIENNE GLOBALNE =================
calibration_points = []
matrix = None


def get_score_from_warped_point(x, y):
    """
    Oblicza punkty na podstawie "wyprostowanych" współrzędnych (x, y).
    """
    dist = math.sqrt(x ** 2 + y ** 2)

    r_bull_in = 6.35
    r_bull_out = 15.9
    r_triple_in = 99.0
    r_triple_out = 107.0
    r_double_in = 162.0
    r_double_out = 170.0

    if dist <= r_bull_in: return 50, "BULLSEYE"
    if dist <= r_bull_out: return 25, "25"
    if dist > r_double_out: return 0, "OUT"

    multiplier = 1
    prefix = "S"

    if r_double_in < dist <= r_double_out:
        multiplier = 2
        prefix = "D"
    elif r_triple_in < dist <= r_triple_out:
        multiplier = 3
        prefix = "T"

    angle_rad = math.atan2(y, x)
    angle_deg = math.degrees(angle_rad)

    adjusted_angle = angle_deg + 90 + 9
    if adjusted_angle < 0: adjusted_angle += 360

    sector_idx = int(adjusted_angle / 18) % 20
    base_score = SECTORS[sector_idx]

    return base_score * multiplier, f"{prefix}{base_score}"


def mouse_callback(event, x, y, flags, param):
    global calibration_points, matrix

    if event == cv2.EVENT_LBUTTONDOWN and len(calibration_points) < 4:
        calibration_points.append((x, y))
        print(f"✅ Punkt {len(calibration_points)} dodany: {(x, y)}")

        if len(calibration_points) == 4:
            calculate_matrix()


def calculate_matrix():
    global matrix, calibration_points
    print("🔄 Obliczam macierz perspektywy...")

    src_pts = np.float32(calibration_points)

    R = REAL_RADIUS_DOUBLE
    dst_pts = np.float32([
        [0, -R],  # Góra (Double 20)
        [R, 0],   # Prawo (Double 6)
        [0, R],   # Dół (Double 3)
        [-R, 0]   # Lewo (Double 11)
    ])

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    print("🚀 KALIBRACJA ZAKOŃCZONA! Możesz grać.")


# ================= START (TUTAJ BYŁA ZMIANA) =================

print("Wczytuję model...")
model = YOLO(MODEL_PATH)

# 1. Próbujemy otworzyć kamerę z DirectShow (pomaga na Windows wymusić HD)
cap = cv2.VideoCapture(WEBCAM_ID, cv2.CAP_DSHOW)

# Jeśli DirectShow nie zadziała, próbujemy normalnie
if not cap.isOpened():
    print("Nie udało się otworzyć z DirectShow, próbuję standardowo...")
    cap = cv2.VideoCapture(WEBCAM_ID)

# 2. Najpierw ustawiamy rozdzielczość...
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Opcjonalnie: Próba zablokowania ostrości (żeby nie pływała)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
# cap.set(cv2.CAP_PROP_FOCUS, 255) # Odkomentuj jeśli obraz będzie nieostry (zakres 0-255)

# 3. ...a dopiero potem sprawdzamy, czy kamera posłuchała
real_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
real_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print(f"--------------------------------------------------")
print(f"📷 KAMERA DZIAŁA W: {int(real_w)}x{int(real_h)}")
print(f"--------------------------------------------------")

if real_w < 1280:
    print("⚠️ UWAGA! Rozdzielczość jest niska (640x480 lub mniej).")
    print("   Model może nie widzieć grotów!")
    print("   Upewnij się, że kamera jest w porcie USB 3.0 (niebieskim).")

# 1. Tworzymy okno z flagą WINDOW_NORMAL (pozwala na zmianę rozmiaru)
cv2.namedWindow("DART PRO", cv2.WINDOW_NORMAL)

# 2. Wymuszamy rozmiar wyświetlania na np. 1280x720 (lub mniejszy)
# To zmienia tylko WYGLĄD okna, nie zmienia jakości analizy modelu!
cv2.resizeWindow("DART PRO", 1024, 576)

cv2.setMouseCallback("DART PRO", mouse_callback)

print("INSTRUKCJA (Klikaj dokładnie w tej kolejności!):")
print("1. Zewnętrzny drut DOUBLE 20 (Góra)")
print("2. Zewnętrzny drut DOUBLE 6 (Prawo)")
print("3. Zewnętrzny drut DOUBLE 3 (Dół)")
print("4. Zewnętrzny drut DOUBLE 11 (Lewo)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Błąd odczytu klatki z kamery!")
        break

    display_frame = frame.copy()

    # --- RYSOWANIE KALIBRACJI ---
    for pt in calibration_points:
        cv2.circle(display_frame, pt, 5, (0, 255, 255), -1)

    if len(calibration_points) < 4:
        texts = ["Kliknij: DOUBLE 20 (Gora)", "Kliknij: DOUBLE 6 (Prawo)", "Kliknij: DOUBLE 3 (Dol)",
                 "Kliknij: DOUBLE 11 (Lewo)"]
        msg = texts[len(calibration_points)]
        cv2.putText(display_frame, msg, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    else:
        # --- TRYB GRY ---
        results = model.predict(frame, conf=CONF_THRESHOLD, imgsz=2560, verbose=False)
        #ciezko stwierdzic czy ten aumnet pomaga czy nie
        #results = model.predict(frame, conf=CONF_THRESHOLD, imgsz=2560, augment=True, verbose=False)

        current_scores = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            tip_x = int((x1 + x2) / 2)
            tip_y = int((y1 + y2) / 2)

            point_vec = np.array([[[tip_x, tip_y]]], dtype=np.float32)
            warped_point = cv2.perspectiveTransform(point_vec, matrix)

            wx = warped_point[0][0][0]
            wy = warped_point[0][0][1]

            points, label = get_score_from_warped_point(wx, wy)
            current_scores.append(points)

            cv2.circle(display_frame, (tip_x, tip_y), 4, (0, 0, 255), -1)
            cv2.putText(display_frame, label, (tip_x, tip_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        total = sum(current_scores)
        cv2.putText(display_frame, f"SUMA: {total}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("DART PRO", display_frame)

    key = cv2.waitKey(10)
    if key == ord('q'): break
    if key == ord('r'):
        calibration_points = []
        matrix = None
        print("Reset kalibracji!")

#uczenie na bieżąco
    if key == ord('s'):
        import time
        filename = f"trudny_przypadek_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame) # Zapisujemy czystą klatkę (bez ramek)
        print(f"📸 ZAPISANO ZDJĘCIE: {filename}")
cap.release()
cv2.destroyAllWindows()