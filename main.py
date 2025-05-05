import cv2
import numpy as np
import os

# Parametry kamery i układu
FOCAL_LENGTH = 2444  # ogniskowa w px (przykład)
BASELINE = 0.88       # przesunięcie między zdjęciami w metrach

# Ścieżki do obrazów
image_paths = ["./wybrane/C1.png", "./wybrane/C2.png"]
boxes = []

# Przetwarzanie każdego obrazu
for i, path in enumerate(image_paths):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Nie można wczytać obrazu: {path}")

    os.makedirs("tmp", exist_ok=True)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 3
    )

    # Kontury
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_box = None
    max_area = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = w / h if h != 0 else 0
        if area > 1000 and 0.5 < aspect_ratio < 2.0:
            if area > max_area:
                max_area = area
                best_box = (x, y, w, h)

    if best_box:
        x, y, w, h = best_box
        boxes.append((x, y, w, h))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print(f"Obraz {i+1}: x={x}, y={y}, szerokość={w}px, wysokość={h}px")
    else:
        boxes.append(None)
        print(f"Obraz {i+1}: Nie znaleziono pudełka")

    cv2.imwrite(f"tmp/box_img_{i+1}.jpg", image)

# Sprawdzenie, czy wykryto pudełka na obu obrazach
if None not in boxes:
    (x1, y1, w1, h1) = boxes[0]
    (x2, y2, w2, h2) = boxes[1]

    # Użyj środka pudełka do disparity
    center1 = x1 + w1 // 2
    center2 = x2 + w2 // 2
    disparity = abs(center1 - center2)

    if disparity == 0:
        print("Disparity wynosi 0 – nie można oszacować głębokości.")
    else:
        # Estymacja głębokości (Z)
        Z = (FOCAL_LENGTH * BASELINE) / disparity
        print(f"Estymowana głębokość obiektu Z = {Z:.3f} m")

        # Załóżmy, że wysokość w pikselach z pierwszego obrazu to h1
        # Oszacowanie wysokości w metrach:
        pixel_height = h1
        height_in_meters = (pixel_height * Z) / FOCAL_LENGTH
        print(f"Estymowana wysokość obiektu: {height_in_meters:.3f} m")

else:
    print("Nie wykryto pudełka na obu obrazach – nie można estymować wysokości.")
