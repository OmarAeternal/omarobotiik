import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while True:

    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Definisikan rentang warna biru dalam ruang warna HSV
    lower_blue = np.array([90, 0, 0])
    upper_blue = np.array([140, 255, 200])

    # Buat mask untuk mendeteksi warna biru
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Terapkan mask ke frame asli
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Tampilkan hasil
    cv2.imshow('Deteksi Warna Biru', result)

    # Tekan 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan resource
cap.release()
cv2.destroyAllWindows()
