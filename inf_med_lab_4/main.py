import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as plt

# Ścieżki
test_path = r"C:\Users\jacek\OneDrive\Pulpit\inf_med_4\Altered-custom\1__M_Left_index_finger_szum.BMP"
dataDir = r"C:\Users\jacek\OneDrive\Pulpit\inf_med_4\Real_subset"

# Wczytanie obrazu testowego
test_original = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
if test_original is None:
    print("❌ Nie udało się wczytać obraz testowy.")
    exit()

# Funkcje pomocnicze
def preprocess(image):
    return image

def features_extraction(image):
    return sift.detectAndCompute(image, None)

# Inicjalizacja
sift = cv2.SIFT_create()
test_preprocessed = preprocess(test_original)
keypoints_1, descriptors_1 = features_extraction(test_preprocessed)

# Szukaj najlepszego dopasowania
best = {
    "file": "",
    "image": None,
    "keypoints_2": [],
    "matches": [],
    "match_time": 0,
    "mean_diff": 0
}

for file in os.listdir(dataDir):
    path = os.path.join(dataDir, file)
    db_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if db_image is None:
        continue

    start = time.time()
    keypoints_2, descriptors_2 = features_extraction(preprocess(db_image))

    matches = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10), dict()).knnMatch(descriptors_1, descriptors_2, k=2)

    match_points = []
    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            match_points.append(p)

    end = time.time()
    match_time = end - start

    keypoints_count = min(len(keypoints_1), len(keypoints_2))
    if keypoints_count == 0:
        continue

    match_ratio = len(match_points) / keypoints_count

    # Jeśli to najlepszy dotąd wynik – zapisz
    if match_ratio > len(best["matches"]) / keypoints_count:
        best.update({
            "file": file,
            "image": db_image,
            "keypoints_2": keypoints_2,
            "matches": match_points,
            "match_time": match_time,
            "mean_diff": np.mean(cv2.absdiff(test_preprocessed, preprocess(db_image)))
        })

# 📝 WYPISZ DANE DO RAPORTU
print("\nNajlepsze dopasowanie:")
print("Plik:", best["file"])
print("Czas dopasowania: {:.4f} s".format(best["match_time"]))
print("Liczba dopasowanych punktów:", len(best["matches"]))
print("Średnia różnica pikseli:", best["mean_diff"])

# 🖼️ OBRAZ RÓŻNICOWY
diff_img = cv2.absdiff(test_preprocessed, preprocess(best["image"]))
cv2.imshow("Obraz różnicowy", diff_img)

# ♟️ SZACHOWNICA (checkerboard)
checker = test_preprocessed.copy()
checker[::20, :] = preprocess(best["image"])[::20, :]
cv2.imshow("Checkerboard", checker)

# 🔗 DOPASOWANIE
result = cv2.drawMatches(test_original, keypoints_1, best["image"],
                         best["keypoints_2"], best["matches"], None)
result = cv2.resize(result, None, fx=2.0, fy=2.0)
cv2.imshow("Dopasowanie", result)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Różnica pikseli (po preprocessingu!)
diff = cv2.absdiff(test_preprocessed, preprocess(best["image"]))

# Tworzymy siatkę X, Y
X = np.arange(diff.shape[1])
Y = np.arange(diff.shape[0])
X, Y = np.meshgrid(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, diff, cmap='viridis')
ax.set_title('Różnica między obrazami (przed i po)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Różnica intensywności')
plt.show()
