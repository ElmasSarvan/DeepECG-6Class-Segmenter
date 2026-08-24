import tensorflow as tf
import numpy as np
import cv2
import os
import glob

# --- 1. AYARLAR ---
# Model ismini v4 olarak güncelledim
MODEL_PATH = 'ekg_segmentasyon_v4_final.h5'
IMG_SIZE = 128
WINDOW_SIZE_SAMP = 1500
SAMPLING_RATE = 360

# Senin belirlediğin renk sırasına göre isim listesi
CLASS_NAMES = ["Arkaplan", "Normal (N)", "LBBB (L)", "RBBB (R)", "PVC (V)", "APC (A)", "Paced (/)"]

# --- 2. MODEL YÜKLEME ---
if not os.path.exists(MODEL_PATH):
    print(f"Hata: {MODEL_PATH} dosyası bulunamadı!")
    exit()

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("[INFO] Model başarıyla yüklendi.")


def analyze_specific_patient(images_path, patient_id="P100"):
    # Tüm resimleri tara
    search_pattern = os.path.join(images_path, f"{patient_id}_*.png")
    all_files = sorted(glob.glob(search_pattern))

    # --- KRİTİK: AUGMENTASYON DOSYALARINI ALMA ---
    # Dosya adında "_aug_" geçmeyenleri (orijinal pencereleri) seçiyoruz
    image_files = [f for f in all_files if "_aug_" not in f]

    if not image_files:
        print(f"Hata: {patient_id} için orijinal veri bulunamadı!")
        return

    findings = {name: 0 for name in CLASS_NAMES if name != "Arkaplan"}
    all_rr_intervals = []

    print(f"[INFO] Hasta {patient_id} Analiz Ediliyor ({len(image_files)} Orijinal Pencere)...")

    for img_p in image_files:
        img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        if img is None: continue

        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype('float32') / 255.0
        img_input = np.expand_dims(np.expand_dims(img_resized, axis=-1), axis=0)

        # Tahmin
        pred = model.predict(img_input, verbose=0)
        mask = np.argmax(pred[0], axis=-1)

        # --- NABIZ ANALİZİ (Korundu) ---
        col_has_beat = np.any(mask > 0, axis=0)
        beat_indices = np.where(col_has_beat)[0]

        if len(beat_indices) > 0:
            peaks = []
            temp_group = [beat_indices[0]]
            for i in range(1, len(beat_indices)):
                if beat_indices[i] - beat_indices[i - 1] < 10:
                    temp_group.append(beat_indices[i])
                else:
                    peaks.append(np.mean(temp_group))
                    temp_group = [beat_indices[i]]
            peaks.append(np.mean(temp_group))

            time_per_window = WINDOW_SIZE_SAMP / SAMPLING_RATE
            for i in range(1, len(peaks)):
                pixel_diff = peaks[i] - peaks[i - 1]
                time_diff_seconds = (pixel_diff / IMG_SIZE) * time_per_window
                if 0.3 < time_diff_seconds < 2.0:
                    all_rr_intervals.append(time_diff_seconds)

        # --- TEŞHİS MANTIĞI (Senin Renk Değerlerine Göre) ---
        # 1: Normal, 2: L, 3: R, 4: V, 5: A, 6: Paced (/)
        unique, counts = np.unique(mask, return_counts=True)
        stats = dict(zip(unique, counts))

        # Piksellerin en yoğun olduğu anormal sınıfı saptıyoruz
        anormal_pixels = {k: v for k, v in stats.items() if k >= 2 and v > 8}

        if anormal_pixels:
            best_cls = max(anormal_pixels, key=anormal_pixels.get)
            findings[CLASS_NAMES[best_cls]] += 1
        elif 1 in stats and stats[1] > 10:
            findings[CLASS_NAMES[1]] += 1

    # --- 3. RAPORLAMA ---
    print("\n" + "=" * 60)
    print(f"      HASTA {patient_id} - ANALİZ RAPORU (Sadece Orijinal Veri)      ")
    print("=" * 60)

    if all_rr_intervals:
        avg_rr = np.mean(all_rr_intervals)
        bpm = 60 / avg_rr
        std_rr = np.std(all_rr_intervals)
        print(f"> ORTALAMA NABIZ     : {int(bpm)} BPM")
        ritim_notu = "DÜZENSİZ (Aritmi Riski)" if std_rr > 0.12 else "DÜZENLİ"
        print(f"> RİTİM DURUMU       : {ritim_notu}")

    print("-" * 60)
    print(f"TESPİT EDİLEN VURUŞ TİPLERİ:")

    anormal_total = 0
    for label, count in findings.items():
        if count > 0:
            print(f"> {label.ljust(22)}: {str(count).rjust(4)} pencere")
            if "Normal" not in label:
                anormal_total += count

    print("-" * 60)
    if anormal_total > 5:
        print(f"SONUÇ: ANORMAL (ARİTMİK) BULGULAR SAPTANDI.")
        if findings["Paced (/)"] > 0: print("NOT  : Kayıtta Paced (/) vuruşlar tespit edildi.")
    else:
        print("SONUÇ: TÜM KAYITLAR NORMAL SINIRLARDA.")
    print("=" * 60)


# Kullanım
image_path = r'C:\Users\elmas\PyCharmMiscProject\ekg_projesi\final_veriseti_v4\images'
analyze_specific_patient(image_path, patient_id="P100")