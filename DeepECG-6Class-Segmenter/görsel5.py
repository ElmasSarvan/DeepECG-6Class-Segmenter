import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from scipy.signal import butter, filtfilt

# --- 1. YOLLAR VE AYARLAR ---
base_path = r'C:\Users\elmas\PyCharmMiscProject\ekg_projesi\final_veriseti_v4'
data_path = r'C:\Users\elmas\PyCharmMiscProject\ekg_projesi\mitbih_database'

img_dir = os.path.join(base_path, 'images')
mask_dir = os.path.join(base_path, 'masks')
os.makedirs(img_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

IMG_SIZE = 256
DPI = 100
FIG_SIZE = float(IMG_SIZE) / float(DPI)
WINDOW_SIZE = 1500
FS = 360

# --- RENK HARİTASI (İstediğin Düzenleme) ---
# Pace (/) vuruşunu 1.0 (Beyaz) yaptık, diğerleri koyulaşarak gidiyor.
CLASS_MAP = {
    '/': 1.0,  # Pace vuruşu -> BEYAZ
    'A': 0.8,  # Atriyal Erken Vuruş
    'V': 0.6,  # Ventriküler Erken Vuruş
    'L': 0.4,  # Sol Dal Bloğu
    'R': 0.3,  # Sağ Dal Bloğu
    'N': 0.2  # Normal Vuruş
}


# --- 2. HASSAS FİLTRELEME (Korundu) ---
def ekg_filtrele(data, lowcut=0.5, highcut=40.0, fs=360.0, order=4):
    if len(data) <= order * 3:
        return data
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(0.99, highcut / nyq)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, np.nan_to_num(data))


# --- 3. SENKRONİZE KAYDETME (Tip Hataları Düzeltildi) ---
def save_synchronized(chunk, ann_chunk, patient_id, start_idx, suffix=""):
    file_name = f"P{patient_id}_{start_idx}{suffix}.png"

    # 1. EKG SİNYAL GÖRÜNTÜSÜ
    fig_img = plt.figure(figsize=(FIG_SIZE, FIG_SIZE), dpi=DPI)
    # IDE uyarısını gidermek için tuple olarak geçiyoruz
    ax_img = fig_img.add_axes((0.0, 0.0, 1.0, 1.0))

    # Sadece ilk iki kanalı çiz
    for i, col in enumerate(chunk.columns[1:3]):
        y_raw = chunk[col].values
        y_filt = ekg_filtrele(y_raw)

        y_min = np.min(y_filt)
        y_max = np.max(y_filt)
        y_norm = (y_filt - y_min) / (y_max - y_min + 1e-5)

        ax_img.plot(y_norm + (i * 1.3), color='black' if i == 0 else 'gray', linewidth=1.0)

    ax_img.set_xlim(0, WINDOW_SIZE)
    ax_img.set_ylim(-0.2, 2.8)
    ax_img.axis('off')
    fig_img.savefig(os.path.join(img_dir, file_name))
    plt.close(fig_img)

    # 2. MASKE GÖRÜNTÜSÜ
    fig_mask = plt.figure(figsize=(FIG_SIZE, FIG_SIZE), dpi=DPI, facecolor='black')
    ax_mask = fig_mask.add_axes((0.0, 0.0, 1.0, 1.0))
    ax_mask.set_facecolor('black')

    for _, ann in ann_chunk.iterrows():
        label = str(ann['Type']).strip()
        if label in CLASS_MAP:
            c_val = CLASS_MAP[label]
            pos = int(ann['Sample']) - start_idx
            # Dikey hizalama korunuyor
            ax_mask.axvline(x=pos, color=(c_val, c_val, c_val), linewidth=8, antialiased=False)

    ax_mask.set_xlim(0, WINDOW_SIZE)
    ax_mask.set_ylim(0, 1)
    ax_mask.axis('off')
    fig_mask.savefig(os.path.join(mask_dir, file_name), facecolor='black')
    plt.close(fig_mask)


# --- 4. ANA DÖNGÜ (Veri Çoğaltma Aktif) ---
def prepare_dataset():
    csv_files = sorted(glob.glob(os.path.join(data_path, "*.csv")))
    print(f"Islem basliyor: {len(csv_files)} dosya bulundu.")

    for file in csv_files:
        p_name = os.path.basename(file).split('.')[0]
        # ID'yi temizle
        p_id = "".join([s for s in p_name if s.isdigit()])
        ann_path = os.path.join(data_path, f"{p_id}annotations.txt")

        if not os.path.exists(ann_path):
            continue

        try:
            df_sig = pd.read_csv(file, skipinitialspace=True)
            df_sig.columns = [str(c).replace("'", "").strip() for c in df_sig.columns]

            # Annotation Okuma
            ann_list = []
            with open(ann_path, 'r') as f:
                for line in f:
                    p = line.split()
                    if len(p) >= 3 and ":" in p[0]:
                        try:
                            s_idx = int(p[1])
                            t_sym = p[2]
                            if t_sym in CLASS_MAP:
                                ann_list.append({'Sample': s_idx, 'Type': t_sym})
                        except ValueError:
                            continue

            df_ann = pd.DataFrame(ann_list)
            if df_ann.empty:
                continue

            # Pencereleme ve Veri Çoğaltma
            for start in range(0, len(df_sig) - WINDOW_SIZE, WINDOW_SIZE):
                c_ann = df_ann[(df_ann['Sample'] >= start) & (df_ann['Sample'] < start + WINDOW_SIZE)]

                if not c_ann.empty:
                    # Normal Kayıt
                    save_synchronized(df_sig.iloc[start:start + WINDOW_SIZE], c_ann, p_name, start)

                    # VERİ ÇOĞALTMA (A ve V vuruşları için dikey kaydırma)
                    chunk_types = c_ann['Type'].unique()
                    if 'A' in chunk_types or 'V' in chunk_types:
                        for shift in [120, -120]:
                            n_s = start + shift
                            if 0 <= n_s and n_s + WINDOW_SIZE <= len(df_sig):
                                n_ann = df_ann[(df_ann['Sample'] >= n_s) & (df_ann['Sample'] < n_s + WINDOW_SIZE)]
                                if not n_ann.empty:
                                    save_synchronized(df_sig.iloc[n_s:n_s + WINDOW_SIZE], n_ann, p_name, n_s,
                                                      suffix=f"_aug_{shift}")

            print(f"Tamamlandi: {p_name}")

        except Exception as e:
            print(f"Hata {p_name}: {str(e)}")


if __name__ == "__main__":
    prepare_dataset()