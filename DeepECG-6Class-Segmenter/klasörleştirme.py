import os
import shutil

# --- AYARLAR ---
# Kendi bilgisayarındaki tam yolu buraya yaz
base_path = r'C:\Users\elmas\PyCharmMiscProject\ekg_projesi\final_veriseti_v4_deneme'
images_dir = os.path.join(base_path, 'images')


def hastaları_klasörle():
    # Klasördeki tüm dosyaları listele
    files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]

    print(f"Toplam {len(files)} dosya bulundu. İşlem başlıyor...")

    for file_name in files:
        # Dosya isminden hasta ID'sini al (Örn: P100_0.png -> P100)
        # Alt çizgiye (_) göre ayırıp ilk parçayı alıyoruz
        if "_" in file_name:
            patient_id = file_name.split("_")[0]
        else:
            # Eğer dosya isminde alt çizgi yoksa noktaya göre ayır (Örn: P100.png)
            patient_id = file_name.split(".")[0]

        # Hasta adına özel klasör yolu oluştur
        patient_folder = os.path.join(images_dir, patient_id)

        # Klasör yoksa oluştur
        if not os.path.exists(patient_folder):
            os.makedirs(patient_folder)
            print(f"Yeni klasör oluşturuldu: {patient_id}")

        # Dosyayı yeni klasörüne taşı
        src_path = os.path.join(images_dir, file_name)
        dst_path = os.path.join(patient_folder, file_name)

        try:
            shutil.move(src_path, dst_path)
        except Exception as e:
            print(f"Hata ({file_name}): {str(e)}")

    print("İşlem başarıyla tamamlandı!")


if __name__ == "__main__":
    hastaları_klasörle()