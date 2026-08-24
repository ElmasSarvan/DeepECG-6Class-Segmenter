import sys
import os
import glob
import cv2
import shutil
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import requests  # ENGEL TANIMAYAN COHERE BULUT BAĞLANTISI
import time      # HATA ANINDA BEKLEME SÜRESİ İÇİN
from scipy.signal import butter, filtfilt, find_peaks
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,
                             QHBoxLayout, QWidget, QFileDialog, QTextEdit, QLabel,
                             QProgressBar)
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt
from tensorflow.keras import layers, models, Input

# --- AYARLAR ---
MODEL_PATH = 'ekg_transformer_unet_v3_final.h5'
IMG_SIZE = 128
CHANNELS = 2
TEMP_DIR = "csv_to_images_temp"
CLASS_NAMES = ["Arkaplan", "Normal (N)", "LBBB (L)", "RBBB (R)", "PVC (V)", "APC (A)", "Paced (/)"]
WINDOW_SIZE_SAMP = 1500
SAMPLING_RATE = 360


# --- VERİ HAZIRLAMA KODUNDAKİ HASSAS FİLTRELEME (Milimetrik Eşlendi) ---
def butter_bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=360.0, order=4):
    if len(data) <= order * 3:
        return data
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(0.99, highcut / nyq)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, np.nan_to_num(data))


# --- MİLİMETRİK HTML BAŞLIK VE ALT SATIR DÜZENLİ COHERE MOTORU ---
class EKGInterpreter:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = "https://api.cohere.com/v1/chat"

    def get_interpretation_pure_llm(self, findings, baskin_sonuc, bpm, ritim):
        # ⚠️ YENİLİK: Başlıkları alt satıra indiren ve açık mavi yapan ultra hassas pro-prompt
        prompt = f"""
        Sen kıdemli bir Kardiyoloji Uzmanısın. Aşağıda bir derin öğrenme modelinden çıkan ham EKG analiz verileri yer almaktadır:
        - Ölçülen Nabız: {bpm} BPM
        - Genel Ritim Durumu: {ritim}
        - Baskın Atım Türü: {baskin_sonuc}
        - Tüm Atım Dağılımları: {findings}

        Senden ricam, bu verileri KENDİ GENİŞ KLİNİK BİLGİNLE SENTEZLEYEREK yorumlaman. 
        Yanıtını kesinlikle düz yazı blokları halinde birleşik verme! Tamamen HTML formatında (<b>, <br>, <ul>, <li>) etiketlerini kullanarak ADIM ADIM, maddeler halinde TERTEMİZ yaz.

        KRİTİK GÖRSEL KURAL: Raporun 3 ana başlığı mutlaka alt satırda, bağımsız olarak başlayacaktır. Başlık adından hemen önce <br><br> ekleyerek bir alt satıra inmesini sağlayın. Başlıkların rengi mutlaka açık mavi (#00d2d3) olmalıdır.

        Lütfen tam olarak şu şablonu eksiksiz doldur:

        <br><br><b><font color='#00d2d3'>1. Klinik Anlam ve Değerlendirme:</font></b>
        <ul>
          <li>Baskın olan atım türünü ve bunun klinik karşılığını (Örn: RBBB ise Sağ Demet Dal Bloğu) net cümlelerle açıkla.</li>
          <li>Sinyalde saptanan diğer tüm patolojik veya ektopik vuru sayılarını (PVC, APC, RBBB, LBBB) tek tek listeyle tıbbi olarak yorumla.</li>
        </ul>

        <br><br><b><font color='#00d2d3'>2. Risk Derecesi:</font></b>
        <ul>
          <li>Nabız hızını ({bpm} BPM) ve tespit edilen aritmi yoğunluğunu harmanlayarak hastanın risk durumunu (Düşük, Orta, Yüksek) net bir cümleyle gerekçelendirerek yaz.</li>
        </ul>

        <br><br><b><font color='#00d2d3'>3. Önerilen İleri Tetkikler (Nokta Atışı):</font></b>
        <ul>
          <li>Bu hastanın spesifik bulgularına ve aritmi yüküne göre yapılması gereken medikal testleri adım adım gerekçeleriyle alt alta listele.</li>
        </ul>

        ÖNEMLİ KURAL: Giriş, selamlaşma veya açıklama cümleleri ('İşte HTML raporu' vb.) asla kurma. Doğrudan birinci başlık etiketleriyle yanıtı başlat.
        """

        payload = {
            "model": "command-r-plus-08-2024",
            "message": prompt,
            "temperature": 0.1
        }

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(self.url, json=payload, headers=headers, timeout=25)
                if response.status_code == 200:
                    return response.json().get('text', '').strip()
                elif response.status_code == 429:
                    time.sleep(2)
                    continue
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                if attempt < max_retries - 1:
                    time.sleep(1.5)
                    continue
                else:
                    return "<b style='color:#ff9f43;'>[BAĞLANTI GECİKTİ]</b> Sunucu paket yanıtı zaman aşımına uğradı. Lütfen butona bir kez daha basarak analizi tazeleyin."
            except Exception as e:
                return f"<b style='color:#ff9f43;'>[BAĞLANTI HATASI]</b> Yapay zeka katmanına ulaşılamadı."


# --- MODEL MİMARİSİ (Aynen Korundu) ---
def transformer_block(inputs, num_heads, embed_dim, ff_dim, dropout=0.1):
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    attention_output = layers.Dropout(dropout)(attention_output)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    ffn_output = layers.Dense(ff_dim, activation="relu")(out1)
    ffn_output = layers.Dense(embed_dim)(ffn_output)
    ffn_output = layers.Dropout(dropout)(ffn_output)
    return layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)


def build_transformer_unet():
    inputs = Input((IMG_SIZE, IMG_SIZE, CHANNELS))
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    b1 = layers.Conv2D(256, 3, activation='relu', padding='same')(p3)
    shape = b1.shape
    flattened = layers.Reshape((shape[1] * shape[2], shape[3]))(b1)
    trans_out = transformer_block(flattened, num_heads=4, embed_dim=shape[3], ff_dim=512)
    reshaped = layers.Reshape((shape[1], shape[2], shape[3]))(trans_out)

    u1 = layers.UpSampling2D((2, 2))(reshaped)
    m1 = layers.concatenate([u1, c3])
    d1 = layers.Conv2D(128, 3, activation='relu', padding='same')(m1)
    u2 = layers.UpSampling2D((2, 2))(d1)
    m2 = layers.concatenate([u2, c2])
    d2 = layers.Conv2D(64, 3, activation='relu', padding='same')(m2)
    u3 = layers.UpSampling2D((2, 2))(d2)
    m3 = layers.concatenate([u3, c1])
    d3 = layers.Conv2D(32, 3, activation='relu', padding='same')(m3)

    outputs = layers.Conv2D(len(CLASS_NAMES), 1, activation='softmax')(d3)
    return models.Model(inputs, outputs)


class EKGAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.interpreter = EKGInterpreter(api_key="iRxkJ9thUkJAvCnSZzVa9cxu7keYhG3CRlNPnk6q")
        self.model = None
        self.is_running = False
        self.init_ui()
        self.load_model_file()

    def load_model_file(self):
        try:
            if os.path.exists(MODEL_PATH):
                self.model = build_transformer_unet()
                self.model.load_weights(MODEL_PATH)
                self.log("<font color='#2ecc71'><b>[SİSTEM] Model ve Güçlendirilmiş Otomatik Retry Katmanı Başarıyla Devreye Alındı.</b></font>")
        except Exception as e:
            self.log(f"<font color='#ff9f43'><b>[KRİTİK HATA] {str(e)}</b></font>")

    def init_ui(self):
        self.setWindowTitle("AI EKG - Klinik Karar Destek Paneli")
        self.setMinimumSize(1150, 950)
        self.setStyleSheet("""
            QMainWindow { background-color: #1e272e; }
            QLabel { color: #ecf0f1; }
            QPushButton { 
                background-color: #34495e; color: white; border-radius: 12px; 
                font-weight: bold; font-size: 16px; border: 1px solid #576574;
            }
            QPushButton:hover { background-color: #576574; border: 1px solid #3498db; }
            #stop_btn { background-color: #c0392b; border: 1px solid #a93226; }
            #exit_btn { background-color: #4b4b4b; border: 1px solid #333; margin-top: 5px; }
            #exit_btn:hover { background-color: #2f3640; border: 1px solid #e74c3c; }
            QProgressBar { border: 2px solid #34495e; border-radius: 5px; text-align: center; color: white; height: 25px; }
            QTextEdit { background-color: #2c3e50; color: #ffffff; border-radius: 10px; padding: 15px; font-size: 20px; }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        header = QLabel("EKG KLİNİK KARAR DESTEK SİSTEMİ")
        header.setFont(QFont('Segoe UI', 28, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)

        options_layout = QHBoxLayout()
        self.btn_folder = QPushButton("📂 KLASÖR ANALİZİ\n(PNG Dosyalarını Tara)")
        self.btn_folder.setFixedHeight(90)
        self.btn_folder.clicked.connect(self.start_folder_analysis)
        self.btn_csv = QPushButton("📊 CSV VERİ ANALİZİ\n(Görsele Çevir ve Tara)")
        self.btn_csv.setFixedHeight(90)
        self.btn_csv.clicked.connect(self.start_csv_analysis)
        options_layout.addWidget(self.btn_folder)
        options_layout.addWidget(self.btn_csv)
        main_layout.addLayout(options_layout)

        middle_layout = QHBoxLayout()
        self.report_area = QTextEdit()
        self.report_area.setReadOnly(True)
        middle_layout.addWidget(self.report_area, 2)

        # GÖRSEL ALANI (ORTALANMIŞ)
        img_container = QVBoxLayout()
        self.img_label_title = QLabel("ANALİZ GÖRSELİ")
        self.img_label_title.setFont(QFont('Segoe UI', 12, QFont.Bold))
        self.img_label_title.setAlignment(Qt.AlignCenter)
        img_container.addWidget(self.img_label_title)

        self.image_display = QLabel()
        self.image_display.setFixedSize(450, 450)
        self.image_display.setStyleSheet("border: 2px solid #3498db; background-color: #161d23; border-radius: 15px;")
        self.image_display.setAlignment(Qt.AlignCenter)
        img_container.addWidget(self.image_display)
        middle_layout.addLayout(img_container, 1)
        main_layout.addLayout(middle_layout)

        self.p_bar = QProgressBar()
        main_layout.addWidget(self.p_bar)

        self.btn_stop = QPushButton("🛑 ANALİZİ DURDUR")
        self.btn_stop.setObjectName("stop_btn")
        self.btn_stop.setFixedHeight(55)
        self.btn_stop.clicked.connect(self.stop_requested)
        main_layout.addWidget(self.btn_stop)

        self.btn_clear = QPushButton("🧹 EKRANI TEMİZLE")
        self.btn_clear.setFixedHeight(55)
        self.btn_clear.clicked.connect(self.clear_screen)
        self.btn_clear.hide()
        main_layout.addWidget(self.btn_clear)

        self.btn_exit = QPushButton("❌ SİSTEMDEN ÇIKIŞ YAP")
        self.btn_exit.setObjectName("exit_btn")
        self.btn_exit.setFixedHeight(55)
        self.btn_exit.clicked.connect(self.close)
        main_layout.addWidget(self.btn_exit)

    def log(self, text):
        self.report_area.append(f"<span>{text}</span>")

    def display_ekg_image(self, img_path):
        pixmap = QPixmap(img_path)
        scaled_pixmap = pixmap.scaled(self.image_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_display.setPixmap(scaled_pixmap)

    def clear_screen(self):
        self.report_area.clear()
        self.p_bar.setValue(0)
        self.image_display.clear()
        self.btn_clear.hide()

    def stop_requested(self):
        self.is_running = False
        self.log("<br><b style='color:#ff9f43;'>[UYARI] İşlem durduruldu.</b>")
        self.btn_clear.show()

    def start_folder_analysis(self):
        self.btn_clear.hide()
        folder = QFileDialog.getExistingDirectory(self, "Klasör Seç")
        if folder: self.run_process(folder)

    def start_csv_analysis(self):
        self.btn_clear.hide()

        # 📂 CSV Dosyası Seçme Penceresini Açıyoruz
        csv_file, _ = QFileDialog.getOpenFileName(self, "Analiz Edilecek EKG CSV Dosyasını Seçin", "",
                                                  "CSV Files (*.csv)")

        if not csv_file:
            self.log("<font color='#ff9f43'><b>[UYARI] CSV dosya seçimi iptal edildi.</b></font>")
            return

        self.log(
            f"<font color='#3498db'><b>[SİSTEM] {os.path.basename(csv_file)} yükleniyor ve ön işleme tabi tutuluyor...</b></font>")
        QApplication.processEvents()

        try:
            # 1. CSV Verisini Okuma
            df_sig = pd.read_csv(csv_file, skipinitialspace=True)
            df_sig.columns = [str(c).replace("'", "").strip() for c in df_sig.columns]

            # ⚙️ VERİ SETİ KODUYLA BİREBİR UYUMLU 30 DAKİKALIK SINIRLAMA AYARI
            max_samples_30_min = SAMPLING_RATE * 60 * 30
            if len(df_sig) > max_samples_30_min:
                df_sig = df_sig.iloc[:max_samples_30_min]
                self.log("<font color='#f1c40f'><b>[BİLGİ] Sinyal jüri optimizasyonu için ilk 30 dakika ile sınırlandırıldı.</b></font>")

            # 3. Geçici Klasör Oluşturma
            if os.path.exists(TEMP_DIR):
                shutil.rmtree(TEMP_DIR)
            os.makedirs(TEMP_DIR, exist_ok=True)

            # 4. Sinyali 1500'lük Pencerelere Bölme ve Veri Seti Koduna Göre EKG Çizdirme
            self.log(
                "<font color='#3498db'><b>[SİSTEM] Sinyal pencereleniyor ve eğitim şablonundaki eksen yerleşimlerine göre basılıyor...</b></font>")
            total_samples = len(df_sig)
            chunk_idx = 0

            dpi_val = 100
            fig_size_val = float(256) / float(dpi_val)

            for start_pt in range(0, total_samples - WINDOW_SIZE_SAMP, WINDOW_SIZE_SAMP):
                chunk = df_sig.iloc[start_pt:start_pt + WINDOW_SIZE_SAMP]

                # 🖼️ VERİ SETİ GÖRSELLEŞTİRME KODUNUN MATPLOTLIB MOTORU (Birebir Eşlendi)
                fig_img = plt.figure(figsize=(fig_size_val, fig_size_val), dpi=dpi_val)
                ax_img = fig_img.add_axes((0.0, 0.0, 1.0, 1.0))

                # Veri setindeki columns[1:3] mantığı ile Siyah ve Gri iki kanalı çizdirme
                for i, col in enumerate(chunk.columns[1:3]):
                    y_raw = chunk[col].values
                    y_filt = butter_bandpass_filter(y_raw)

                    y_min = np.min(y_filt)
                    y_max = np.max(y_filt)
                    y_norm = (y_filt - y_min) / (y_max - y_min + 1e-5)

                    # Üstte gri (i=1), altta siyah (i=0) olacak şekilde 1.3 dikey kaydırma formülü
                    ax_img.plot(y_norm + (i * 1.3), color='black' if i == 0 else 'gray', linewidth=1.0)

                ax_img.set_xlim(0, WINDOW_SIZE_SAMP)
                ax_img.set_ylim(-0.2, 2.8)
                ax_img.axis('off')

                img_path = os.path.join(TEMP_DIR, f"chunk_{chunk_idx}.png")
                fig_img.savefig(img_path)
                plt.close(fig_img)
                chunk_idx += 1

            self.log(
                f"<font color='#2ecc71'><b>✓ {chunk_idx} adet iki kanallı EKG sinyal penceresi eğitim formatında başarıyla üretildi.</b></font>")

            # 5. Üretilen Görselleri Modelin Batch Fonksiyonuna Gönderme
            self.run_process(TEMP_DIR)

            # 🧹 OTOMATİK SİLME MOTORU: Analiz bittiğinde üretilen görselleri siler
            if os.path.exists(TEMP_DIR):
                shutil.rmtree(TEMP_DIR)
                self.log("<font color='#2ecc71'><b>✓ [MÜHENDİSLİK] Bellek Optimizasyonu: Geçici şerit resimleri sistemden otomatik silindi.</b></font>")

        except Exception as e:
            self.log(f"<font color='#c0392b'><b>[HATA] CSV işleme esnasında hata oluştu: {str(e)}</b></font>")
            self.btn_clear.show()
            if os.path.exists(TEMP_DIR):
                shutil.rmtree(TEMP_DIR)

    def run_process(self, folder):
        self.is_running = True
        self.report_area.clear()
        self.process_batch(folder)

    def process_batch(self, folder):
        files = [f for f in glob.glob(os.path.join(folder, "*.png")) if "_aug_" not in f]
        findings = {name: 0 for name in CLASS_NAMES if name != "Arkaplan"}
        all_rr_seconds = []
        total = len(files)

        for i in range(total):
            if not self.is_running: break
            p = files[i]
            if i % 2 == 0: self.display_ekg_image(p)
            img_bgr = cv2.imread(p)
            img_res = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
            gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
            _, black_channel = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            dual_channel = np.zeros((IMG_SIZE, IMG_SIZE, 2), dtype=np.float32)
            dual_channel[..., 0] = black_channel / 255.0
            dual_channel[..., 1] = gray / 255.0

            pred = self.model.predict(np.expand_dims(dual_channel, axis=0), verbose=0)
            mask = np.argmax(pred[0], axis=-1)

            beat_mask = (mask >= 1).astype(np.uint8)
            peaks, _ = find_peaks(np.sum(beat_mask, axis=0), height=2, distance=15)
            if len(peaks) > 1:
                sec_per_pixel = (WINDOW_SIZE_SAMP / SAMPLING_RATE) / IMG_SIZE
                all_rr_seconds.extend(np.diff(peaks) * sec_per_pixel)

            unique, counts = np.unique(mask, return_counts=True)
            stats = dict(zip(unique, counts))
            anormal = {k: v for k, v in stats.items() if k >= 2 and v > 5}
            if anormal:
                best_cls = max(anormal, key=anormal.get)
                findings[CLASS_NAMES[best_cls]] += 1
            elif 1 in stats and stats[1] > 8:
                findings[CLASS_NAMES[1]] += 1

            self.p_bar.setValue(int(((i + 1) / total) * 100))
            QApplication.processEvents()

        self.btn_clear.show()
        self.show_report(findings, all_rr_seconds)

    def show_report(self, findings, all_rr_seconds):
        self.log("<br><b style='color: #ffffff;'>" + "=" * 32 + "</b>")
        self.log("<center><b style='font-size: 26px; color: #00d2d3;'>AI EKG ANALİZ RAPORU</b></center>")
        self.log("<b style='color: #ffffff;'>" + "=" * 32 + "</b>")

        patolojik_bulgular = {k: v for k, v in findings.items() if k not in ["Normal (N)", "Arkaplan"]}
        anormal_total = sum(patolojik_bulgular.values())
        baskin_tip = max(findings, key=findings.get)

        bpm = int(60 / np.mean(all_rr_seconds)) if all_rr_seconds else 0
        std_rr = np.std(all_rr_seconds) if all_rr_seconds else 0
        ritim = "DÜZENLİ (Sinüzal)" if std_rr < 0.08 else "DÜZENSİZ (Aritmik)"

        self.log(f"<b>📊 ÖLÇÜLEN NABIZ: <span style='color:#3498db;'>{bpm} BPM</span></b>")
        self.log(f"<b>💓 RİTİM DURUMU: <span style='color:#9b59b6;'>{ritim}</span></b>")
        self.log("<b style='color: #ffffff;'>" + "-" * 32 + "</b>")
        self.log("<b>BULGU DAĞILIMI:</b>")
        for label, count in findings.items():
            if count > 0:
                color = "#2ecc71" if "Normal" in label else "#ff9f43"
                self.log(f"<span style='color:{color};'>- {label}: {count} kayıt</span>")
        self.log("<b style='color: #ffffff;'>" + "-" * 32 + "</b>")

        if anormal_total > 5:
            self.log("<h2 style='color:#ff9f43;'>SONUÇ: ARİTMİ TESPİT EDİLDİ</h2>")
            self.log(f"<b style='color:#00d2d3; font-size: 22px;'>BASKIN SONUÇ: {baskin_tip}</b>")
        else:
            self.log("<h2 style='color:#2ecc71;'>SONUÇ: ARİTMİ TESPİT EDİLMEDİ (NORMAL RİTİM)</h2>")
            self.log(f"<b style='color:#2ecc71; font-size: 22px;'>BASKIN SONUÇ: {baskin_tip}</b>")

        self.log("<br><b style='color: #f1c40f;'>🤖 YAPAY ZEKA KLİNİK YORUMU (LLM):</b>")
        self.log("<i style='color: #bdc3c7;'>Yorum hazırlanıyor...</i>")
        QApplication.processEvents()

        # Saf LLM yorumlama katmanını çalıştırıyoruz
        interpretation = self.interpreter.get_interpretation_pure_llm(findings, baskin_tip, bpm, ritim)
        self.log(
            f"<div style='color: #ecf0f1; background-color: #34495e; padding: 15px; border-radius: 10px;'>{interpretation}</div>")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EKGAnalyzerApp()
    window.show()
    sys.exit(app.exec_())