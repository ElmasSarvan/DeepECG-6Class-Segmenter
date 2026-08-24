import sys
import os
import glob
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from tensorflow.keras import layers, models, Input, backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# --- 1. AYARLAR ---
IMG_SIZE = 128
NUM_CLASSES = 7
CHANNELS = 2  # 1. Kanal: Siyah Sinyal (Önemli), 2. Kanal: Gri Ton
BATCH_SIZE = 8
EPOCHS = 15
dataset_path = r'C:\Users\elmas\PyCharmMiscProject\ekg_projesi\final_veriseti_v4'
model_save_path = 'ekg_transformer_unet_v3_final.h5'

CLASS_NAMES = ["Arkaplan", "Normal", "LBBB", "RBBB", "PVC", "APC", "Paced"]
CLASS_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype="float32")


# --- 2. METRİKLER VE KAYIP FONKSİYONU ---
def dice_coefficient(y_true, y_pred):
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=len(CLASS_NAMES))[..., 1:])
    y_pred_f = K.flatten(y_pred[..., 1:])
    return (2. * K.sum(y_true_f * y_pred_f) + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def jaccard_index(y_true, y_pred):
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=len(CLASS_NAMES))[..., 1:])
    y_pred_f = K.flatten(y_pred[..., 1:])
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def weighted_categorical_crossentropy(weights):
    def loss(y_true, y_pred):
        scce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        weight_map = tf.gather(weights, tf.cast(y_true, tf.int32))
        return scce * tf.squeeze(weight_map, axis=-1)

    return loss


# --- 3. TRANSFORMER BLOĞU ---
def transformer_block(inputs, num_heads, embed_dim, ff_dim, dropout=0.1):
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    attention_output = layers.Dropout(dropout)(attention_output)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    ffn_output = layers.Dense(ff_dim, activation="relu")(out1)
    ffn_output = layers.Dense(embed_dim)(ffn_output)
    ffn_output = layers.Dropout(dropout)(ffn_output)
    return layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)


# --- 4. DATA GENERATOR (Siyah Kanal Öncelikli) ---
class EKGDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, img_paths, mask_paths, batch_size=8, img_size=128):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.img_size = img_size

    def __len__(self):
        return int(np.ceil(len(self.img_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_img_paths = self.img_paths[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_mask_paths = self.mask_paths[idx * self.batch_size: (idx + 1) * self.batch_size]
        x, y = [], []

        for img_p, msk_p in zip(batch_img_paths, batch_mask_paths):
            img_bgr = cv2.imread(img_p)
            img_res = cv2.resize(img_bgr, (self.img_size, self.img_size))
            gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)

            # Siyah kanalı (Önemli Kanal) - Eşikleme ile sinyali belirginleştir
            _, black_channel = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

            dual_channel = np.zeros((self.img_size, self.img_size, 2), dtype=np.float32)
            dual_channel[..., 0] = black_channel / 255.0  # 1. KANAL: Siyah Sinyal (ÖNCELİKLİ)
            dual_channel[..., 1] = gray / 255.0  # 2. KANAL: Gri Ton

            mask_raw = cv2.imread(msk_p, cv2.IMREAD_GRAYSCALE)
            mask_raw = cv2.resize(mask_raw, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            final_mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            final_mask[(mask_raw > 45) & (mask_raw < 56)] = 1
            final_mask[(mask_raw > 97) & (mask_raw < 107)] = 2
            final_mask[(mask_raw > 70) & (mask_raw < 82)] = 3
            final_mask[(mask_raw > 148) & (mask_raw < 158)] = 4
            final_mask[(mask_raw > 199) & (mask_raw < 210)] = 5
            final_mask[mask_raw > 250] = 6

            x.append(dual_channel)
            y.append(np.expand_dims(final_mask, axis=-1))

        return np.array(x), np.array(y)


# --- 5. VERİ HAZIRLAMA (AUG VERİLERİ DAHİL) ---
all_images = sorted(glob.glob(os.path.join(dataset_path, "images", "*.png")))
all_masks = sorted(glob.glob(os.path.join(dataset_path, "masks", "*.png")))

train_img, val_img, train_msk, val_msk = train_test_split(all_images, all_masks, test_size=0.15, random_state=42)

train_gen = EKGDataGenerator(train_img, train_msk, batch_size=BATCH_SIZE, img_size=IMG_SIZE)
val_gen = EKGDataGenerator(val_img, val_msk, batch_size=BATCH_SIZE, img_size=IMG_SIZE)


# --- 6. TRANSFORMER-UNET MODELİ ---
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


model = build_transformer_unet()
model.compile(optimizer='adam', loss=weighted_categorical_crossentropy(CLASS_WEIGHTS),
              metrics=['accuracy', dice_coefficient, jaccard_index])

# --- 7. EĞİTİM ---
callbacks = [
    ModelCheckpoint(model_save_path, monitor='val_dice_coefficient', save_best_only=True, mode='max'),
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
]
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)


# --- 8. ANALİZ ---
def perform_full_analysis(generator, model, history):
    # Metrik Grafikleri
    plt.figure(figsize=(20, 5))
    metrics_list = ['loss', 'accuracy', 'dice_coefficient', 'jaccard_index']
    for i, m in enumerate(metrics_list):
        plt.subplot(1, 4, i + 1)
        plt.plot(history.history[m], label='Tr')
        plt.plot(history.history['val_' + m], label='Val')
        plt.title(m.replace('_', ' ').title());
        plt.legend()
    plt.show()

    # Tahmin Toplama
    all_y_true, all_y_pred_probs = [], []
    for i in range(min(len(generator), 40)):
        x_val, y_val = generator[i]
        preds = model.predict(x_val, verbose=0)
        all_y_true.extend(y_val.flatten())
        all_y_pred_probs.extend(preds.reshape(-1, len(CLASS_NAMES)))

    all_y_true, all_y_pred_probs = np.array(all_y_true), np.array(all_y_pred_probs)
    all_y_pred = np.argmax(all_y_pred_probs, axis=-1)

    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(all_y_true, all_y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES);
    plt.show()

    # ROC Eğrisi
    plt.figure(figsize=(12, 8))
    y_true_oh = tf.keras.utils.to_categorical(all_y_true, num_classes=len(CLASS_NAMES))
    for i in range(len(CLASS_NAMES)):
        fpr, tpr, _ = roc_curve(y_true_oh[:, i], all_y_pred_probs[:, i])
        plt.plot(fpr, tpr, label=f'ROC {CLASS_NAMES[i]} (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--');
    plt.title('Multi-Class ROC Curve');
    plt.legend();
    plt.show()

    print("\n--- CLASSIFICATION REPORT ---\n", classification_report(all_y_true, all_y_pred, target_names=CLASS_NAMES))

    # Örnek Tahmin (X, Y, Z Görselleştirme)
    x_test, y_test = generator[0]
    z_test = np.argmax(model.predict(x_test, verbose=0), axis=-1)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1);
    plt.imshow(x_test[0][..., 0], cmap='gray');
    plt.title("X: Siyah Sinyal Kanalı")
    plt.subplot(1, 3, 2);
    plt.imshow(y_test[0].squeeze(), cmap='jet');
    plt.title("Y: Gerçek Maske")
    plt.subplot(1, 3, 3);
    plt.imshow(z_test[0], cmap='jet');
    plt.title("Z: Transformer-UNet Tahmini")
    plt.show()


perform_full_analysis(val_gen, model, history)