"""
FILE: digitalization.py
TUJUAN: Modul untuk operasi digitalisasi khususnya manipulasi level warna (tonal) dan pencahayaan.

LANGKAH-LANGKAH UTAMA:
1. apply_quantization: Melakukan pengurangan kedalaman level warna (kuantisasi) pada citra menggunakan operasi matematika pembagian dan perkalian lantai.
2. apply_clahe (jika ada): Menggunakan algoritma Contrast Limited Adaptive Histogram Equalization untuk meratakan pencahayaan gambar.
"""
import cv2
import numpy as np

def apply_quantization(image, levels):
    """
    Fungsi operasi digitalisasi: Kuantisasi Warna (menurunkan kedalaman warna/level).
    Mengubah nilai pixel kontinu menjadi sejumlah 'levels' nilai diskrit.
    """
    if levels >= 256:
        return image
        
    # Faktor pembagi
    factor = 256 / levels
    
    # Lakukan kuantisasi
    quantized = np.uint8(np.floor(image / factor) * factor)
    
    return quantized

def auto_enhance(image):
    """
    Meningkatkan kontras lokal menggunakan CLAHE secara otomatis untuk
    mempertajam batas tepi objek (bulat/kapsul) agar deteksi tepi lebih tangguh
    tanpa merusak warna asli secara berlebihan.
    """
    if len(image.shape) == 3:
        # Konversi ke ruang warna LAB untuk memisahkan intensitas cahaya (L)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Terapkan CLAHE pada L-channel
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        
        # Gabungkan kembali ke BGR
        merged = cv2.merge((cl, a_channel, b_channel))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        return enhanced
    else:
        # Jika gambar sudah grayscale
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        return clahe.apply(image)
