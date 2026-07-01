import cv2
import numpy as np
def convert_to_grayscale(image_bgr):
    """Konversi citra berwarna ke Grayscale"""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
def rgb_to_grayscale(image_rgb):
    """Konversi citra RGB ke Grayscale (alias untuk UI baru)"""
    if len(image_rgb.shape) == 3:
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return image_rgb

def convert_to_rgb(image_bgr):
    """Konversi format BGR (OpenCV) ke RGB (untuk Web)"""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

def get_image_info(name, img):
    """Mendapatkan metadata gambar dasar"""
    height, width = img.shape[:2]
    channels = img.shape[2] if len(img.shape) == 3 else 1
    return {
        "width": width,
        "height": height,
        "total_pixels": width * height,
        "channels": channels,
        "depth": img.dtype.itemsize * 8
    }

def extract_pixel_matrix(gray_img, cx, cy, size=7):
    """Ekstraksi neighborhood pixel di titik tertentu"""
    half = size // 2
    
    # Batasi koordinat agar tidak keluar dari dimensi gambar
    y1 = max(0, cy - half)
    y2 = min(gray_img.shape[0], cy + half + 1)
    x1 = max(0, cx - half)
    x2 = min(gray_img.shape[1], cx + half + 1)
    
    matrix = gray_img[y1:y2, x1:x2]
    
    # Pad dengan 0 jika potongannya berada di pinggir (ukurannya kurang dari 'size')
    if matrix.shape[0] != size or matrix.shape[1] != size:
        padded = np.zeros((size, size), dtype=np.uint8)
        padded[:matrix.shape[0], :matrix.shape[1]] = matrix
        return padded.tolist()
        
    return matrix.tolist()