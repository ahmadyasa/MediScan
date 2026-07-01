import cv2

def resize_image(image, target_width=800):
    """Fungsi operasi geometri untuk mengubah ukuran (scaling) gambar berdasarkan lebar tertentu"""
    height, width = image.shape[:2]
    new_height = int((target_width / width) * height)
    resized = cv2.resize(image, (target_width, new_height))
    return resized

def scale_image(image, factor):
    """Memperbesar atau memperkecil gambar berdasarkan faktor pengali"""
    if factor == 1.0: 
        return image
    h, w = image.shape[:2]
    return cv2.resize(image, (int(w * factor), int(h * factor)))

def rotate_image(image, angle):
    """Merotasi gambar sejauh 'angle' derajat"""
    if angle == 0: 
        return image
    h, w = image.shape[:2]
    # Titik pusat rotasi
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Gunakan warpAffine untuk merotasi
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

def negate_image(image):
    """Melakukan negasi warna (membalikkan nilai intensitas)"""
    return cv2.bitwise_not(image)

def classify_pill_shape(contour):
    """
    Logika klasifikasi bentuk pil berdasarkan Bounding Box Aspect Ratio.
    """
    # 1. Ambil data Bounding Box dari kontur
    x, y, w, h = cv2.boundingRect(contour)
    
    # Proteksi tambahan untuk mencegah error zero-division jika noise
    if h == 0:
        return "Tidak Diketahui", (x, y, w, h)
        
    # 2. Hitung aspect_ratio = w / h
    aspect_ratio = float(w) / h
    
    # 3. Buat kondisi klasifikasi berdasarkan rasio
    # KALIBRASI: Batas toleransi untuk Pil Bulat diperlebar (0.70 - 1.35) 
    # untuk mengkompensasi perubahan bentuk/distorsi perspektif kamera jika pil tidak tegak lurus (AR anomalies)
    if 0.70 <= aspect_ratio <= 1.35:
        label = "Pil Bulat"
    elif aspect_ratio < 0.70 or aspect_ratio > 1.35:
        label = "Kapsul"
    else:
        label = "Tidak Diketahui"
        
    return label, (x, y, w, h)