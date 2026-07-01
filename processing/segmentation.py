import cv2
import numpy as np
from processing.edge_detection import separate_and_find_contours
from processing.geometry import classify_pill_shape

def create_robust_mask(edges, kernel_size_close=5):
    """
    Membuat mask padat dari garis tepi dengan membersihkan noise 
    dan mengisi bagian dalam objek (fill contours).
    """
    # 1. Sambungkan garis-garis tepi yang terputus (Closing)
    # KALIBRASI: Menggunakan kernel_size_close yang lebih kecil (misal 5x5) 
    # agar garis noise dari kemasan foil tidak menyambung menjadi gumpalan raksasa.
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_close, kernel_size_close))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
    
    # 2. Isi bagian dalam bentuk (Fill Contours) agar menjadi mask putih padat (solid)
    mask = np.zeros(closed_edges.shape, dtype=np.uint8)
    
    # Gunakan RETR_LIST agar semua kontur (baik dalam maupun luar) terdeteksi
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Batas ukuran area maksimum untuk membuang bingkai luar gambar / kemasan raksasa
    img_area = closed_edges.shape[0] * closed_edges.shape[1]
    
    for contour in contours:
        area = cv2.contourArea(contour)
        # Abaikan jika kontur terlalu besar (misal lebih dari 15% layar, ini pasti noise foil/teks raksasa)
        # Abaikan juga kontur yang terlalu kecil
        if area < 0.15 * img_area and area > 100:
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            
    return mask

def segment_edge_based(edge_image, original_image, kernel_size_close=5):
    """
    Melakukan segmentasi berbasis tepi dan mengklasifikasikan bentuk
    berdasarkan rasio geometri (bounding box).
    Mengembalikan: citra_berlabel, mask_proses, jumlah_bulat, jumlah_kapsul, daftar_koordinat
    """
    # 1. Dapatkan mask solid dari keseluruhan objek pil berdasarkan deteksi tepi terpilih
    mask = create_robust_mask(edge_image, kernel_size_close)

    # 2. Pisahkan pil yang menempel menggunakan algoritma Watershed
    contours, processed_mask = separate_and_find_contours(mask, original_image)

    img_segmented = original_image.copy()
    
    img_area = original_image.shape[0] * original_image.shape[1]
    # KALIBRASI: Filter Area dinamis (0.05% dari ukuran gambar)
    # Ini menjamin noise kecil hilang, tetapi pil asli tidak terbuang
    # meskipun gambar beresolusi 4K atau 480p.
    min_area_threshold = img_area * 0.0005
    
    rects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area_threshold: 
            jenis, (x, y, w, h) = classify_pill_shape(contour)
            if jenis != "Tidak Diketahui":
                rects.append([x, y, w, h, jenis])

    # 3. Hapus Duplikat Bounding Box (Non-Maximum Suppression) jika ada tumpang tindih
    final_rects = []
    for r in rects:
        x, y, w, h, jenis = r
        is_duplicate = False
        for fr in final_rects:
            fx, fy, fw, fh, fjenis = fr
            intersect_w = max(0, min(x+w, fx+fw) - max(x, fx))
            intersect_h = max(0, min(y+h, fy+fh) - max(y, fy))
            intersect_area = intersect_w * intersect_h
            
            # Jika area tumpang tindih lebih dari 30%, anggap duplikat
            if intersect_area > 0.3 * min(w*h, fw*fh):
                is_duplicate = True
                break
        
        if not is_duplicate:
            final_rects.append(r)

    # 4. Menggambar bounding box beserta label teks di gambar hasil
    for (x, y, w, h, jenis) in final_rects:
        color = (0, 255, 0) if jenis == "Pil Bulat" else (0, 200, 255)
        cv2.rectangle(img_segmented, (x, y), (x+w, y+h), color, 2)
        
        # Desain label teks agar rapi
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(jenis, font, font_scale, thickness)
        
        y_text = y if (y - text_h - 6) > 0 else (y + h + text_h + 6)
        
        # Kotak background untuk teks agar mudah dibaca
        if y_text == y:
            cv2.rectangle(img_segmented, (x, y_text - text_h - 6), (x + text_w + 4, y_text), color, -1)
            cv2.putText(img_segmented, jenis, (x + 2, y_text - 3), font, font_scale, (0, 0, 0), thickness)
        else:
            cv2.rectangle(img_segmented, (x, y_text - text_h - 4), (x + text_w + 4, y_text + 4), color, -1)
            cv2.putText(img_segmented, jenis, (x + 2, y_text + 2), font, font_scale, (0, 0, 0), thickness)
            
    # Hitung jumlah masing-masing kategori
    count_bulat = sum(1 for r in final_rects if r[4] == "Pil Bulat")
    count_kapsul = sum(1 for r in final_rects if r[4] == "Kapsul")

    return img_segmented, processed_mask, count_bulat, count_kapsul, final_rects