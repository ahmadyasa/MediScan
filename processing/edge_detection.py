import cv2
import numpy as np

def detect_edges_sobel(gray_image):
    """Deteksi tepi menggunakan metode Sobel"""
    # Menggunakan Bilateral Filter untuk melicinkan tekstur kemasan foil 
    # tanpa mengaburkan ketajaman batas pinggiran pil.
    blur = cv2.bilateralFilter(gray_image, 9, 75, 75)
    
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = cv2.magnitude(sobelx, sobely)
    sobel_combined = cv2.convertScaleAbs(magnitude)
    return sobel_combined

def detect_edges_prewitt(gray_image):
    """Deteksi tepi menggunakan metode Prewitt"""
    blur = cv2.bilateralFilter(gray_image, 9, 75, 75)
    
    # Kernel Prewitt
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=float)
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float)
    
    prewittx = cv2.filter2D(blur, cv2.CV_64F, kernelx)
    prewitty = cv2.filter2D(blur, cv2.CV_64F, kernely)
    
    magnitude = cv2.magnitude(prewittx, prewitty)
    return cv2.convertScaleAbs(magnitude)

def detect_edges_laplacian(gray_image):
    """Deteksi tepi menggunakan metode Laplacian"""
    blur = cv2.bilateralFilter(gray_image, 9, 75, 75)
    laplacian = cv2.Laplacian(blur, cv2.CV_64F)
    return cv2.convertScaleAbs(laplacian)

def separate_and_find_contours(mask, original_image):
    """
    Pemisahan objek (pil) yang menumpuk menggunakan Algoritma Watershed
    berbasis Transformasi Jarak (Distance Transform). Sangat akurat untuk
    mengiris pil yang menempel rapat layaknya kepingan puzzle.
    """
    # 1. Bersihkan noise kecil
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_clean, iterations=2)
    
    # 2. Cari 'Sure Background' (area yang sudah pasti latar belakang)
    sure_bg = cv2.dilate(opening, kernel_clean, iterations=3)
    
    # 3. Cari 'Sure Foreground' (pusat pasti setiap pil) menggunakan Erosi Kuat
    # Kita menggunakan Erosi (bukan sekadar puncak Jarak) agar kapsul yang panjang 
    # tidak terpotong menjadi 2 titik puncak yang mengakibatkan over-segmentation.
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    sure_fg = cv2.erode(opening, kernel_erode, iterations=5)
    sure_fg = np.uint8(sure_fg)
    
    # 4. Tentukan area perbatasan yang belum jelas (Unknown Region)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # 5. Labeli setiap titik inti pil dengan ID unik (Connected Components)
    ret, markers = cv2.connectedComponents(sure_fg)
    
    # Beri label 1 pada background (bukan 0, agar Watershed bisa membedakan area luar)
    markers = markers + 1
    # Tandai area Unknown dengan 0
    markers[unknown == 255] = 0
    
    # 6. Aplikasikan Algoritma Watershed
    if len(original_image.shape) == 2:
        img_color = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        img_color = original_image.copy()
        
    markers = cv2.watershed(img_color, markers)
    
    final_contours = []
    separated_mask = np.zeros_like(mask)
    
    # 7. Ekstrak kontur untuk masing-masing pil (ID mulai dari 2, karena 1 adalah background)
    for label in range(2, ret + 1):
        obj_mask = np.zeros_like(mask, dtype=np.uint8)
        obj_mask[markers == label] = 255
        
        # Restorasi ringan karena Watershed mengiris 1 piksel sebagai garis batas antar pil (-1)
        obj_mask = cv2.dilate(obj_mask, kernel_clean, iterations=1)
        
        separated_mask = cv2.bitwise_or(separated_mask, obj_mask)
        
        cnts, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            final_contours.append(c)
            
    return final_contours, separated_mask