import streamlit as st
import cv2
import numpy as np
import os
import sys
import json
import pandas as pd

# Menambahkan direktori parent ke PATH untuk mendeteksi modul lokal 'processing'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from processing.representation import get_image_info, rgb_to_grayscale, extract_pixel_matrix, convert_to_rgb
from processing.digitalization import apply_quantization
from processing.geometry import scale_image, negate_image, rotate_image, resize_image
from processing.segmentation import segment_edge_based, create_robust_mask
from processing.edge_detection import detect_edges_sobel, detect_edges_prewitt, detect_edges_laplacian

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="MediScan - Analisis Obat",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS CUSTOM UNTUK DESAIN MODERN & GLASSMORPHISM ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }
    
    .stApp {
        background-color: #0B1120;
        color: #e2e8f0;
    }
    
    /* Modern Header */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 2.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-left: 5px solid #00E676;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(circle at top right, rgba(0, 230, 118, 0.1), transparent 50%);
        pointer-events: none;
    }

    .main-header h1 {
        color: #f8fafc;
        margin: 0;
        font-size: 2.5rem !important;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #94a3b8;
        margin: 0.8rem 0 0 0;
        font-size: 1.15rem;
        font-weight: 300;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1e293b;
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        border-radius: 8px;
        color: #94a3b8;
        border: none;
        padding: 0 20px;
        background-color: transparent;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #334155 !important;
        color: #00E676 !important;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Image container */
    img {
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
        max-width: 100%;
        object-fit: contain;
    }
    img:hover {
        transform: scale(1.01);
    }
    
    /* Metrics glassmorphism */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #00E676;
    }
    [data-testid="stMetricValue"] {
        color: #00E676 !important;
        font-weight: 700;
    }
    
    /* Camera fix */
    [data-testid="stCameraInput"] video, [data-testid="stCameraInput"] canvas, [data-testid="stCameraInput"] img {
        transform: scaleX(-1) !important;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# --- JUDUL HALAMAN ---
st.markdown("""
<div class="main-header">
    <h1>💊 MediScan: Analisis Pengolahan Citra</h1>
    <p>Sistem cerdas evaluasi piksel, transformasi geometri, dan deteksi morfologi obat (Sesuai Syarat UAS Pengolahan Citra Digital).</p>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR: KONTROL INPUT ---
with st.sidebar:
    # Header bergaya modern
    st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h1 style="font-size: 2.5rem; margin-bottom: 0;">🔬</h1>
            <h2 style="margin-top: 0; font-weight: 800; background: -webkit-linear-gradient(#00E676, #00B0FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">MediScan</h2>
            <p style="font-size: 0.8rem; color: #888; font-weight: 600; letter-spacing: 1px;">VISION ENGINE 2.0</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📥 Sumber Gambar")
    input_method = st.radio("Pilih metode pengambilan:", ["Unggah File", "Kamera Web"], horizontal=True)
    
    image_data = None
    
    if input_method == "Unggah File":
        st.markdown("<div style='font-size:0.8rem; color:#aaa; margin-bottom: 5px;'>Mendukung resolusi tinggi (FHD/4K)</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Muat Gambar (JPG, PNG, BMP)", type=['png', 'jpg', 'jpeg', 'bmp', 'webp'], label_visibility="collapsed")
        if uploaded_file is not None:
            image_data = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_data = cv2.imdecode(image_data, 1)
            st.success("✅ Gambar berhasil dimuat!")
            
    elif input_method == "Kamera Web":
        st.info("💡 **Tips:** Posisikan objek di tengah dengan pencahayaan terang.")
        camera_file = st.camera_input("Ambil Foto")
        if camera_file is not None:
            image_data = np.asarray(bytearray(camera_file.read()), dtype=np.uint8)
            image_data = cv2.imdecode(image_data, 1)
            image_data = cv2.flip(image_data, 1) # Flip agar hasil tangkapan sama dengan preview layar
            st.success("✅ Tangkapan layar sukses!")

    st.markdown("<br><hr style='border:1px dashed #444;'>", unsafe_allow_html=True)
    
    # Kotak Info Modern
    st.info("👨‍💻 **Sistem Progresif Aktif**\n\nPerubahan di satu Tab akan otomatis memengaruhi tahap selanjutnya secara langsung (*Real-time*).")
    
    st.markdown("""
        <div style="background: rgba(0, 230, 118, 0.1); border-left: 4px solid #00E676; padding: 10px; border-radius: 4px; margin-top: 10px;">
            <p style="margin: 0; font-size: 0.85rem; color: #ddd;">
            🎯 <b>Target:</b> Ujian Akhir Semester<br>
            📚 <b>Mata Kuliah:</b> Pengolahan Citra Digital
            </p>
        </div>
    """, unsafe_allow_html=True)

# --- MAIN PIPELINE ---
if image_data is not None:
    # PRE-PROCESSING DASAR (Resize jika gambar terlalu besar)
    h_orig, w_orig = image_data.shape[:2]
    max_dim = 800
    if max(h_orig, w_orig) > max_dim:
        scale_factor = max_dim / max(h_orig, w_orig)
        original_img = cv2.resize(image_data, (int(w_orig * scale_factor), int(h_orig * scale_factor)), interpolation=cv2.INTER_AREA)
    else:
        original_img = image_data.copy()
        
    original_rgb = convert_to_rgb(original_img)

    # --- MEMBUAT TABS UNTUK AKSESIBILITAS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 1. Representasi", 
        "🎛️ 2. Geometri & Aritmatika", 
        "🔪 3. Deteksi Tepi", 
        "💊 4. Hasil Segmentasi"
    ])

    # ==========================================
    # TAHAP 1: REPRESENTASI CITRA
    # ==========================================
    with tab1:
        st.subheader("Representasi Citra Dasar")
        st.write("Menganalisis anatomi citra murni dari perspektif data piksel komputer (Resolusi, Kedalaman, Matriks, & Distribusi).")
        
        info = get_image_info("image", original_rgb)
        
        col_b1, col_b2 = st.columns([1, 1])
        with col_b1:
            st.markdown("##### 📏 Informasi Metrik Citra")
            st.info(f"**Resolusi (W x H):** {info['width']} x {info['height']} pixel\n\n"
                    f"**Total Piksel:** {info['total_pixels']:,} titik\n\n"
                    f"**Saluran Warna:** {info['channels']} Channel (RGB)\n\n"
                    f"**Kedalaman Data:** {info.get('depth', 24)}-bit")
            
            st.markdown("##### 🧮 Matriks Piksel Inti (Tengah Citra)")
            gray_for_matrix = rgb_to_grayscale(original_rgb)
            matrix_data = extract_pixel_matrix(gray_for_matrix, info['width']//2, info['height']//2, size=7)
            
            if matrix_data:
                st.caption("*(Heat Map: Semakin terang warna, nilai piksel semakin mendekati 255)*")
                df = pd.DataFrame(matrix_data)
                styled_df = df.style.background_gradient(cmap='plasma', axis=None, vmin=0, vmax=255)
                st.dataframe(styled_df, use_container_width=True)
            
        with col_b2:
            st.markdown("##### 📈 Histogram Distribusi (Grayscale)")
            gray_hist = rgb_to_grayscale(original_rgb)
            hist_values, _ = np.histogram(gray_hist.flatten(), bins=256, range=[0, 256])
            
            hist_df = pd.DataFrame(hist_values, columns=['Frekuensi Piksel'])
            st.area_chart(hist_df, color="#00E676")
            
            st.markdown("##### 🧭 Tetangga Piksel (N4 Neighborhood)")
            if matrix_data and len(matrix_data) >= 7:
                cx, cy = info['width'] // 2, info['height'] // 2
                val_t = matrix_data[2][3]; val_l = matrix_data[3][2]; val_c = matrix_data[3][3]; val_r = matrix_data[3][4]; val_b = matrix_data[4][3]
                
                html_code = f"""
                <div style="background: rgba(30, 41, 59, 0.5); padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); display: flex; justify-content: center; margin-bottom: 20px;">
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px; width: 100%; max-width: 280px;">
                        <div style="opacity: 0;"></div>
                        <div style="background: #064e3b; color: #a7f3d0; padding: 10px; border-radius: 8px; text-align: center; box-shadow: inset 0 0 10px rgba(0,0,0,0.2);">
                            <div style="font-size: 11px; opacity: 0.8;">Atas ({cx},{cy-1})</div>
                            <div style="font-size: 16px; font-weight: bold; color: white; margin-top: 4px;">{val_t}</div>
                        </div>
                        <div style="opacity: 0;"></div>
                        
                        <div style="background: #064e3b; color: #a7f3d0; padding: 10px; border-radius: 8px; text-align: center; box-shadow: inset 0 0 10px rgba(0,0,0,0.2);">
                            <div style="font-size: 11px; opacity: 0.8;">Kiri ({cx-1},{cy})</div>
                            <div style="font-size: 16px; font-weight: bold; color: white; margin-top: 4px;">{val_l}</div>
                        </div>
                        <div style="background: linear-gradient(135deg, #00E676 0%, #00B259 100%); color: #064e3b; padding: 10px; border-radius: 8px; text-align: center; box-shadow: 0 4px 15px rgba(0,230,118,0.3);">
                            <div style="font-size: 11px; font-weight: bold; opacity: 0.9;">Pusat ({cx},{cy})</div>
                            <div style="font-size: 18px; font-weight: 900; margin-top: 4px;">{val_c}</div>
                        </div>
                        <div style="background: #064e3b; color: #a7f3d0; padding: 10px; border-radius: 8px; text-align: center; box-shadow: inset 0 0 10px rgba(0,0,0,0.2);">
                            <div style="font-size: 11px; opacity: 0.8;">Kanan ({cx+1},{cy})</div>
                            <div style="font-size: 16px; font-weight: bold; color: white; margin-top: 4px;">{val_r}</div>
                        </div>
                        
                        <div style="opacity: 0;"></div>
                        <div style="background: #064e3b; color: #a7f3d0; padding: 10px; border-radius: 8px; text-align: center; box-shadow: inset 0 0 10px rgba(0,0,0,0.2);">
                            <div style="font-size: 11px; opacity: 0.8;">Bawah ({cx},{cy+1})</div>
                            <div style="font-size: 16px; font-weight: bold; color: white; margin-top: 4px;">{val_b}</div>
                        </div>
                        <div style="opacity: 0;"></div>
                    </div>
                </div>
                """
                import streamlit.components.v1 as components
                components.html(html_code, height=240, scrolling=False)

    # ==========================================
    # TAHAP 2: GEOMETRI & ARITMATIKA
    # ==========================================
    with tab2:
        st.subheader("Operasi Geometri & Transformasi Aritmatika")
        
        # Buat 2 kolom utama dengan rasio Kiri lebih besar (1.4) dibanding Kanan (1)
        # Hal ini memaksa gambar di kolom kanan menjadi lebih kecil (menyesuaikan lebar kolom)
        # sehingga total tingginya akan sejajar dengan panel parameter di sebelah kiri.
        col_left, col_right = st.columns([1.4, 1])
        
        with col_left:
            # 1. Parameter Geometri
            st.markdown("##### 📐 Parameter Geometri")
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                rot_angle = st.slider("Rotasi Citra (°)", -180, 180, 0, 1)
                flip_h = st.checkbox("Flip Horizontal")
            with col_g2:
                scale_fac = st.slider("Scaling (Zoom)", 0.5, 2.0, 1.0, 0.1)
                flip_v = st.checkbox("Flip Vertikal")
                
            prep_img = rotate_image(original_img, rot_angle)
            if scale_fac != 1.0: prep_img = scale_image(prep_img, scale_fac)
            if flip_h: prep_img = cv2.flip(prep_img, 1)
            if flip_v: prep_img = cv2.flip(prep_img, 0)
            
            st.markdown("---")
            
            # 2. Tonal & Aritmatika
            st.markdown("##### 🎨 Tonal & Aritmatika")
            auto_opt = st.checkbox("✨ Auto-Enhance (Optimasi Tepi & Bentuk)", value=True, help="Menerapkan CLAHE secara otomatis untuk menonjolkan batas pil bulat dan kapsul sebelum dideteksi.")
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                use_gray = st.checkbox("Grayscale")
                quant_level = st.slider("Kuantisasi (Bits)", 2, 256, 256, 1)
                bright = st.slider("Brightness (Manual)", -100, 100, 0, 1)
            with col_t2:
                use_negasi = st.checkbox("Negasi Warna")
                contrast = st.slider("Contrast (Manual)", 0.1, 3.0, 1.0, 0.1)
                
            if auto_opt:
                from processing.digitalization import auto_enhance
                prep_img = auto_enhance(prep_img)
                
            # Slider manual tetap dijalankan setelah auto-enhance (jika aktif)
            prep_img = cv2.convertScaleAbs(prep_img, alpha=contrast, beta=bright)
            if quant_level < 256: prep_img = apply_quantization(prep_img, quant_level)
            if use_gray: prep_img = rgb_to_grayscale(prep_img)
            if use_negasi: prep_img = negate_image(prep_img)
            
        with col_right:
            # 3. Citra Referensi dan Hasil Pra-proses
            st.markdown("##### 🖼️ Citra Referensi")
            st.image(original_rgb, use_container_width=True)
            
            st.markdown("##### ✨ Hasil Pra-proses")
            st.image(convert_to_rgb(prep_img) if len(prep_img.shape) == 3 else prep_img, use_container_width=True)

    # ==========================================
    # TAHAP 3: MASKING & DETEKSI TEPI
    # ==========================================
    with tab3:
        st.subheader("Pembuatan Mask & Deteksi Tepi")
        st.write("Batas warna obat dideteksi menggunakan algoritma terpilih, kemudian ditambal melalui morfologi `Closing`.")
        
        edge_method = st.selectbox("Pilih Metode Deteksi Tepi (Syarat Minimal 2 UAS):", 
                                  ["Canny (Rekomendasi Modern)", "Sobel", "Prewitt", "Laplacian"])
        
        if len(prep_img.shape) == 3:
            gray_for_edge = cv2.cvtColor(prep_img, cv2.COLOR_BGR2GRAY)
            prep_bgr_for_segmentation = prep_img.copy()
        else:
            gray_for_edge = prep_img.copy()
            prep_bgr_for_segmentation = cv2.cvtColor(prep_img, cv2.COLOR_GRAY2BGR)
        
        if edge_method == "Canny (Rekomendasi Modern)":
            # Ganti Gaussian Blur dengan Bilateral Filter untuk meratakan tekstur foil
            blurred_viz = cv2.bilateralFilter(prep_bgr_for_segmentation, 9, 75, 75)
            edges_viz = cv2.Canny(blurred_viz, 30, 150)
            desc = "Deteksi tepi optimal dan tahan noise (Modern)."
        elif edge_method == "Sobel":
            edges_viz = detect_edges_sobel(gray_for_edge)
            desc = "Deteksi tepi berbasis gradien orde pertama (Metode Wajib 1)."
        elif edge_method == "Prewitt":
            edges_viz = detect_edges_prewitt(gray_for_edge)
            desc = "Mirip Sobel dengan kernel berbeda (Metode Wajib 2)."
        else:
            edges_viz = detect_edges_laplacian(gray_for_edge)
            desc = "Deteksi tepi turunan orde kedua (Metode Wajib 3)."
            
        st.markdown("##### ⚙️ Tuning Masking (Penambal Tepi)")
        kernel_size_close = st.slider(
            "Sensitivitas Penggabungan (Ukuran Kernel Closing)", 
            min_value=3, max_value=21, value=5, step=2, 
            help="Gunakan angka kecil (misal 5) untuk kemasan foil yang ramai agar tidak menyatu jadi gumpalan. Gunakan angka besar (misal 11 atau 15) untuk latar belakang bersih (kertas/meja)."
        )
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_close, kernel_size_close))
        closed_viz = cv2.morphologyEx(edges_viz, cv2.MORPH_CLOSE, kernel_close)
        
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.markdown(f"##### 1️⃣ {edge_method} Edge")
            st.image(edges_viz, clamp=True)
            st.caption(desc)
        with col_e2:
            st.markdown("##### 2️⃣ Morfologi Closing")
            st.image(closed_viz, clamp=True)
            st.caption("Garis tepi ditambal agar membungkus pil secara utuh.")

    # ==========================================
    # TAHAP 4: SEGMENTASI, SEPARASI OBJEK & HASIL AKHIR
    # ==========================================
    with tab4:
        st.subheader("Segmentasi Bounding Box & Pemisahan Pil")
        with st.spinner('Menjalankan ekstraksi rasio geometri untuk klasifikasi...'):
            img_segmented, final_mask, count_bulat, count_kapsul, final_rects = segment_edge_based(edges_viz, prep_bgr_for_segmentation, kernel_size_close)
            
            col_m1, col_m2 = st.columns([1, 1.5])
            with col_m1:
                st.markdown("##### 🧩 Solid Mask")
                st.image(final_mask)
                st.caption("Erosi & Dilasi memastikan pil yang menempel bisa terpisah.")
            with col_m2:
                st.markdown("##### ✅ Final Bounding Box")
                st.image(convert_to_rgb(img_segmented))
                
        st.markdown("---")
        st.markdown("### 📊 Statistik Analisis")
        col_met1, col_met2, col_met3 = st.columns(3)
        with col_met1:
            st.metric(label="Total Objek Obat", value=count_bulat + count_kapsul)
        with col_met2:
            st.metric(label="🟢 Pil Bulat", value=count_bulat)
        with col_met3:
            st.metric(label="💊 Kapsul Oval", value=count_kapsul)
            
        # Ekspor Data Format
        json_output = []
        for (x, y, w, h, jenis) in final_rects:
            json_output.append({
                "label": jenis,
                "box": [
                    int((y / h_orig) * 1000), 
                    int((x / w_orig) * 1000), 
                    int(((y + h) / h_orig) * 1000), 
                    int(((x + w) / w_orig) * 1000)
                ]
            })
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Ekspor Data (JSON)",
            data=json.dumps(json_output, indent=2),
            file_name="hasil_segmentasi.json",
            mime="application/json"
        )

else:
    st.info("👋 Selamat datang! Silakan pilih sumber gambar di panel sebelah kiri untuk memulai analisis.")
    
    st.markdown("""
    <div style='background: rgba(30, 41, 59, 0.5); padding: 25px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.05); margin-top: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); backdrop-filter: blur(10px);'>
        <h4 style='color: #f8fafc; margin-top: 0; display: flex; align-items: center; gap: 10px;'>
            ✅ Kesesuaian Kriteria UAS:
        </h4>
        <ul style='color: #94a3b8; line-height: 1.8; margin-bottom: 0;'>
            <li><strong style='color:#e2e8f0;'>1. Representasi Citra</strong>: Matriks Pixel, RGB, Grayscale, dan Hubungan antar Pixel (Neighborhood).</li>
            <li><strong style='color:#e2e8f0;'>2. Digitalisasi</strong>: Resolusi gambar, Sampling, Kuantisasi warna.</li>
            <li><strong style='color:#e2e8f0;'>3. Geometri & Aritmatika</strong>: Modul <i>Rotasi, Scale, Flip, Grayscale, Negasi</i> (Total 5 - melebihi batas minimal 4).</li>
            <li><strong style='color:#e2e8f0;'>4. Deteksi Tepi</strong>: Dilengkapi opsi interaktif <b>Sobel, Prewitt, Laplacian, dan Canny</b> (Melebihi batas minimal 2 metode wajib).</li>
            <li><strong style='color:#e2e8f0;'>5. Segmentasi Citra</strong>: Pendekatan Edge-based dengan deteksi contour & ekstraksi Aspect Ratio untuk klasifikasi bentuk.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)