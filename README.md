# Sistem Rekomendasi Destinasi Wisata Indonesia
## Hybrid Recommendation System (Knowledge-based + Content-based)

---

## 📋 Deskripsi Proyek

Sistem rekomendasi hibrida untuk destinasi wisata Indonesia yang menggabungkan:
- **Knowledge-based filtering**: Filtering berdasarkan constraint pengguna (kota, kategori, harga, waktu)
- **Content-based filtering**: Skor kesamaan konten menggunakan TF-IDF dan cosine similarity

---

## 📁 Struktur File

```
TUGAS-AKHIR-SPR/
├── destinasi-wisata-indonesia.csv   # Dataset destinasi wisata Indonesia
├── recommendation_system.py         # Core hybrid recommendation engine
├── app.py                           # Flask web application
├── evaluation.py                    # Evaluation metrics module
├── verify.py                        # System verification script
├── requirements.txt                 # Python dependencies
└── templates/
    └── index.html                   # Web interface
```

---

## 🚀 Cara Menjalankan

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan Web Application
```bash
python app.py
```

Buka browser dan akses: `http://localhost:5000`

### 3. Jalankan Evaluasi
```bash
python evaluation.py
```

### 4. Verifikasi Sistem
```bash
python verify.py
```

---

## 📊 Fitur Sistem

### Hybrid Recommendation
- **Knowledge-based**: Filter destinasi berdasarkan:
  - Kota (Jakarta, Yogyakarta, Bandung, Semarang, Surabaya)
  - Kategori (Budaya, Taman Hiburan, Cagar Alam, Bahari, Pusat Perbelanjaan, Tempat Ibadah)
  - Harga maksimal (Rp)
  - Waktu kunjungan maksimal (menit)

- **Content-based**: 
  - TF-IDF vectorization pada description, category, dan city
  - Cosine similarity untuk mengukur kesamaan antar destinasi
  - Skor hybrid yang mempertimbangkan similarity, rating, dan jumlah pengunjung

### Metode Input
- Filter by preferences
- Similar destinations (berdasarkan destinasi favorit)
- Popular destinations (berdasarkan rating)

### Web Interface
- Responsive design dengan gradient theme
- Filter controls untuk kota, kategori, harga, waktu
- Real-time recommendations
- Tampilan grid dengan badge kategori dan kota
- Statistik dataset

---

## 📈 Hasil Evaluasi

Metrik evaluasi pada 17 test cases:

| Metrik | K=5 | K=10 | K=20 |
|--------|-----|------|------|
| Precision | 0.4471 | - | - |
| Recall | 0.5121 | - | - |
| F1-Score | 0.4281 | - | - |
| MAP | 0.6837 | - | - |
| Hit Rate | 0.9412 | - | - |
| Diversity | 0.1638 | - | - |
| Novelty | 0.2363 | - | - |
| Coverage | 0.1716 | - | - |

**Interpretasi:**
- **Precision@5 = 0.4471**: 44.71% dari 5 rekomendasi adalah relevan
- **Recall@5 = 0.5121**: 51.21% dari semua destinasi relevan direkomendasikan
- **Hit Rate@5 = 0.9412**: 94.12% kasus mendapatkan minimal 1 rekomendasi relevan
- **MAP@5 = 0.6837**: Rata-rata precision yang sangat baik
- **Coverage = 0.1716**: 17.16% dari dataset dapat direkomendasikan (masuk kriteria filter)

### Hybrid Scoring Strategy
Sistem menggunakan skor hibrida yang seimbang:
- **60% Quality Score** (70% rating + 30% popularity) - memastikan kualitas rekomendasi
- **40% Content Similarity** - memastikan relevansi dan diversity dalam kategori

---

## 🏗️ Arsitektur Sistem

### 1. Data Preprocessing
- Cleaning: Price, Rating, Time_Minutes, Rating_Count
- Feature extraction: Combined features (Category + City + Description)
- TF-IDF vectorization (5000 features max)

### 2. Knowledge-based Component
```
User Input → Filter Constraints → Candidate Items
```

### 3. Content-based Component
```
Candidate Items → Cosine Similarity → Similarity Scores
```

### 4. Hybrid Scoring
```
Quality Score = 0.7 × Rating_Score + 0.3 × Count_Score
Hybrid Score = 0.6 × Quality_Score + 0.4 × Content_Score
```

---

## 📝 Dataset

**Sumber**: destinasi-wisata-indonesia.csv

**Statistik**:
- Total Destinasi: 437
- Kota: 5 (Jakarta, Yogyakarta, Bandung, Semarang, Surabaya)
- Kategori: 6
- Rating Rata-rata: 45.3

---

## 🛠️ Teknologi

- **Python 3.x**
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - TF-IDF & Cosine Similarity
- **Flask** - Web framework

---

## 🎯 Penggunaan

### Contoh Penggunaan (Python API)

```python
from recommendation_system import HybridRecommender

# Initialize
recommender = HybridRecommender()

# Get hybrid recommendations
recs, error = recommender.recommend(
    city='Yogyakarta',
    category='Budaya',
    max_price=50000,
    top_n=10
)

# Get similar destinations
sims, error = recommender.get_similar_destinations('Candi Prambanan', top_n=5)

# Get popular destinations
popular = recommender.get_popular(city='Jakarta', top_n=10)
```

---

## 📄 Format Laporan

Proyek ini memenuhi ketentuan tugas akhir dengan:

### BAB I - Pendahuluan
- Latar belakang: Sistem rekomendasi untuk wisata Indonesia
- Rumusan masalah: Bagaimana merekomendasikan destinasi wisata yang relevan
- Tujuan: Membangun sistem rekomendasi hybrid
- Manfaat: Membantu wisatawan menemukan destinasi sesuai preferensi

### BAB II - Tinjauan Pustaka
- Sistem Rekomendasi
- Knowledge-based filtering
- Content-based filtering
- Hybrid recommendation systems

### BAB III - Metodologi
- Dataset: 437 destinasi wisata Indonesia
- Tahapan penelitian: Preprocessing, modeling, evaluation
- Algoritma: TF-IDF, Cosine Similarity, Hybrid scoring
- Metode evaluasi: Precision, Recall, F1, MAP, Hit Rate, Coverage

### BAB IV - Hasil dan Pembahasan
- Implementasi sistem web dengan Flask
- Demo rekomendasi
- Analisis hasil evaluasi
- Visualisasi metrik

### BAB V - Kesimpulan
- Sistem hybrid bekerja dengan baik (Precision@5: 44.71%)
- Kombinasi knowledge-based + content-based efektif
- Web interface user-friendly

---

## 👨‍💻 Developer

Tugas Akhir Mata Kuliah Sistem Rekomendasi
Menggunakan dataset destinasi-wisata-indonesia.csv

---

## 📚 Referensi

1. Aggarwal, C. C. (2016). Recommender Systems: The Textbook. Springer.
2. Ricci, F., et al. (2015). Recommender Systems Handbook. Springer.
3. Pazzani, M. J., & Billsus, D. (2007). Content-Based Recommendation Systems. Springer.
4. Burke, R. (2002). Hybrid Recommender Systems. User Modeling and User-Adapted Interaction.
5. Lops, P., et al. (2011). Content-based Recommender Systems. Springer.
6. Adomavicius, G., & Tuzhilin, A. (2005). Toward the Next Generation of Recommender Systems. IEEE.
7. Resnick, P., & Varian, H. R. (1997). Recommender Systems. CACM.
8. Sarwar, B., et al. (2001). Item-based Collaborative Filtering. WWW.
9. Salton, G., & McGill, M. J. (1983). Introduction to Modern Information Retrieval. McGraw-Hill.
10. Manning, C. D., et al. (2008). Introduction to Information Retrieval. Cambridge University Press.

---

## 📞 Kontak

Untuk pertanyaan lebih lanjut tentang implementasi sistem ini.