# SistemPemberiRekomendasi-DestinasiWisata

## Deskripsi Proyek
Sistem rekomendasi hibrida destinasi wisata Indonesia. Dibangun dengan Python, Flask, dan scikit-learn menggunakan pendekatan Knowledge-based + Content-based filtering.

## Struktur File
```
├── app.py                           # Aplikasi web Flask
├── recommendation_system.py         # Engine rekomendasi hybrid
├── destinasi-wisata-indonesia.csv  # Dataset destinasi wisata
├── requirements.txt                 # Dependensi Python
├── starter.py                       # Script starter
├── evaluate.py                      # Evaluasi sistem
├── verify.py                        # Verifikasi sistem
├── static/                          # Assets statis
└── templates/                       # Template HTML
    └── index.html                   # Halaman utama
```

## Cara Menjalankan
1. Install dependensi: `pip install -r requirements.txt`
2. Jalankan aplikasi: `python app.py`
3. Buka browser di: `http://localhost:5000`

## Teknologi yang Digunakan
- Python 3.x
- Flask (Web Framework)
- Pandas & NumPy (Data Processing)
- Scikit-learn (Machine Learning)
- HTML/CSS/JavaScript (Frontend)