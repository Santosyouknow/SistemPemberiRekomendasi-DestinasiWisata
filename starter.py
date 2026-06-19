"""
Quick Start Script - Sistem Rekomendasi Destinasi Wisata
Jalankan file ini untuk memulai sistem
"""

import os
import sys

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import flask
        import pandas
        import sklearn
        print("✓ Semua dependencies sudah terinstall")
        return True
    except ImportError as e:
        print("✗ Missing dependency:", str(e))
        print("\nInstalling requirements...")
        os.system("pip install -q pandas numpy scikit-learn flask")
        return True

def main():
    print("="*60)
    print("SISTEM REKOMENDASI DESTINASI WISATA INDONESIA")
    print("Hybrid Recommendation System (Knowledge-based + Content-based)")
    print("="*60)
    print()
    
    # Check dependencies
    check_dependencies()
    
    print("\n" + "="*60)
    print("MENJALANKAN WEB APPLICATION")
    print("="*60)
    print("\n🌐 Browser akan buka di: http://localhost:5000")
    print("📌 Tekan Ctrl+C untuk berhenti\n")
    print("="*60 + "\n")
    
    # Run Flask app
    try:
        from app import app
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("Sistem dihentikan. Terima kasih!")
        print("="*60)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nPastikan file berikut ada:")
        print("  - destinasi-wisata-indonesia.csv")
        print("  - recommendation_system.py")
        print("  - app.py")
        print("  - templates/index.html")
        input("\nTekan Enter untuk keluar...")

if __name__ == "__main__":
    main()