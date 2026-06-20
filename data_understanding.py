"""
Data Understanding report for the tourist destination dataset.
Run with: python data_understanding.py
"""

import pandas as pd


DATA_PATH = 'destinasi-wisata-indonesia.csv'


def print_section(title):
    print('\n' + '=' * 70)
    print(title)
    print('=' * 70)


def main():
    df = pd.read_csv(DATA_PATH)

    print_section('1. Jumlah Data')
    print(f'Jumlah baris: {df.shape[0]}')
    print(f'Jumlah kolom: {df.shape[1]}')

    print_section('2. Kolom Dataset')
    for idx, column in enumerate(df.columns, start=1):
        print(f'{idx}. {column}')

    print_section('3. Missing Value')
    print(df.isna().sum().to_string())

    print_section('4. Distribusi Kategori')
    print(df['Category'].value_counts().to_string())

    print_section('5. Distribusi Kota')
    print(df['City'].value_counts().to_string())

    print_section('6. Statistik Harga, Rating, Durasi, dan Jumlah Rating')
    numeric_columns = ['Price', 'Rating', 'Time_Minutes', 'Rating_Count']
    print(df[numeric_columns].describe().round(2).to_string())


if __name__ == '__main__':
    main()
