"""
Hybrid Recommendation System for Indonesian Tourist Destinations
Combines Knowledge-based filtering + Content-based filtering
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle
import os
import re

class HybridRecommender:
    def __init__(self, data_path='destinasi-wisata-indonesia.csv'):
        """Initialize the hybrid recommendation system"""
        self.df = pd.read_csv(data_path)
        self.preprocess_data()
        self.build_content_model()
        
    def preprocess_data(self):
        """Preprocess the dataset"""
        # Parse coordinates from 'Coordinate' column (format: {'lat': value, 'lng': value})
        def parse_coordinate(coord_str):
            try:
                import ast
                coord_dict = ast.literal_eval(coord_str)
                return coord_dict.get('lat'), coord_dict.get('lng')
            except Exception:
                return None, None

        # Extract lat/lng from Coordinate column
        self.df[['Lat', 'Long']] = self.df['Coordinate'].apply(
            lambda x: pd.Series(parse_coordinate(x))
        )

        # Remove columns that are empty, duplicated, or not used by the model.
        unused_columns = ['Column1', '_1', 'Coordinate']
        self.df = self.df.drop(columns=[c for c in unused_columns if c in self.df.columns])

        # Clean categorical columns
        self.df['Category'] = self.df['Category'].fillna('Tidak Diketahui').astype(str).str.strip()
        self.df['City'] = self.df['City'].fillna('Tidak Diketahui').astype(str).str.strip()

        self.df['Price'] = pd.to_numeric(self.df['Price'], errors='coerce').fillna(0)
        self.df['Rating'] = pd.to_numeric(self.df['Rating'], errors='coerce').fillna(0)
        self.df['Time_Minutes'] = pd.to_numeric(self.df['Time_Minutes'], errors='coerce').fillna(0)
        self.df['Rating_Count'] = pd.to_numeric(self.df['Rating_Count'], errors='coerce').fillna(0)

        # Fill empty descriptions and clean text for TF-IDF.
        self.df['Description'] = self.df['Description'].fillna('')
        self.df['Description_Clean'] = self.df['Description'].apply(self.clean_text)

        # Normalize rating to 0-1. The dataset mostly stores 4.2 as 42, so convert to
        # a 1-5 scale first when values are encoded as tens.
        max_rating = self.df['Rating'].max()
        rating_on_five = self.df['Rating'] / 10.0 if max_rating > 5 else self.df['Rating']
        self.df['Rating_Normalized'] = (rating_on_five / 5.0).clip(0, 1)

        scaler = MinMaxScaler()
        numeric_columns = ['Price', 'Time_Minutes', 'Rating_Count']
        scaled_values = scaler.fit_transform(self.df[numeric_columns])
        self.df[['Price_Normalized', 'Time_Normalized', 'Rating_Count_Normalized']] = scaled_values
        self.numeric_scaler = scaler
        
        # Create combined features for content-based filtering
        self.df['combined_features'] = (
            self.df['Category'].astype(str) + ' ' +
            self.df['City'].astype(str) + ' ' +
            self.df['Description_Clean'].astype(str)
        )

    @staticmethod
    def clean_text(text):
        """Lowercase text, remove punctuation/digits, and collapse whitespace."""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
        
    def build_content_model(self):
        """Build a combined feature matrix for content-based filtering."""
        self.feature_preprocessor = ColumnTransformer(
            transformers=[
                ('description_tfidf', TfidfVectorizer(max_features=5000), 'Description_Clean'),
                ('category_city_ohe', OneHotEncoder(handle_unknown='ignore'), ['Category', 'City']),
                (
                    'numeric_scaled',
                    MinMaxScaler(),
                    ['Price', 'Time_Minutes', 'Rating', 'Rating_Count']
                )
            ]
        )
        self.feature_matrix = self.feature_preprocessor.fit_transform(self.df)
        self.tfidf_matrix = self.feature_matrix
        self.cosine_sim = cosine_similarity(self.feature_matrix, self.feature_matrix)
        
    def knowledge_based_filter(self, city=None, category=None, min_price=None, max_price=None, max_time=None):
        """
        Filter destinations based on user constraints (knowledge-based)
        Returns filtered dataframe
        """
        filtered_df = self.df.copy()
        
        if city and city != 'Semua':
            filtered_df = filtered_df[filtered_df['City'].str.lower() == city.lower()]
            
        if category and category != 'Semua':
            filtered_df = filtered_df[filtered_df['Category'].str.lower() == category.lower()]
            
        if min_price is not None and min_price > 0:
            filtered_df = filtered_df[filtered_df['Price'] >= min_price]
            
        if max_price is not None and max_price > 0:
            filtered_df = filtered_df[filtered_df['Price'] <= max_price]
            
        if max_time is not None and max_time > 0:
            filtered_df = filtered_df[filtered_df['Time_Minutes'] <= max_time]
            
        return filtered_df
    
    def content_based_score(self, idx, filtered_df):
        """
        Calculate content-based similarity score for a given item
        Higher score = more similar to TOP-RATED items in filtered set
        """
        if len(filtered_df) <= 1:
            return 0.5
        
        # Get indices in original dataframe
        indices = [self.df.index[self.df['Place_Id'] == pid].tolist()[0] 
                   for pid in filtered_df['Place_Id']]
        
        # Find top 3 highest rated items in filtered set as "prototypes"
        filtered_with_idx = [(i, self.df.iloc[i]['Rating']) for i in indices]
        filtered_with_idx.sort(key=lambda x: x[1], reverse=True)
        top_indices = [x[0] for x in filtered_with_idx[:min(3, len(filtered_with_idx))]]
        
        # Calculate similarity to top-rated items (more meaningful than average similarity)
        sim_scores = []
        for top_idx in top_indices:
            if top_idx != idx:
                sim_scores.append(self.cosine_sim[idx][top_idx])
        
        if len(sim_scores) == 0:
            return 0.5
            
        return np.mean(sim_scores)
    
    def recommend(self, city=None, category=None, min_price=None, max_price=None, max_time=None, 
                  top_n=10, preference_text=None, user_lat=None, user_lng=None, distance_weight=0.1):
        """
        Hybrid recommendation: Knowledge-based + Content-based
        
        Parameters:
        - city: preferred city
        - category: preferred category  
        - max_price: maximum price
        - max_time: maximum time in minutes
        - top_n: number of recommendations
        - preference_text: optional text describing preferences for content-based
        """
        # Step 1: Knowledge-based filtering
        filtered_df = self.knowledge_based_filter(city, category, min_price, max_price, max_time)
        
        if len(filtered_df) == 0:
            return pd.DataFrame(), "Tidak ada destinasi yang sesuai dengan kriteria Anda."
        
        # Step 2: Hybrid scoring with better balance
        # If user location is provided, compute distance for each candidate and include it
        scores = []
        for _, row in filtered_df.iterrows():
            idx = self.df.index[self.df['Place_Id'] == row['Place_Id']].tolist()[0]

            # Content-based score (similarity to top-rated items in filtered set)
            content_score = self.content_based_score(idx, filtered_df)

            # Normalized quality signals from preprocessing.
            rating_score = row['Rating_Normalized'] if row['Rating'] > 0 else 0
            count_score = row['Rating_Count_Normalized'] if row['Rating_Count'] > 0 else 0

            # Hybrid score: BALANCED combination
            # 60% quality (rating + popularity) ensures we recommend well-rated items
            # 40% content similarity ensures relevance and diversity within category
            quality_score = 0.7 * rating_score + 0.3 * count_score
            hybrid_score = 0.60 * quality_score + 0.40 * content_score

            # If user location provided, compute distance (km) using haversine
            dist_value = None
            if user_lat is not None and user_lng is not None and 'Lat' in row and 'Long' in row:
                try:
                    lat1 = float(user_lat)
                    lon1 = float(user_lng)
                    lat2 = float(row['Lat'])
                    lon2 = float(row['Long'])
                    R = 6371.0
                    phi1 = np.radians(lat1)
                    phi2 = np.radians(lat2)
                    dphi = np.radians(lat2 - lat1)
                    dlambda = np.radians(lon2 - lon1)
                    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
                    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
                    distance_km = R * c
                    dist_value = float(distance_km)
                except Exception:
                    dist_value = None

            scores.append((hybrid_score, dist_value))
        
        filtered_df = filtered_df.copy()
        hybrid_scores = [s[0] for s in scores]
        distances = [s[1] for s in scores]
        filtered_df['Score'] = hybrid_scores
        filtered_df['Distance_Km'] = distances

        # If distances exist and a distance weight is provided, normalize distance and combine
        if any(d is not None for d in distances) and distance_weight and distance_weight > 0:
            # Replace None with a large value so they get low distance score
            available = [d for d in distances if d is not None]
            max_d = max(available) if available else 1.0
            numeric_dists = [d if d is not None else max_d for d in distances]
            distance_scores = [1.0 - (d / max_d) if max_d > 0 else 0.0 for d in numeric_dists]
            filtered_df['Distance_Score'] = distance_scores
            filtered_df['Score'] = filtered_df['Score'] * (1.0 - distance_weight) + filtered_df['Distance_Score'] * distance_weight

        # Sort by final score
        filtered_df = filtered_df.sort_values('Score', ascending=False)

        # Get top N
        recommendations = filtered_df.head(top_n)
        
        return recommendations, None
    
    def get_similar_destinations(self, place_name, top_n=5):
        """Get similar destinations based on content similarity"""
        idx = self.df.index[self.df['Place_Name'].str.lower() == place_name.lower()].tolist()
        
        if len(idx) == 0:
            return pd.DataFrame(), "Destinasi tidak ditemukan."
        
        idx = idx[0]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        
        similar_indices = [i[0] for i in sim_scores]
        similar_df = self.df.iloc[similar_indices].copy()
        similar_df['Similarity'] = [i[1] for i in sim_scores]
        
        return similar_df, None
    
    def get_popular(self, top_n=10, city=None):
        """Get popular destinations based on rating"""
        df = self.df.copy()
        if city and city != 'Semua':
            df = df[df['City'].str.lower() == city.lower()]
        
        # Sort by rating and rating count
        df['Popularity'] = df['Rating'] * np.log1p(df['Rating_Count'])
        df = df.sort_values('Popularity', ascending=False)
        return df.head(top_n)
    
    def get_stats(self):
        """Get dataset statistics"""
        stats = {
            'total_destinations': len(self.df),
            'cities': self.df['City'].unique().tolist(),
            'categories': self.df['Category'].unique().tolist(),
            'avg_rating': round(self.df['Rating'].mean() / 10.0, 2),
            'price_range': (self.df['Price'].min(), self.df['Price'].max()),
            'missing_values': self.df.isna().sum().to_dict(),
            'category_distribution': self.df['Category'].value_counts().to_dict(),
            'city_distribution': self.df['City'].value_counts().to_dict(),
            'price_stats': self.df['Price'].describe().round(2).to_dict(),
            'rating_stats': self.df['Rating'].describe().round(2).to_dict()
        }
        return stats


def save_model(model, filename='model.pkl'):
    """Save trained model to file"""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
        
def load_model(filename='model.pkl'):
    """Load trained model from file"""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None


if __name__ == "__main__":
    # Test the recommendation system
    recommender = HybridRecommender()
    
    print("=== Hybrid Recommendation System ===\n")
    
    # Test 1: Knowledge-based + Content-based
    print("Test 1: Recommendations for Jakarta - Budaya - Max Price 50000")
    recs, err = recommender.recommend(city='Jakarta', category='Budaya', max_price=50000, top_n=5)
    if err:
        print(err)
    else:
        for _, row in recs.iterrows():
            print(f"- {row['Place_Name']} | Rating: {row['Rating']} | Price: Rp{row['Price']:,.0f}")
    
    print("\nTest 2: Recommendations for Yogyakarta - Taman Hiburan")
    recs, err = recommender.recommend(city='Yogyakarta', category='Taman Hiburan', top_n=5)
    if err:
        print(err)
    else:
        for _, row in recs.iterrows():
            print(f"- {row['Place_Name']} | Rating: {row['Rating']} | Price: Rp{row['Price']:,.0f}")
    
    print("\nTest 3: Similar to Candi Prambanan")
    sims, err = recommender.get_similar_destinations('Candi Prambanan', top_n=5)
    if err:
        print(err)
    else:
        for _, row in sims.iterrows():
            print(f"- {row['Place_Name']} | Similarity: {row['Similarity']:.3f}")
    
    print("\nDataset Stats:")
    stats = recommender.get_stats()
    print(f"Total destinations: {stats['total_destinations']}")
    print(f"Cities: {', '.join(stats['cities'])}")
    print(f"Categories: {', '.join(stats['categories'])}")
