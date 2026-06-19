"""
Web Interface for Hybrid Recommendation System
"""

from flask import Flask, render_template, request, jsonify
from recommendation_system import HybridRecommender, save_model
import os
from datetime import datetime
import math

app = Flask(__name__)

# Initialize recommender
print("Loading recommendation system...")
recommender = HybridRecommender()
stats = recommender.get_stats()
print(f"Loaded {stats['total_destinations']} destinations")


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', 
                         cities=stats['cities'],
                         categories=stats['categories'],
                         stats=stats,
                         cache_bust=int(datetime.now().timestamp()))


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates in km using haversine formula"""
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(R * c, 1)


@app.route('/recommend', methods=['POST'])
def recommend():
    """API endpoint for recommendations"""
    data = request.json
    
    city = data.get('city', 'Semua')
    category = data.get('category', 'Semua')
    min_price = float(data.get('min_price', 0))
    max_price = float(data.get('max_price', 0))
    max_time = float(data.get('max_time', 0))
    top_n = int(data.get('top_n', 10))
    user_lat = data.get('user_lat')
    user_lng = data.get('user_lng')
    
    recs, err = recommender.recommend(
        city=city,
        category=category,
        min_price=min_price if min_price > 0 else None,
        max_price=max_price if max_price > 0 else None,
        max_time=max_time if max_time > 0 else None,
        top_n=top_n
    )
    
    if err:
        return jsonify({'error': err, 'results': []})
    
    results = []
    for _, row in recs.iterrows():
        dest_lat = row['Lat'] if pd.notna(row['Lat']) else None
        dest_lng = row['Long'] if pd.notna(row['Long']) else None
        distance = None
        if user_lat is not None and user_lng is not None and dest_lat is not None and dest_lng is not None:
            try:
                distance = haversine_km(float(user_lat), float(user_lng), float(dest_lat), float(dest_lng))
            except Exception:
                distance = None
        
        results.append({
            'id': int(row['Place_Id']),
            'name': row['Place_Name'],
            'city': row['City'],
            'category': row['Category'],
            'price': int(row['Price']),
            'rating': float(row['Rating']),
            'rating_count': int(row['Rating_Count']),
            'time': row['Time_Minutes'] if pd.notna(row['Time_Minutes']) else 0,
            'score': float(row['Score']),
            'distance': distance
        })
    
    return jsonify({'error': None, 'results': results})


@app.route('/similar/<place_name>')
def similar(place_name):
    """Get similar destinations"""
    sims, err = recommender.get_similar_destinations(place_name, top_n=5)
    
    if err:
        return jsonify({'error': err, 'results': []})
    
    results = []
    for _, row in sims.iterrows():
        results.append({
            'id': int(row['Place_Id']),
            'name': row['Place_Name'],
            'city': row['City'],
            'category': row['Category'],
            'price': int(row['Price']),
            'rating': float(row['Rating']),
            'similarity': float(row['Similarity'])
        })
    
    return jsonify({'error': None, 'results': results})


import pandas as pd

@app.route('/popular')
def popular():
    """Get popular destinations"""
    city = request.args.get('city', 'Semua')
    top_n = int(request.args.get('top_n', 10))
    user_lat = request.args.get('user_lat')
    user_lng = request.args.get('user_lng')
    
    pop = recommender.get_popular(top_n=top_n, city=city)
    
    results = []
    for _, row in pop.iterrows():
        dest_lat = row['Lat'] if pd.notna(row['Lat']) else None
        dest_lng = row['Long'] if pd.notna(row['Long']) else None
        distance = None
        if user_lat is not None and user_lng is not None and dest_lat is not None and dest_lng is not None:
            try:
                distance = haversine_km(float(user_lat), float(user_lng), float(dest_lat), float(dest_lng))
            except Exception:
                distance = None
        
        results.append({
            'id': int(row['Place_Id']),
            'name': row['Place_Name'],
            'city': row['City'],
            'category': row['Category'],
            'price': int(row['Price']),
            'rating': float(row['Rating']),
            'rating_count': int(row['Rating_Count']),
            'distance': distance
        })
    
    return jsonify({'results': results})


@app.route('/stats')
def get_stats():
    """Get dataset statistics"""
    return jsonify(stats)


if __name__ == '__main__':
    # Create templates directory if not exists
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Pre-load and save model
    save_model(recommender)
    
    print("\n" + "="*50)
    print("Hybrid Recommendation System Started!")
    print("="*50)
    print(f"Open browser at: http://localhost:5000")
    print("="*50 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000)