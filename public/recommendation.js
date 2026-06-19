/**
 * JavaScript Recommendation Engine for Indonesian Tourist Destinations
 * Based on Hybrid Recommendation System: Knowledge-based + Content-based
 */

class HybridRecommender {
    constructor(data) {
        this.destinations = data;
        this.cities = [...new Set(data.map(d => d.City))].sort();
        this.categories = [...new Set(data.map(d => d.Category))].sort();
        this.totalCount = data.length;
        this.avgRating = parseFloat((data.reduce((s, d) => s + d.Rating, 0) / data.length / 10).toFixed(2));
        this.priceRange = [Math.min(...data.map(d => d.Price)), Math.max(...data.map(d => d.Price))];
        
        // Precompute TF-IDF for content-based filtering
        this.buildContentModel();
        
        // Compute popularity scores in advance
        this.computePopularity();
    }
    
    buildContentModel() {
        // Build term frequency vectors for each destination
        const documents = this.destinations.map(d => {
            return (d.Category + ' ' + d.City + ' ' + d.Description).toLowerCase();
        });
        
        // Build vocabulary
        const vocab = new Map();
        let vocabIdx = 0;
        
        documents.forEach(doc => {
            const words = doc.split(/[\s,.'"]+/).filter(w => w.length > 2);
            words.forEach(w => {
                if (!vocab.has(w)) vocab.set(w, vocabIdx++);
            });
        });
        
        this.vocab = vocab;
        this.vocabSize = vocab.size;
        
        // Compute IDF for each term
        const docFreq = new Array(vocab.size).fill(0);
        documents.forEach(doc => {
            const words = new Set(doc.split(/[\s,.'"]+/).filter(w => w.length > 2));
            words.forEach(w => {
                const idx = vocab.get(w);
                if (idx !== undefined) docFreq[idx]++;
            });
        });
        
        this.idf = docFreq.map(df => Math.log(documents.length / (df + 1)));
        
        // Compute TF-IDF vectors for each document
        this.tfidfVectors = documents.map(doc => {
            const words = doc.split(/[\s,.'"]+/).filter(w => w.length > 2);
            const tf = new Array(vocab.size).fill(0);
            words.forEach(w => {
                const idx = vocab.get(w);
                if (idx !== undefined) tf[idx]++;
            });
            return tf.map((freq, i) => freq * this.idf[i]);
        });
        
        // Compute cosine similarity matrix
        this.computeCosineSimilarity();
    }
    
    computeCosineSimilarity() {
        const n = this.tfidfVectors.length;
        this.cosineSim = new Array(n);
        
        for (let i = 0; i < n; i++) {
            this.cosineSim[i] = new Array(n);
            for (let j = i; j < n; j++) {
                const sim = this.cosineSimilarity(i, j);
                this.cosineSim[i][j] = sim;
                this.cosineSim[j][i] = sim;
            }
        }
    }
    
    cosineSimilarity(i, j) {
        const a = this.tfidfVectors[i];
        const b = this.tfidfVectors[j];
        let dot = 0, magA = 0, magB = 0;
        for (let k = 0; k < a.length; k++) {
            dot += a[k] * b[k];
            magA += a[k] * a[k];
            magB += b[k] * b[k];
        }
        const denom = Math.sqrt(magA) * Math.sqrt(magB);
        return denom === 0 ? 0 : dot / denom;
    }
    
    computePopularity() {
        this.destinations.forEach(d => {
            d.Popularity = (d.Rating || 0) * Math.log1p(d.Rating_Count || 0);
        });
    }
    
    getStats() {
        return {
            total_destinations: this.totalCount,
            cities: this.cities,
            categories: this.categories,
            avg_rating: this.avgRating,
            price_range: this.priceRange
        };
    }
    
    knowledgeBasedFilter(city, category, minPrice, maxPrice, maxTime) {
        return this.destinations.filter(d => {
            if (city && city !== 'Semua' && d.City.toLowerCase() !== city.toLowerCase()) return false;
            if (category && category !== 'Semua' && d.Category.toLowerCase() !== category.toLowerCase()) return false;
            if (minPrice && minPrice > 0 && d.Price < minPrice) return false;
            if (maxPrice && maxPrice > 0 && d.Price > maxPrice) return false;
            if (maxTime && maxTime > 0 && d.Time_Minutes && d.Time_Minutes > maxTime) return false;
            return true;
        });
    }
    
    contentBasedScore(idx, filtered) {
        if (filtered.length <= 1) return 0.5;
        
        const indices = filtered.map(f => this.destinations.indexOf(f));
        const withIdx = indices.map(i => ({ i, rating: this.destinations[i].Rating || 0 }));
        withIdx.sort((a, b) => b.rating - a.rating);
        const topIndices = withIdx.slice(0, Math.min(3, withIdx.length)).map(x => x.i);
        
        const sims = topIndices.filter(ti => ti !== idx).map(ti => this.cosineSim[idx][ti]);
        if (sims.length === 0) return 0.5;
        
        return sims.reduce((a, b) => a + b, 0) / sims.length;
    }
    
    recommend(city, category, minPrice, maxPrice, maxTime, topN) {
        const filtered = this.knowledgeBasedFilter(city, category, minPrice, maxPrice, maxTime);
        
        if (filtered.length === 0) {
            return { recs: [], error: 'Tidak ada destinasi yang sesuai dengan kriteria Anda.' };
        }
        
        const scores = filtered.map(d => {
            const idx = this.destinations.indexOf(d);
            const contentScore = this.contentBasedScore(idx, filtered);
            const ratingScore = (d.Rating || 0) / 100;
            const countScore = Math.min((d.Rating_Count || 0) / 50, 1);
            const qualityScore = 0.7 * ratingScore + 0.3 * countScore;
            const hybridScore = 0.6 * qualityScore + 0.4 * contentScore;
            return { dest: d, score: hybridScore };
        });
        
        scores.sort((a, b) => b.score - a.score);
        
        const results = scores.slice(0, topN).map(s => ({
            id: s.dest.Place_Id,
            name: s.dest.Place_Name,
            city: s.dest.City,
            category: s.dest.Category,
            price: s.dest.Price,
            rating: s.dest.Rating,
            rating_count: s.dest.Rating_Count,
            time: s.dest.Time_Minutes || 0,
            score: parseFloat(s.score.toFixed(4)),
            description: s.dest.Description
        }));
        
        return { recs: results, error: null };
    }
    
    getSimilarDestinations(placeName, topN) {
        const idx = this.destinations.findIndex(d => d.Place_Name.toLowerCase() === placeName.toLowerCase());
        if (idx === -1) return { sims: [], error: 'Destinasi tidak ditemukan.' };
        
        const simScores = this.cosineSim[idx]
            .map((sim, i) => ({ i, sim }))
            .filter(item => item.i !== idx)
            .sort((a, b) => b.sim - a.sim)
            .slice(0, topN);
        
        const results = simScores.map(item => ({
            id: this.destinations[item.i].Place_Id,
            name: this.destinations[item.i].Place_Name,
            city: this.destinations[item.i].City,
            category: this.destinations[item.i].Category,
            price: this.destinations[item.i].Price,
            rating: this.destinations[item.i].Rating,
            similarity: parseFloat(item.sim.toFixed(4))
        }));
        
        return { sims: results, error: null };
    }
    
    getPopular(topN, city) {
        let df = this.destinations.filter(d => {
            return !city || city === 'Semua' || d.City.toLowerCase() === city.toLowerCase();
        });
        df.sort((a, b) => b.Popularity - a.Popularity);
        return df.slice(0, topN).map(d => ({
            id: d.Place_Id,
            name: d.Place_Name,
            city: d.City,
            category: d.Category,
            price: d.Price,
            rating: d.Rating,
            rating_count: d.Rating_Count,
            Popularity: parseFloat(d.Popularity.toFixed(2))
        }));
    }
}

// Haversine formula for distance calculation
function haversineKm(lat1, lon1, lat2, lon2) {
    const R = 6371;
    const toRad = d => d * Math.PI / 180;
    const dLat = toRad(lat2 - lat1);
    const dLon = toRad(lon2 - lon1);
    const phi1 = toRad(lat1);
    const phi2 = toRad(lat2);
    const a = Math.sin(dLat / 2) ** 2 + Math.cos(phi1) * Math.cos(phi2) * Math.sin(dLon / 2) ** 2;
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return Math.round(R * c * 10) / 10;
}

// Initialize when data is loaded
let recommender = null;
let currentStats = null;

function initApp() {
    fetch('destinations.json')
        .then(r => r.json())
        .then(data => {
            // Clean data
            data.forEach(d => {
                d.Price = parseFloat(d.Price) || 0;
                d.Rating = parseFloat(d.Rating) || 0;
                d.Rating_Count = parseFloat(d.Rating_Count) || 0;
                d.Time_Minutes = d.Time_Minutes === '' ? 0 : parseFloat(d.Time_Minutes);
                d.Lat = d.Lat === '' || d.Lat === null ? null : parseFloat(d.Lat);
                d.Long = d.Long === '' || d.Long === null ? null : parseFloat(d.Long);
            });
            
            recommender = new HybridRecommender(data);
            currentStats = recommender.getStats();
            populateFilters();
        })
        .catch(err => {
            console.error('Failed to load data:', err);
            document.getElementById('results-container').innerHTML = 
                '<div class="no-results">Gagal memuat data destinasi. Silakan refresh halaman.</div>';
        });
}

function populateFilters() {
    const citySel = document.getElementById('city');
    const catSel = document.getElementById('category');
    
    citySel.innerHTML = '<option value="Semua">Semua</option>' +
        currentStats.cities.map(c => `<option value="${c}">${c}</option>`).join('');
    catSel.innerHTML = '<option value="Semua">Semua</option>' +
        currentStats.categories.map(c => `<option value="${c}">${c}</option>`).join('');
    
    // Update stat cards
    document.getElementById('stat-total').textContent = currentStats.total_destinations;
    document.getElementById('stat-cities').textContent = currentStats.cities.length;
    document.getElementById('stat-categories').textContent = currentStats.categories.length;
    document.getElementById('stat-rating').textContent = currentStats.avg_rating.toFixed(1);
}

function getLocation() {
    return new Promise((resolve) => {
        if (!navigator.geolocation) { resolve(null); return; }
        navigator.geolocation.getCurrentPosition(
            pos => resolve({ lat: pos.coords.latitude, lng: pos.coords.longitude }),
            () => resolve(null),
            { enableHighAccuracy: true, timeout: 8000 }
        );
    });
}

async function getRecommendations() {
    const city = document.getElementById('city').value;
    const category = document.getElementById('category').value;
    const minPrice = parseFloat(document.getElementById('min_price').value) || 0;
    const maxPrice = parseFloat(document.getElementById('max_price').value) || 0;
    const maxTime = parseFloat(document.getElementById('max_time').value) || 0;
    const topN = parseInt(document.getElementById('top_n').value);
    
    const container = document.getElementById('results-container');
    container.innerHTML = '<div class="loading">Memproses rekomendasi...</div>';
    
    const loc = await getLocation();
    
    if (!recommender) {
        container.innerHTML = '<div class="no-results">Memuat data...</div>';
        return;
    }
    
    const { recs, error } = recommender.recommend(city, category, minPrice || null, maxPrice || null, maxTime || null, topN);
    
    if (error) {
        container.innerHTML = `<div class="no-results">${error}</div>`;
        return;
    }
    
    // Calculate distances if location available
    const results = recs.map(r => {
        let distance = null;
        if (loc && r.Lat !== null && r.Long !== null) {
            try { distance = haversineKm(loc.lat, loc.lng, r.Lat, r.Long); } catch {}
        }
        return { ...r, distance };
    });
    
    displayResults(results);
}

async function getPopular() {
    const city = document.getElementById('city').value;
    const container = document.getElementById('results-container');
    container.innerHTML = '<div class="loading">Memuat destinasi populer...</div>';
    
    const loc = await getLocation();
    
    if (!recommender) {
        container.innerHTML = '<div class="no-results">Memuat data...</div>';
        return;
    }
    
    const results = recommender.getPopular(10, city).map(r => {
        let distance = null;
        if (loc && r.Lat !== null && r.Long !== null) {
            try { distance = haversineKm(loc.lat, loc.lng, r.Lat, r.Long); } catch {}
        }
        return { ...r, distance };
    });
    
    displayResults(results);
}

function getSimilar() {
    const placeName = document.getElementById('similar-input').value.trim();
    if (!placeName) {
        alert('Masukkan nama destinasi');
        return;
    }
    
    const container = document.getElementById('results-container');
    container.innerHTML = '<div class="loading">Mencari destinasi serupa...</div>';
    
    if (!recommender) {
        container.innerHTML = '<div class="no-results">Memuat data...</div>';
        return;
    }
    
    const { sims, error } = recommender.getSimilarDestinations(placeName, 5);
    
    if (error) {
        container.innerHTML = `<div class="no-results">${error}</div>`;
        return;
    }
    
    displayResults(sims);
}

function displayResults(results) {
    const container = document.getElementById('results-container');
    const countElement = document.getElementById('results-count');
    
    if (!results || results.length === 0) {
        container.innerHTML = '<div class="no-results">Tidak ada destinasi yang ditemukan</div>';
        countElement.textContent = '0 hasil';
        return;
    }
    
    countElement.textContent = results.length + ' hasil';
    
    let html = '<div class="destination-grid">';
    
    results.forEach((dest, index) => {
        const price = dest.price.toLocaleString('id-ID');
        const truncatedDesc = dest.description && dest.description.length > 120 
            ? dest.description.substring(0, 120) + '...' 
            : (dest.description || '');
        
        html += `
            <div class="destination-card">
                <div class="destination-name">${index + 1}. ${dest.name}</div>
                <div class="destination-meta">
                    <span class="badge badge-city">${dest.city}</span>
                    <span class="badge badge-category">${dest.category}</span>
                </div>
                ${truncatedDesc ? `<p style="font-size:0.8rem;color:#777;margin:8px 0;line-height:1.5;">${truncatedDesc}</p>` : ''}
                <div class="destination-details">
                    <div class="detail-row">
                        <span class="detail-label">Rating</span>
                        <span class="detail-value rating">${dest.rating.toFixed(1)} (${dest.rating_count} ulasan)</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Harga</span>
                        <span class="detail-value price">Rp ${price}</span>
                    </div>
                    ${dest.distance != null ? `
                    <div class="detail-row">
                        <span class="detail-label">Jarak</span>
                        <span class="detail-value">${dest.distance} km</span>
                    </div>
                    ` : ''}
                    <div class="detail-row">
                        <span class="detail-label">Durasi</span>
                        <span class="detail-value">${dest.time} menit</span>
                    </div>
                    ${dest.similarity != null ? `
                    <div class="detail-row">
                        <span class="detail-label">Kesamaan</span>
                        <span class="detail-value">${(dest.similarity * 100).toFixed(0)}%</span>
                    </div>
                    ` : ''}
                    ${dest.score != null ? `
                    <div class="detail-row">
                        <span class="detail-label">Skor</span>
                        <span class="detail-value">${dest.score.toFixed(4)}</span>
                    </div>
                    ` : ''}
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    container.innerHTML = html;
}

function clearFilters() {
    document.getElementById('city').value = 'Semua';
    document.getElementById('category').value = 'Semua';
    document.getElementById('min_price').value = '';
    document.getElementById('max_price').value = '';
    document.getElementById('max_time').value = '';
    document.getElementById('top_n').value = '10';
    
    document.getElementById('results-count').textContent = '';
    
    document.getElementById('results-container').innerHTML = `
        <div class="no-results">
            <p class="welcome-text">Selamat datang di Sistem Rekomendasi Wisata.</p>
            <p>Gunakan filter di atas untuk menemukan destinasi wisata yang sesuai dengan preferensi Anda.</p>
            <p>Sistem mengombinasikan kualitas destinasi dan kesamaan konten untuk memberikan rekomendasi terbaik.</p>
            <div class="tips-box">
                <strong>Tips:</strong> Biarkan filter harga dan waktu kosong untuk melihat semua rekomendasi di kategori yang dipilih.
            </div>
        </div>
    `;
}

// Smooth scroll for anchor links
document.addEventListener('DOMContentLoaded', () => {
    initApp();
    
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
    });
});