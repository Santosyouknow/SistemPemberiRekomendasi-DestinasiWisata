from recommendation_system import HybridRecommender

r = HybridRecommender()
print('=== SYSTEM VERIFICATION ===')
print('Total destinations:', r.get_stats()['total_destinations'])

print('\n1. Hybrid Recommendations (Jakarta, Budaya, <50k):')
recs, _ = r.recommend(city='Jakarta', category='Budaya', max_price=50000, top_n=5)
for _, row in recs.iterrows():
    print('  -', row['Place_Name'], '(Rating:', row['Rating'], ', Price: Rp', int(row['Price']), ')')

print('\n2. Similar to Candi Prambanan:')
sims, _ = r.get_similar_destinations('Candi Prambanan', top_n=3)
for _, row in sims.iterrows():
    print('  -', row['Place_Name'], '(' + str(round(row['Similarity'], 3)) + ')')

print('\n3. Popular in Yogyakarta:')
pop = r.get_popular(top_n=3, city='Yogyakarta')
for _, row in pop.iterrows():
    print('  -', row['Place_Name'], '(Rating:', row['Rating'], ')')

print('\n=== VERIFICATION COMPLETE ===')