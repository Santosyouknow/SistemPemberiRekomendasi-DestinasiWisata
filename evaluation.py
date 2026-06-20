"""
Evaluation Metrics for Hybrid Recommendation System
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from recommendation_system import HybridRecommender

class RecommenderEvaluator:
    def __init__(self, recommender):
        self.recommender = recommender
        self.df = recommender.df.copy()
        
    def precision_at_k(self, recommendations, relevant_items, k=10):
        """
        Precision@K: Proportion of recommended items that are relevant
        """
        if len(recommendations) == 0:
            return 0.0
        
        recs_at_k = recommendations[:k]
        relevant_in_recs = len(set(recs_at_k) & set(relevant_items))
        return relevant_in_recs / min(k, len(recs_at_k))
    
    def recall_at_k(self, recommendations, relevant_items, k=10):
        """
        Recall@K: Proportion of relevant items that are recommended
        """
        if len(relevant_items) == 0:
            return 0.0
        
        recs_at_k = recommendations[:k]
        relevant_in_recs = len(set(recs_at_k) & set(relevant_items))
        return relevant_in_recs / len(relevant_items)
    
    def f1_score(self, precision, recall):
        """F1-Score: Harmonic mean of precision and recall"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def mean_average_precision(self, recommendations, relevant_items, k=10):
        """
        MAP@K: Mean Average Precision at K
        """
        if len(recommendations) == 0 or len(relevant_items) == 0:
            return 0.0
        
        recs_at_k = recommendations[:k]
        precisions = []
        num_relevant_found = 0
        
        for i, item in enumerate(recs_at_k):
            if item in relevant_items:
                num_relevant_found += 1
                precision_at_i = num_relevant_found / (i + 1)
                precisions.append(precision_at_i)
        
        if len(precisions) == 0:
            return 0.0
        
        return np.mean(precisions)

    def ndcg_at_k(self, recommendations, relevant_items, k=10):
        """
        NDCG@K: Ranking quality based on relevant items near the top.
        Uses binary relevance because the dataset has no explicit user ratings.
        """
        if len(recommendations) == 0 or len(relevant_items) == 0:
            return 0.0

        recs_at_k = recommendations[:k]
        relevance = [1 if item in relevant_items else 0 for item in recs_at_k]

        dcg = sum(rel / np.log2(rank + 2) for rank, rel in enumerate(relevance))
        ideal_relevance = [1] * min(len(relevant_items), k)
        idcg = sum(rel / np.log2(rank + 2) for rank, rel in enumerate(ideal_relevance))

        return dcg / idcg if idcg > 0 else 0.0
    
    def hit_rate(self, recommendations, relevant_items, k=10):
        """
        Hit Rate@K: Whether at least one relevant item is in top-K
        """
        recs_at_k = recommendations[:k]
        return 1.0 if len(set(recs_at_k) & set(relevant_items)) > 0 else 0.0
    
    def coverage(self, all_recommendations, total_items):
        """
        Coverage: Proportion of items that can be recommended
        """
        unique_recommended = set()
        for recs in all_recommendations:
            unique_recommended.update(recs)
        return len(unique_recommended) / total_items
    
    def diversity(self, recommendations):
        """
        Diversity: Average pairwise similarity between recommended items
        Lower is more diverse
        """
        if len(recommendations) < 2:
            return 0.0
        
        indices = [self.recommender.df.index[self.recommender.df['Place_Id'] == pid].tolist()[0] 
                   for pid in recommendations]
        
        similarities = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                sim = self.recommender.cosine_sim[indices[i]][indices[j]]
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def novelty(self, recommendations):
        """
        Novelty: Based on popularity (inverse of rating count)
        Higher = more novel/less popular items
        """
        if len(recommendations) == 0:
            return 0.0
        
        novelty_scores = []
        for pid in recommendations:
            row = self.df[self.df['Place_Id'] == pid].iloc[0]
            # Use inverse of rating count as novelty measure
            popularity = row['Rating_Count'] if row['Rating_Count'] > 0 else 1
            novelty = 1 / (1 + np.log1p(popularity))
            novelty_scores.append(novelty)
        
        return np.mean(novelty_scores)
    
    def evaluate_recommendations(self, test_cases, k=10):
        """
        Evaluate the recommender on multiple test cases
        
        test_cases: list of dict with keys:
            - city, category, max_price, max_time
            - relevant_items: list of Place_Id that are considered relevant
        """
        results = {
            'precision': [],
            'recall': [],
            'f1': [],
            'map': [],
            'ndcg': [],
            'hit_rate': [],
            'diversity': [],
            'novelty': [],
            'scenario_details': []
        }
        
        all_recs = []
        
        for test in test_cases:
            recs, _ = self.recommender.recommend(
                city=test.get('city'),
                category=test.get('category'),
                max_price=test.get('max_price'),
                max_time=test.get('max_time'),
                top_n=k
            )
            
            rec_ids = recs['Place_Id'].tolist() if len(recs) > 0 else []
            relevant = test.get('relevant_items', [])
            
            all_recs.append(rec_ids)
            
            # Calculate metrics
            precision = self.precision_at_k(rec_ids, relevant, k)
            recall = self.recall_at_k(rec_ids, relevant, k)
            f1 = self.f1_score(precision, recall)
            map_score = self.mean_average_precision(rec_ids, relevant, k)
            ndcg = self.ndcg_at_k(rec_ids, relevant, k)
            hit = self.hit_rate(rec_ids, relevant, k)
            div = self.diversity(rec_ids)
            nov = self.novelty(rec_ids)
            
            results['precision'].append(precision)
            results['recall'].append(recall)
            results['f1'].append(f1)
            results['map'].append(map_score)
            results['ndcg'].append(ndcg)
            results['hit_rate'].append(hit)
            results['diversity'].append(div)
            results['novelty'].append(nov)
            results['scenario_details'].append({
                'name': test.get('name', f"{test.get('city', 'Semua')} - {test.get('category', 'Semua')}"),
                'precision': precision,
                'recall': recall,
                'ndcg': ndcg,
                'hit_rate': hit,
                'recommendation_count': len(rec_ids),
                'relevant_count': len(relevant)
            })
        
        # Aggregate results
        aggregated = {
            'precision@{}'.format(k): np.mean(results['precision']),
            'recall@{}'.format(k): np.mean(results['recall']),
            'f1@{}'.format(k): np.mean(results['f1']),
            'map@{}'.format(k): np.mean(results['map']),
            'ndcg@{}'.format(k): np.mean(results['ndcg']),
            'hit_rate@{}'.format(k): np.mean(results['hit_rate']),
            'diversity': np.mean(results['diversity']),
            'novelty': np.mean(results['novelty']),
            'coverage': self.coverage(all_recs, len(self.df)),
            'scenario_details': results['scenario_details']
        }
        
        return aggregated
    
    def generate_test_cases(self, n_cases=20):
        """
        Generate synthetic test cases from the dataset
        """
        test_cases = []
        
        # Get unique combinations of city and category
        city_cat_combos = self.df.groupby(['City', 'Category']).size().reset_index().head(n_cases)
        
        for _, row in city_cat_combos.iterrows():
            city = row['City']
            category = row['Category']
            
            # Get items in this city-category
            items = self.df[(self.df['City'] == city) & (self.df['Category'] == category)]
            
            if len(items) < 3:
                continue
            
            # Use top-rated items as "relevant"
            relevant_threshold = items['Rating'].quantile(0.7)
            relevant_items = items[items['Rating'] >= relevant_threshold]['Place_Id'].tolist()
            
            # Get price/time constraints from data
            max_price = items['Price'].quantile(0.8)
            max_time = items['Time_Minutes'].quantile(0.8)
            
            test_cases.append({
                'city': city,
                'category': category,
                'max_price': max_price if max_price > 0 else None,
                'max_time': max_time if max_time > 0 else None,
                'relevant_items': relevant_items[:10]  # Top 10 as relevant
            })
        
        return test_cases[:n_cases]

    def generate_scenario_test_cases(self):
        """
        Build explicit business scenarios from common user constraints.
        Relevant items are derived from the same constraints and high rating.
        """
        scenario_specs = [
            {
                'name': 'Wisata budaya murah di Jakarta',
                'city': 'Jakarta',
                'category': 'Budaya',
                'max_price': 50000,
                'max_time': None
            },
            {
                'name': 'Taman hiburan keluarga di Yogyakarta',
                'city': 'Yogyakarta',
                'category': 'Taman Hiburan',
                'max_price': 100000,
                'max_time': 180
            },
            {
                'name': 'Wisata alam di Bandung dengan durasi singkat',
                'city': 'Bandung',
                'category': 'Cagar Alam',
                'max_price': 75000,
                'max_time': 120
            },
            {
                'name': 'Destinasi bahari di Surabaya',
                'city': 'Surabaya',
                'category': 'Bahari',
                'max_price': None,
                'max_time': None
            },
            {
                'name': 'Tempat ibadah populer lintas kota',
                'city': None,
                'category': 'Tempat Ibadah',
                'max_price': None,
                'max_time': None
            }
        ]

        test_cases = []
        for spec in scenario_specs:
            items = self.df.copy()
            if spec['city']:
                items = items[items['City'].str.lower() == spec['city'].lower()]
            if spec['category']:
                items = items[items['Category'].str.lower() == spec['category'].lower()]
            if spec['max_price']:
                items = items[items['Price'] <= spec['max_price']]
            if spec['max_time']:
                items = items[items['Time_Minutes'] <= spec['max_time']]

            if len(items) == 0:
                continue

            threshold = items['Rating'].quantile(0.7)
            relevant_items = items[items['Rating'] >= threshold] \
                .sort_values(['Rating', 'Rating_Count'], ascending=False)['Place_Id'] \
                .head(10) \
                .tolist()

            test_cases.append({
                **spec,
                'relevant_items': relevant_items
            })

        return test_cases
    
    def full_evaluation(self, k_values=[5, 10, 20]):
        """
        Run full evaluation with multiple K values
        """
        print("="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        # Generate test cases
        test_cases = self.generate_test_cases(n_cases=20)
        print(f"\nEvaluating on {len(test_cases)} test cases...")
        
        all_metrics = {}
        
        for k in k_values:
            print(f"\n--- Metrics @ K={k} ---")
            metrics = self.evaluate_recommendations(test_cases, k=k)
            all_metrics[k] = metrics
            
            for metric, value in metrics.items():
                if metric == 'scenario_details':
                    continue
                print(f"{metric:20s}: {value:.4f}")
        
        # Calculate average across all K
        print("\n--- Summary ---")
        avg_precision = np.mean([all_metrics[k]['precision@{}'.format(k)] for k in k_values])
        avg_recall = np.mean([all_metrics[k]['recall@{}'.format(k)] for k in k_values])
        avg_f1 = np.mean([all_metrics[k]['f1@{}'.format(k)] for k in k_values])
        avg_ndcg = np.mean([all_metrics[k]['ndcg@{}'.format(k)] for k in k_values])
        
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print(f"Average F1-Score: {avg_f1:.4f}")
        print(f"Average NDCG: {avg_ndcg:.4f}")
        print(f"Coverage: {all_metrics[k_values[0]]['coverage']:.4f}")
        print(f"Diversity: {all_metrics[k_values[0]]['diversity']:.4f}")
        print(f"Novelty: {all_metrics[k_values[0]]['novelty']:.4f}")

        print("\n--- Scenario-based Evaluation @ K=10 ---")
        scenario_cases = self.generate_scenario_test_cases()
        scenario_metrics = self.evaluate_recommendations(scenario_cases, k=10)
        for detail in scenario_metrics['scenario_details']:
            print(
                f"{detail['name']}: "
                f"Precision={detail['precision']:.4f}, "
                f"Recall={detail['recall']:.4f}, "
                f"NDCG={detail['ndcg']:.4f}, "
                f"HitRate={detail['hit_rate']:.4f}"
            )
        
        return all_metrics


if __name__ == "__main__":
    # Load recommender and evaluate
    print("Loading recommendation system...")
    recommender = HybridRecommender()
    
    evaluator = RecommenderEvaluator(recommender)
    metrics = evaluator.full_evaluation(k_values=[5, 10, 20])
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
