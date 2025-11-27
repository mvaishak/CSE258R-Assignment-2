"""
Recommendation Engine for DiamondHood
Hybrid recommendation system combining feature-based and visual similarity
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import joblib


class DiamondRecommender:
    """
    Hybrid recommendation system for diamonds.
    Combines feature-based similarity with visual embeddings.
    """
    
    def __init__(self, df, embeddings=None, feature_weight=0.5, visual_weight=0.5):
        """
        Initialize recommender system.
        
        Args:
            df: DataFrame with diamond features
            embeddings: Array of visual embeddings (optional)
            feature_weight: Weight for feature-based similarity
            visual_weight: Weight for visual similarity
        """
        self.df = df.copy()
        self.embeddings = embeddings
        self.feature_weight = feature_weight
        self.visual_weight = visual_weight
        self.scaler = StandardScaler()
        
        # Prepare feature matrix
        self.feature_matrix = self._prepare_feature_matrix()
        
    def _prepare_feature_matrix(self):
        """
        Prepare normalized feature matrix for similarity computation.
        
        Returns:
            Scaled feature matrix
        """
        # Select numerical features
        feature_cols = [col for col in self.df.columns 
                       if col not in ['price', 'diamond_id', 'cut', 'color', 'clarity'] 
                       and self.df[col].dtype in ['int64', 'float64']]
        
        X = self.df[feature_cols].values
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled
    
    def compute_feature_similarity(self, query_idx, top_k=10, metric='cosine'):
        """
        Compute feature-based similarity.
        
        Args:
            query_idx: Index of query diamond
            top_k: Number of recommendations
            metric: Similarity metric ('cosine' or 'euclidean')
            
        Returns:
            Indices and scores of similar diamonds
        """
        query_features = self.feature_matrix[query_idx].reshape(1, -1)
        
        if metric == 'cosine':
            similarities = cosine_similarity(query_features, self.feature_matrix)[0]
            # Higher is better for cosine
            similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
            scores = similarities[similar_indices]
        else:  # euclidean
            distances = euclidean_distances(query_features, self.feature_matrix)[0]
            # Lower is better for distance
            similar_indices = np.argsort(distances)[1:top_k+1]
            # Convert to similarity score (0-1 range)
            scores = 1 / (1 + distances[similar_indices])
        
        return similar_indices, scores
    
    def compute_visual_similarity(self, query_idx, top_k=10):
        """
        Compute visual similarity using image embeddings.
        
        Args:
            query_idx: Index of query diamond
            top_k: Number of recommendations
            
        Returns:
            Indices and scores of visually similar diamonds
        """
        if self.embeddings is None:
            raise ValueError("No embeddings provided. Cannot compute visual similarity.")
        
        query_embedding = self.embeddings[query_idx].reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k (excluding self)
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
        scores = similarities[similar_indices]
        
        return similar_indices, scores
    
    def hybrid_recommend(self, query_idx, top_k=10, normalize=True):
        """
        Generate hybrid recommendations combining features and visual similarity.
        
        Args:
            query_idx: Index of query diamond
            top_k: Number of recommendations
            normalize: Whether to normalize scores before combining
            
        Returns:
            DataFrame with recommendations and scores
        """
        # Get feature-based recommendations
        feature_indices, feature_scores = self.compute_feature_similarity(
            query_idx, top_k=top_k*2  # Get more for better coverage
        )
        
        # Initialize visual scores
        visual_scores = np.zeros(len(self.df))
        
        # Get visual similarity if embeddings available
        if self.embeddings is not None:
            visual_indices, vis_scores = self.compute_visual_similarity(
                query_idx, top_k=top_k*2
            )
            visual_scores[visual_indices] = vis_scores
        
        # Normalize scores if requested
        if normalize and self.embeddings is not None:
            feature_scores_norm = (feature_scores - feature_scores.min()) / (feature_scores.max() - feature_scores.min() + 1e-10)
            visual_scores_norm = visual_scores.copy()
            if visual_scores.max() > 0:
                visual_scores_norm = (visual_scores - visual_scores.min()) / (visual_scores.max() - visual_scores.min() + 1e-10)
        else:
            feature_scores_norm = feature_scores
            visual_scores_norm = visual_scores
        
        # Compute hybrid scores for all candidates
        all_candidates = set(feature_indices)
        if self.embeddings is not None:
            all_candidates.update(visual_indices)
        
        hybrid_scores = {}
        for idx in all_candidates:
            feat_score = feature_scores_norm[np.where(feature_indices == idx)[0][0]] if idx in feature_indices else 0
            vis_score = visual_scores_norm[idx] if self.embeddings is not None else 0
            
            hybrid_scores[idx] = (
                self.feature_weight * feat_score + 
                self.visual_weight * vis_score
            )
        
        # Sort by hybrid score
        sorted_candidates = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, score in sorted_candidates[:top_k]]
        top_scores = [score for idx, score in sorted_candidates[:top_k]]
        
        # Create results DataFrame
        results = self.df.iloc[top_indices].copy()
        results['similarity_score'] = top_scores
        results['recommendation_rank'] = range(1, len(top_indices) + 1)
        
        return results
    
    def recommend_by_features(self, query_features, top_k=10):
        """
        Recommend diamonds based on desired features.
        
        Args:
            query_features: Dictionary of feature constraints
                           e.g., {'carat': (1.0, 1.5), 'price': (5000, 7000)}
            top_k: Number of recommendations
            
        Returns:
            DataFrame with matching diamonds
        """
        # Filter by constraints
        filtered_df = self.df.copy()
        
        for feature, (min_val, max_val) in query_features.items():
            if feature in filtered_df.columns:
                filtered_df = filtered_df[
                    (filtered_df[feature] >= min_val) & 
                    (filtered_df[feature] <= max_val)
                ]
        
        # Sort by relevance (e.g., best value)
        if 'price' in filtered_df.columns and 'carat' in filtered_df.columns:
            filtered_df['value_score'] = filtered_df['carat'] / (filtered_df['price'] + 1)
            filtered_df = filtered_df.sort_values('value_score', ascending=False)
        
        return filtered_df.head(top_k)
    
    def recommend_by_budget(self, budget, carat_range=None, top_k=10):
        """
        Recommend best diamonds within a budget.
        
        Args:
            budget: Maximum price
            carat_range: Optional (min_carat, max_carat) tuple
            top_k: Number of recommendations
            
        Returns:
            DataFrame with recommendations
        """
        # Filter by budget
        filtered_df = self.df[self.df['price'] <= budget].copy()
        
        # Filter by carat if specified
        if carat_range:
            min_carat, max_carat = carat_range
            filtered_df = filtered_df[
                (filtered_df['carat'] >= min_carat) & 
                (filtered_df['carat'] <= max_carat)
            ]
        
        # Calculate value score
        # Consider carat, cut quality, clarity
        if 'cut_encoded' in filtered_df.columns:
            filtered_df['quality_score'] = (
                filtered_df['carat'] * 0.4 +
                filtered_df['cut_encoded'] * 0.3 +
                filtered_df['clarity_encoded'] * 0.2 +
                filtered_df['color_encoded'] * 0.1
            )
            filtered_df = filtered_df.sort_values('quality_score', ascending=False)
        else:
            filtered_df = filtered_df.sort_values('carat', ascending=False)
        
        return filtered_df.head(top_k)
    
    def find_similar_by_id(self, diamond_id, top_k=10):
        """
        Find similar diamonds by diamond ID.
        
        Args:
            diamond_id: ID of the query diamond
            top_k: Number of recommendations
            
        Returns:
            DataFrame with recommendations
        """
        # Find index
        query_idx = self.df[self.df['diamond_id'] == diamond_id].index[0]
        
        # Get recommendations
        return self.hybrid_recommend(query_idx, top_k=top_k)
    
    def evaluate_recommendations(self, test_pairs, top_k=10):
        """
        Evaluate recommendation quality using test pairs.
        
        Args:
            test_pairs: List of (query_id, relevant_id) tuples
            top_k: Number of recommendations to generate
            
        Returns:
            Dictionary of evaluation metrics
        """
        precisions = []
        recalls = []
        hits = 0
        
        for query_id, relevant_id in test_pairs:
            # Get recommendations
            try:
                recs = self.find_similar_by_id(query_id, top_k=top_k)
                rec_ids = recs['diamond_id'].values
                
                # Check if relevant item is in recommendations
                if relevant_id in rec_ids:
                    hits += 1
                    precision = 1.0 / (list(rec_ids).index(relevant_id) + 1)
                    precisions.append(precision)
                else:
                    precisions.append(0.0)
                
            except Exception as e:
                print(f"Error processing pair ({query_id}, {relevant_id}): {e}")
                precisions.append(0.0)
        
        # Calculate metrics
        metrics = {
            'precision@k': np.mean(precisions),
            'hit_rate@k': hits / len(test_pairs),
            'mean_reciprocal_rank': np.mean(precisions)
        }
        
        return metrics
    
    def save(self, filepath):
        """
        Save recommender to disk.
        
        Args:
            filepath: Path to save file
        """
        data = {
            'df': self.df,
            'embeddings': self.embeddings,
            'feature_weight': self.feature_weight,
            'visual_weight': self.visual_weight,
            'scaler': self.scaler,
            'feature_matrix': self.feature_matrix
        }
        joblib.dump(data, filepath)
        print(f"✓ Recommender saved: {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load recommender from disk.
        
        Args:
            filepath: Path to load from
            
        Returns:
            DiamondRecommender instance
        """
        data = joblib.load(filepath)
        
        recommender = cls(
            df=data['df'],
            embeddings=data['embeddings'],
            feature_weight=data['feature_weight'],
            visual_weight=data['visual_weight']
        )
        recommender.scaler = data['scaler']
        recommender.feature_matrix = data['feature_matrix']
        
        print(f"✓ Recommender loaded: {filepath}")
        return recommender


if __name__ == "__main__":
    print("DiamondRecommender module loaded successfully")
    print("\nAvailable recommendation methods:")
    print("  - hybrid_recommend: Combine features + visual similarity")
    print("  - recommend_by_features: Filter by feature constraints")
    print("  - recommend_by_budget: Find best value within budget")
    print("  - find_similar_by_id: Find similar diamonds by ID")
