"""
Feature engineering utilities for software effort estimation
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression


class FeatureEngineer:
    """Feature engineering for software effort estimation datasets"""
    
    def __init__(self):
        self.feature_names = None
        self.selected_features = None
        self.poly_features = None
        
    def create_interaction_features(self, X: np.ndarray, 
                                     degree: int = 2,
                                     include_bias: bool = False) -> np.ndarray:
        """
        Create polynomial and interaction features
        
        Args:
            X: Input features
            degree: Polynomial degree
            include_bias: Whether to include bias term
            
        Returns:
            Extended feature matrix
        """
        self.poly_features = PolynomialFeatures(
            degree=degree, 
            include_bias=include_bias,
            interaction_only=True
        )
        return self.poly_features.fit_transform(X)
    
    def create_eaf_feature(self, X: np.ndarray, 
                           cost_driver_indices: List[int] = None) -> np.ndarray:
        """
        Create Effort Adjustment Factor (EAF) feature
        
        EAF = product of all cost driver multipliers
        
        Args:
            X: Input features
            cost_driver_indices: Indices of cost driver columns
            
        Returns:
            Feature matrix with EAF appended
        """
        if cost_driver_indices is None:
            # Assume first 15 columns are cost drivers (COCOMO format)
            cost_driver_indices = list(range(min(15, X.shape[1] - 1)))
        
        eaf = np.prod(X[:, cost_driver_indices], axis=1, keepdims=True)
        return np.hstack([X, eaf])
    
    def create_size_derived_features(self, X: np.ndarray,
                                      loc_index: int = -1) -> np.ndarray:
        """
        Create features derived from size (LOC)
        
        Args:
            X: Input features
            loc_index: Index of LOC column
            
        Returns:
            Feature matrix with derived features appended
        """
        loc = X[:, loc_index]
        
        # Create derived features
        log_loc = np.log1p(loc).reshape(-1, 1)
        sqrt_loc = np.sqrt(loc).reshape(-1, 1)
        loc_squared = (loc ** 2).reshape(-1, 1)
        
        return np.hstack([X, log_loc, sqrt_loc, loc_squared])
    
    def select_features_kbest(self, X: np.ndarray, y: np.ndarray,
                               k: int = 10,
                               score_func=f_regression) -> Tuple[np.ndarray, List[int]]:
        """
        Select k best features using univariate statistical tests
        
        Args:
            X: Input features
            y: Target values
            k: Number of features to select
            score_func: Scoring function
            
        Returns:
            Tuple of (selected features, selected indices)
        """
        selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        selected_indices = selector.get_support(indices=True).tolist()
        self.selected_features = selected_indices
        
        return X_selected, selected_indices
    
    def select_features_mutual_info(self, X: np.ndarray, y: np.ndarray,
                                     k: int = 10) -> Tuple[np.ndarray, List[int]]:
        """
        Select k best features using mutual information
        
        Args:
            X: Input features
            y: Target values
            k: Number of features to select
            
        Returns:
            Tuple of (selected features, selected indices)
        """
        return self.select_features_kbest(X, y, k, score_func=mutual_info_regression)
    
    def get_feature_importance(self, X: np.ndarray, y: np.ndarray,
                                feature_names: List[str] = None) -> pd.DataFrame:
        """
        Calculate feature importance using multiple methods
        
        Args:
            X: Input features
            y: Target values
            feature_names: Names of features
            
        Returns:
            DataFrame with feature importance scores
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # F-regression scores
        f_scores, f_pvalues = f_regression(X, y)
        
        # Mutual information
        mi_scores = mutual_info_regression(X, y)
        
        # Correlation with target
        correlations = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'F_Score': f_scores,
            'F_PValue': f_pvalues,
            'MI_Score': mi_scores,
            'Correlation': correlations,
            'Abs_Correlation': np.abs(correlations)
        })
        
        # Rank features
        importance_df['F_Rank'] = importance_df['F_Score'].rank(ascending=False)
        importance_df['MI_Rank'] = importance_df['MI_Score'].rank(ascending=False)
        importance_df['Corr_Rank'] = importance_df['Abs_Correlation'].rank(ascending=False)
        importance_df['Avg_Rank'] = (
            importance_df['F_Rank'] + 
            importance_df['MI_Rank'] + 
            importance_df['Corr_Rank']
        ) / 3
        
        return importance_df.sort_values('Avg_Rank')
    
    def create_cocomo_features(self, X: np.ndarray) -> np.ndarray:
        """
        Create COCOMO-specific features
        
        Args:
            X: Input features [cost_drivers, loc]
            
        Returns:
            Extended feature matrix
        """
        # Assuming standard COCOMO format: 15 cost drivers + LOC
        n_cost_drivers = min(15, X.shape[1] - 1)
        
        # EAF
        eaf = np.prod(X[:, :n_cost_drivers], axis=1, keepdims=True)
        
        # LOC
        loc = X[:, -1]
        
        # COCOMO base effort: a * LOC^b (using default values)
        base_effort = 2.94 * (loc ** 1.12)
        base_effort = base_effort.reshape(-1, 1)
        
        # Estimated effort
        estimated_effort = base_effort * eaf
        
        # Log features
        log_loc = np.log1p(loc).reshape(-1, 1)
        log_effort = np.log1p(estimated_effort)
        
        return np.hstack([X, eaf, base_effort, estimated_effort, log_loc, log_effort])
    
    def remove_outliers(self, X: np.ndarray, y: np.ndarray,
                        method: str = "iqr",
                        threshold: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove outliers from dataset
        
        Args:
            X: Features
            y: Target values
            method: 'iqr' or 'zscore'
            threshold: Threshold for outlier detection
            
        Returns:
            Tuple of (X_clean, y_clean)
        """
        if method == "iqr":
            Q1 = np.percentile(y, 25)
            Q3 = np.percentile(y, 75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            mask = (y >= lower) & (y <= upper)
            
        elif method == "zscore":
            from scipy import stats
            z_scores = np.abs(stats.zscore(y))
            mask = z_scores < threshold
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return X[mask], y[mask]


def engineer_features(X: np.ndarray, y: np.ndarray,
                      add_eaf: bool = True,
                      add_size_features: bool = True,
                      add_interactions: bool = False,
                      select_k_best: int = None) -> Tuple[np.ndarray, FeatureEngineer]:
    """
    Complete feature engineering pipeline
    
    Args:
        X: Input features
        y: Target values
        add_eaf: Add EAF feature
        add_size_features: Add size-derived features
        add_interactions: Add interaction features
        select_k_best: Number of best features to select
        
    Returns:
        Tuple of (engineered features, feature engineer object)
    """
    engineer = FeatureEngineer()
    X_new = X.copy()
    
    if add_eaf:
        X_new = engineer.create_eaf_feature(X_new)
    
    if add_size_features:
        X_new = engineer.create_size_derived_features(X_new)
    
    if add_interactions:
        X_new = engineer.create_interaction_features(X_new, degree=2)
    
    if select_k_best:
        X_new, _ = engineer.select_features_kbest(X_new, y, k=select_k_best)
    
    return X_new, engineer


if __name__ == "__main__":
    # Test feature engineering
    from src.data.data_loader import DataLoader
    
    loader = DataLoader("cocomo81")
    X, y = loader.get_features_and_target()
    
    print("Original X shape:", X.shape)
    
    engineer = FeatureEngineer()
    
    # Test EAF feature
    X_eaf = engineer.create_eaf_feature(X)
    print("With EAF:", X_eaf.shape)
    
    # Test size features
    X_size = engineer.create_size_derived_features(X)
    print("With size features:", X_size.shape)
    
    # Test feature importance
    feature_names = loader.get_feature_names()
    importance = engineer.get_feature_importance(X, y, feature_names)
    print("\nTop 5 important features:")
    print(importance.head())