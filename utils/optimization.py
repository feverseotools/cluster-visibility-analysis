"""
Optimization utilities for SEO analysis.

This module provides functions for optimizing API usage, reducing costs,
and improving efficiency in SEO analysis.
"""

import pandas as pd
import math
from typing import Dict, Any, List, Optional, Union
import numpy as np

# Try to import global configuration
try:
    from config_manager import get_config
    config = get_config()
except ImportError:
    config = {}


def calculate_api_cost(num_keywords: int, api_plan: str = "basic", 
                     batch_optimization: bool = True, sampling_rate: float = 1.0) -> Dict[str, Any]:
    """
    Calculate estimated SerpAPI cost based on number of queries with additional optimizations
    
    Args:
        num_keywords: Number of keywords to analyze
        api_plan: SerpAPI plan name (basic, business, enterprise)
        batch_optimization: Whether to apply batch optimization
        sampling_rate: Rate of sampling (1.0 = all keywords, 0.5 = half of keywords)
    
    Returns:
        Dictionary with cost details
    """
    # Get SerpAPI pricing tiers from config or use defaults
    pricing_config = config.get('api', {}).get('serpapi', {}).get('plans', {})
    
    if pricing_config:
        pricing = pricing_config
    else:
        # Default SerpAPI pricing tiers (as of March 2025)
        pricing = {
            "basic": {"monthly_cost": 50, "searches": 5000},
            "business": {"monthly_cost": 250, "searches": 30000},
            "enterprise": {"monthly_cost": 500, "searches": 70000}
        }
    
    # Select pricing tier
    plan = pricing.get(api_plan.lower(), pricing["basic"])
    
    # Apply optimizations
    effective_queries = num_keywords
    
    # Apply sampling rate reduction (if < 1.0)
    effective_queries = math.ceil(effective_queries * sampling_rate)
    
    # Apply batch optimization if enabled (reduces number of required API calls)
    if batch_optimization and effective_queries > 10:
        # Get batch optimization factor from config or use default 5% reduction
        batch_factor = config.get('optimization', {}).get('batching', {}).get('factor', 0.95)
        effective_queries = math.ceil(effective_queries * batch_factor)
    
    # Calculate cost per search
    cost_per_search = plan["monthly_cost"] / plan["searches"]
    
    # Calculate estimated cost
    estimated_cost = effective_queries * cost_per_search
    
    # Calculate percentage of monthly quota
    quota_percentage = (effective_queries / plan["searches"]) * 100
    
    return {
        "num_queries": effective_queries,
        "original_queries": num_keywords,
        "reduction_percentage": round((1 - (effective_queries / num_keywords)) * 100, 2) if num_keywords > 0 else 0,
        "estimated_cost": round(estimated_cost, 2),
        "quota_percentage": round(quota_percentage, 2),
        "plan_details": f"{plan['monthly_cost']}$ for {plan['searches']} searches"
    }


def cluster_representative_keywords(keywords_df: pd.DataFrame, max_representatives: int = 100, 
                                  advanced_sampling: bool = True) -> pd.DataFrame:
    """
    Select representative keywords from each cluster to reduce API costs
    
    This returns a subset of keywords that represent each cluster with improved sampling
    
    Args:
        keywords_df: DataFrame with keywords, must contain 'cluster_name' and 'Avg. monthly searches' columns
        max_representatives: Maximum number of keywords to return
        advanced_sampling: Whether to use advanced sampling techniques
    
    Returns:
        DataFrame with representative keywords
    """
    if keywords_df.empty or 'cluster_name' not in keywords_df.columns:
        return keywords_df
    
    # Group by cluster
    grouped = keywords_df.groupby('cluster_name')
    
    # Get sampling configuration from config
    sampling_config = config.get('optimization', {}).get('sampling', {})
    
    # IMPROVEMENT: Use advanced sampling if enabled
    if advanced_sampling:
        # Calculate diversity of each cluster (by search volume)
        cluster_diversities = {}
        
        for cluster_name, group in grouped:
            # Calculate volume range as a measure of diversity
            if len(group) > 1:
                vol_range = group['Avg. monthly searches'].max() - group['Avg. monthly searches'].min()
                vol_std = group['Avg. monthly searches'].std()
                # Combined diversity score
                cluster_diversities[cluster_name] = (vol_range + vol_std) / 2
            else:
                cluster_diversities[cluster_name] = 0
        
        # Normalize diversities
        total_diversity = sum(cluster_diversities.values())
        if total_diversity > 0:
            for cluster in cluster_diversities:
                cluster_diversities[cluster] /= total_diversity
    
    # Calculate how many representatives to take from each cluster
    # Use proportional distribution based on cluster size and search volume
    cluster_sizes = grouped.size()
    cluster_volumes = grouped['Avg. monthly searches'].sum()
    
    # Get weighting factors from config or use defaults
    if advanced_sampling and sampling_config:
        weighting = sampling_config.get('weighting', {})
        size_weight = weighting.get('cluster_size', 0.3)
        volume_weight = weighting.get('search_volume', 0.5)
        diversity_weight = weighting.get('diversity', 0.2)
    else:
        size_weight = 0.3
        volume_weight = 0.5
        diversity_weight = 0.2
    
    # Create a score combining size and volume
    if advanced_sampling:
        # Include diversity in the score
        cluster_importance = (
            size_weight * (cluster_sizes / cluster_sizes.sum()) + 
            volume_weight * (cluster_volumes / cluster_volumes.sum()) +
            diversity_weight * pd.Series(cluster_diversities)
        )
    else:
        cluster_importance = (
            0.4 * (cluster_sizes / cluster_sizes.sum()) + 
            0.6 * (cluster_volumes / cluster_volumes.sum())
        )
    
    # Calculate representatives per cluster (minimum 1)
    reps_per_cluster = (cluster_importance * max_representatives).apply(lambda x: max(1, round(x)))
    
    # Ensure we don't exceed max_representatives
    while reps_per_cluster.sum() > max_representatives:
        # Find cluster with most representatives
        max_cluster = reps_per_cluster.idxmax()
        # Reduce by 1
        reps_per_cluster[max_cluster] -= 1
    
    # Select representatives
    selected_keywords = []
    
    for cluster, count in reps_per_cluster.items():
        # Get cluster data
        cluster_data = keywords_df[keywords_df['cluster_name'] == cluster]
        
        # IMPROVEMENT: Select keywords with better volume distribution
        if advanced_sampling and len(cluster_data) > count:
            # Divide the volume range into segments and select from each one
            sorted_data = cluster_data.sort_values('Avg. monthly searches')
            if len(sorted_data) > 1:
                step = len(sorted_data) / count
                
                indices = [int(i * step) for i in range(count)]
                # Ensure indices are within bounds
                indices = [min(i, len(sorted_data) - 1) for i in indices]
                selected_keywords.append(sorted_data.iloc[indices])
            else:
                selected_keywords.append(sorted_data)
        else:
            # Sort by search volume (descending) - original method
            sorted_data = cluster_data.sort_values('Avg. monthly searches', ascending=False)
            # Take top keywords as representatives
            selected_keywords.append(sorted_data.head(int(count)))
    
    # Combine all selected keywords
    representative_df = pd.concat(selected_keywords)
    
    return representative_df


def group_similar_keywords(keywords_df: pd.DataFrame, similarity_threshold: float = 0.8) -> pd.DataFrame:
    """
    Group very similar keywords to reduce queries
    
    Uses TF-IDF and cosine similarity to identify similar keywords
    
    Args:
        keywords_df: DataFrame with keywords
        similarity_threshold: Threshold for considering keywords similar (0-1)
    
    Returns:
        DataFrame with similar keywords grouped
    """
    # If few keywords, no need to group
    if len(keywords_df) < 100:
        return keywords_df
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        # If sklearn not available, return original DataFrame
        return keywords_df
    
    # Vectorize keywords
    vectorizer = TfidfVectorizer()
    try:
        X = vectorizer.fit_transform(keywords_df['keyword'])
        
        # Calculate similarity
        similarity_matrix = cosine_similarity(X)
        
        # Identify groups
        groups = {}
        processed = set()
        
        for i in range(len(keywords_df)):
            if i in processed:
                continue
                
            similar_indices = [j for j in range(len(keywords_df)) 
                              if similarity_matrix[i, j] > similarity_threshold and j not in processed]
            
            if similar_indices:
                # Select the keyword with highest volume as representative
                group_df = keywords_df.iloc[[i] + similar_indices]
                representative_idx = group_df['Avg. monthly searches'].idxmax()
                
                processed.update([i] + similar_indices)
                
                # Add to group
                groups[representative_idx] = [i] + similar_indices
        
        # Create new DataFrame with only representatives
        representative_indices = list(groups.keys()) + [i for i in range(len(keywords_df)) if i not in processed]
        return keywords_df.iloc[representative_indices].copy()
    
    except Exception as e:
        # If grouping fails, return the original DataFrame
        import logging
        logging.warning(f"Error grouping similar keywords: {str(e)}")
        return keywords_df


def optimize_batch_sizes(keywords_df: pd.DataFrame) -> Dict[str, int]:
    """
    Determine optimal batch sizes for processing based on cluster characteristics
    
    Args:
        keywords_df: DataFrame with keywords
    
    Returns:
        Dictionary mapping cluster names to optimal batch sizes
    """
    # Default batch size
    default_batch_size = config.get('optimization', {}).get('batching', {}).get('min_batch_size', 3)
    max_batch_size = config.get('optimization', {}).get('batching', {}).get('max_batch_size', 10)
    
    # If no cluster information, return default
    if 'cluster_name' not in keywords_df.columns:
        return {'default': default_batch_size}
    
    # Calculate cluster sizes
    cluster_sizes = keywords_df.groupby('cluster_name').size()
    avg_cluster_size = cluster_sizes.mean()
    
    # Determine optimal batch size for each cluster
    batch_sizes = {}
    
    for cluster, size in cluster_sizes.items():
        # Smaller clusters can have larger batch sizes
        if size < avg_cluster_size / 2:
            batch_sizes[cluster] = min(max_batch_size, max(default_batch_size, int(10 / (size / avg_cluster_size))))
        # Medium clusters use default
        elif size <= avg_cluster_size * 2:
            batch_sizes[cluster] = default_batch_size
        # Large clusters use smaller batch sizes
        else:
            batch_sizes[cluster] = max(default_batch_size - 1, 2)
    
    return batch_sizes


def estimate_processing_time(keywords_df: pd.DataFrame, optimization_settings: Optional[Dict[str, Any]] = None) -> float:
    """
    Estimate processing time based on number of keywords and optimization settings
    
    Args:
        keywords_df: DataFrame with keywords
        optimization_settings: Dictionary with optimization settings
    
    Returns:
        Estimated processing time in seconds
    """
    if optimization_settings is None:
        optimization_settings = {
            "use_representatives": True,
            "advanced_sampling": True, 
            "batch_optimization": True,
            "max_keywords": min(100, len(keywords_df) // 2)
        }
    
    # Time per query with different optimizations
    base_time_per_query = 1.5  # seconds per query
    
    # Calculate effective query count
    if optimization_settings["use_representatives"]:
        sample_df = cluster_representative_keywords(
            keywords_df,
            optimization_settings["max_keywords"],
            advanced_sampling=optimization_settings["advanced_sampling"]
        )
        num_queries = len(sample_df)
    else:
        num_queries = len(keywords_df)
    
    # Adjust time based on batch optimization
    if optimization_settings["batch_optimization"]:
        # Estimate batch size
        avg_batch_size = 5
        return (num_queries / avg_batch_size) * base_time_per_query * 1.2  # 20% overhead
    else:
        return num_queries * base_time_per_query
