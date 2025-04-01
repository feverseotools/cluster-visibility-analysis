"""
Data processing utilities for SEO analysis.

This module provides functions for handling large datasets, caching results,
and optimizing data processing for SEO analysis.
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
from datetime import datetime
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple, Set
import logging
import streamlit as st

# Try to import global configuration
try:
    from config_manager import get_config
    config = get_config()
except ImportError:
    config = {}


def process_large_dataset(keywords_df: pd.DataFrame, max_per_session: int = 5000) -> List[pd.DataFrame]:
    """
    Divide very large datasets into processable fragments
    
    Args:
        keywords_df: DataFrame with keywords
        max_per_session: Maximum keywords to process in a single session
    
    Returns:
        List of DataFrame fragments
    """
    total_keywords = len(keywords_df)
    
    if total_keywords <= max_per_session:
        return [keywords_df]
    
    # Use max_per_session from config if available
    config_max = config.get('performance', {}).get('max_dataset_size')
    if config_max:
        max_per_session = config_max
    
    # Calculate number of sessions needed
    num_sessions = (total_keywords + max_per_session - 1) // max_per_session
    
    # Divide by clusters to maintain coherence if possible
    if 'cluster_name' in keywords_df.columns:
        clusters = keywords_df['cluster_name'].unique()
        clusters_per_session = (len(clusters) + num_sessions - 1) // num_sessions
        
        fragments = []
        for i in range(0, len(clusters), clusters_per_session):
            session_clusters = clusters[i:i+clusters_per_session]
            fragment = keywords_df[keywords_df['cluster_name'].isin(session_clusters)]
            fragments.append(fragment)
        
        # Handle edge cases where clusters are very uneven
        if len(fragments) > 0:
            # Check if any fragment is too large
            for i, fragment in enumerate(fragments):
                if len(fragment) > max_per_session * 1.5:  # 50% more than max
                    # Split this fragment further by volume
                    sorted_fragment = fragment.sort_values('Avg. monthly searches', ascending=False)
                    sub_fragments = [sorted_fragment.iloc[i:i+max_per_session] 
                                   for i in range(0, len(sorted_fragment), max_per_session)]
                    # Replace the original fragment with sub-fragments
                    fragments = fragments[:i] + sub_fragments + fragments[i+1:]
        
        return fragments
    else:
        # If no clusters, divide by search volume to prioritize important keywords
        sorted_df = keywords_df.sort_values('Avg. monthly searches', ascending=False)
        return [sorted_df.iloc[i:i+max_per_session] for i in range(0, total_keywords, max_per_session)]


def generate_cache_id(data: pd.DataFrame, domains: List[str], params: Dict[str, Any]) -> str:
    """
    Generate a unique cache ID based on input data and parameters
    
    Args:
        data: Input DataFrame
        domains: List of domains to analyze
        params: API parameters
    
    Returns:
        String hash representing the cache ID
    """
    # Create a string representation of the inputs
    domains_str = ",".join(sorted(domains))
    
    # Use only relevant params for the hash
    relevant_params = {k: v for k, v in params.items() 
                     if k in ['engine', 'google_domain', 'hl', 'location']}
    params_str = json.dumps(relevant_params, sort_keys=True)
    
    # For the DataFrame, use a hash of column names and size
    columns_str = ",".join(data.columns)
    data_hash = f"{columns_str}:{len(data)}"
    
    # Combine all components
    combined = f"{data_hash}|{domains_str}|{params_str}"
    
    # Generate hash
    return hashlib.md5(combined.encode()).hexdigest()


def save_results_to_cache(results_df: pd.DataFrame, cache_id: str) -> str:
    """
    Save results to persistent cache
    
    Args:
        results_df: Results DataFrame
        cache_id: Unique cache identifier
    
    Returns:
        Path to cache file
    """
    # Get cache directory from config or use default
    cache_dir = config.get('performance', {}).get('cache_dir', "cache")
    
    # Ensure directory exists
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # Create cache filename
    cache_file = f"{cache_dir}/results_{cache_id}.pkl"
    
    # Save DataFrame
    results_df.to_pickle(cache_file)
    
    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "num_keywords": len(results_df['Keyword'].unique()) if 'Keyword' in results_df.columns else len(results_df),
        "domains": results_df['Domain'].unique().tolist() if 'Domain' in results_df.columns else [],
        "version": "1.0"
    }
    
    with open(f"{cache_dir}/metadata_{cache_id}.json", 'w') as f:
        json.dump(metadata, f)
    
    return cache_file


def load_results_from_cache(cache_id: str) -> Tuple[pd.DataFrame, bool]:
    """
    Load results from persistent cache
    
    Args:
        cache_id: Unique cache identifier
    
    Returns:
        Tuple of (DataFrame, success_flag)
    """
    # Get cache directory from config or use default
    cache_dir = config.get('performance', {}).get('cache_dir', "cache")
    
    # Check for cache file
    cache_file = f"{cache_dir}/results_{cache_id}.pkl"
    
    if os.path.exists(cache_file):
        try:
            # Load DataFrame
            results_df = pd.read_pickle(cache_file)
            return results_df, True
        except Exception as e:
            logging.error(f"Error loading cache file {cache_file}: {str(e)}")
    
    # Return empty DataFrame and False flag if loading failed
    return pd.DataFrame(), False


def get_cache_metadata(cache_id: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Get metadata for cache entries
    
    Args:
        cache_id: Optional specific cache ID, if None returns all metadata
    
    Returns:
        Dictionary or list of dictionaries with metadata
    """
    # Get cache directory from config or use default
    cache_dir = config.get('performance', {}).get('cache_dir', "cache")
    
    if not os.path.exists(cache_dir):
        return [] if cache_id is None else {}
    
    if cache_id is not None:
        # Get metadata for specific cache ID
        metadata_file = f"{cache_dir}/metadata_{cache_id}.json"
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Error loading metadata file {metadata_file}: {str(e)}")
                return {}
        return {}
    else:
        # Get all metadata
        metadata_list = []
        for filename in os.listdir(cache_dir):
            if filename.startswith("metadata_") and filename.endswith(".json"):
                cache_id = filename.replace("metadata_", "").replace(".json", "")
                metadata = get_cache_metadata(cache_id)
                if metadata:
                    metadata['cache_id'] = cache_id
                    metadata_list.append(metadata)
        
        # Sort by timestamp if available
        try:
            metadata_list.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        except:
            pass
            
        return metadata_list


def clean_old_cache(max_age_days: int = 30) -> int:
    """
    Clean cache entries older than specified age
    
    Args:
        max_age_days: Maximum age in days
    
    Returns:
        Number of entries removed
    """
    # Get cache directory from config or use default
    cache_dir = config.get('performance', {}).get('cache_dir', "cache")
    
    if not os.path.exists(cache_dir):
        return 0
    
    # Get all metadata
    all_metadata = get_cache_metadata()
    removed_count = 0
    
    # Current time
    now = datetime.now()
    
    for entry in all_metadata:
        try:
            # Parse timestamp
            timestamp = datetime.fromisoformat(entry.get('timestamp', ''))
            age_days = (now - timestamp).total_seconds() / (86400)  # Convert to days
            
            if age_days > max_age_days:
                # Remove this cache entry
                cache_id = entry.get('cache_id')
                if cache_id:
                    cache_file = f"{cache_dir}/results_{cache_id}.pkl"
                    metadata_file = f"{cache_dir}/metadata_{cache_id}.json"
                    
                    if os.path.exists(cache_file):
                        os.remove(cache_file)
                    
                    if os.path.exists(metadata_file):
                        os.remove(metadata_file)
                    
                    removed_count += 1
        except Exception as e:
            logging.error(f"Error cleaning cache entry: {str(e)}")
    
    return removed_count


def merge_results(fragments: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge results from multiple processing fragments
    
    Args:
        fragments: List of result DataFrames from processing fragments
    
    Returns:
        Combined DataFrame with all results
    """
    if not fragments:
        return pd.DataFrame()
    
    if len(fragments) == 1:
        return fragments[0]
    
    # Check if all fragments have the same structure
    first_columns = set(fragments[0].columns)
    for df in fragments[1:]:
        if set(df.columns) != first_columns:
            # Handle mismatched columns by finding common columns
            common_columns = first_columns.intersection(set(df.columns))
            if not common_columns:
                logging.error("Cannot merge fragments with no common columns")
                return fragments[0]  # Return first fragment as fallback
            
            # Filter all fragments to common columns
            fragments = [df[list(common_columns)] for df in fragments]
            break
    
    # Merge all fragments
    return pd.concat(fragments, ignore_index=True)


def calculate_dataset_statistics(keywords_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate statistics for a keywords dataset
    
    Args:
        keywords_df: DataFrame with keywords
    
    Returns:
        Dictionary with statistics
    """
    stats = {
        "total_keywords": len(keywords_df),
        "total_volume": int(keywords_df['Avg. monthly searches'].sum()) if 'Avg. monthly searches' in keywords_df.columns else 0,
        "avg_volume": float(keywords_df['Avg. monthly searches'].mean()) if 'Avg. monthly searches' in keywords_df.columns else 0,
    }
    
    # Cluster statistics if available
    if 'cluster_name' in keywords_df.columns:
        clusters = keywords_df['cluster_name'].unique()
        stats["num_clusters"] = len(clusters)
        
        # Calculate stats per cluster
        cluster_stats = {}
        for cluster in clusters:
            cluster_df = keywords_df[keywords_df['cluster_name'] == cluster]
            cluster_stats[cluster] = {
                "keywords": len(cluster_df),
                "volume": int(cluster_df['Avg. monthly searches'].sum()) if 'Avg. monthly searches' in cluster_df.columns else 0,
                "percentage": round(len(cluster_df) / len(keywords_df) * 100, 1)
            }
        
        stats["clusters"] = cluster_stats
    
    # Volume distribution
    if 'Avg. monthly searches' in keywords_df.columns:
        volume_counts = {
            "high": len(keywords_df[keywords_df['Avg. monthly searches'] >= 1000]),
            "medium": len(keywords_df[(keywords_df['Avg. monthly searches'] >= 100) & (keywords_df['Avg. monthly searches'] < 1000)]),
            "low": len(keywords_df[keywords_df['Avg. monthly searches'] < 100])
        }
        stats["volume_distribution"] = volume_counts
    
    return stats


@st.cache_data
def analyze_result_patterns(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze patterns in SERP results
    
    Args:
        results_df: DataFrame with SERP results
    
    Returns:
        Dictionary with pattern analysis
    """
    patterns = {}
    
    # Check if DataFrame has required columns
    required_columns = ['Keyword', 'Domain', 'Rank']
    if not all(col in results_df.columns for col in required_columns):
        return patterns
    
    # Domain performance by position
    domain_positions = {}
    for domain in results_df['Domain'].unique():
        domain_data = results_df[results_df['Domain'] == domain]
        positions = {
            "top3": len(domain_data[domain_data['Rank'] <= 3]),
            "top10": len(domain_data[domain_data['Rank'] <= 10]),
            "avg_position": float(domain_data['Rank'].mean())
        }
        domain_positions[domain] = positions
    
    patterns["domain_positions"] = domain_positions
    
    # Cluster performance if available
    if 'Cluster' in results_df.columns:
        cluster_performance = {}
        for cluster in results_df['Cluster'].unique():
            cluster_data = results_df[results_df['Cluster'] == cluster]
            performance = {
                "keywords": len(cluster_data['Keyword'].unique()),
                "avg_position": float(cluster_data['Rank'].mean()),
                "visibility": int(cluster_data['Visibility Score'].sum()) if 'Visibility Score' in cluster_data.columns else 0
            }
            cluster_performance[cluster] = performance
        
        patterns["cluster_performance"] = cluster_performance
    
    return patterns
