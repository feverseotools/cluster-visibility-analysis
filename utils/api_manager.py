"""
API management utilities for SEO analysis.

This module provides functions and classes for efficiently managing API calls,
implementing rate limiting, caching, and error handling.
"""

import time
import random
import logging
from typing import Dict, List, Any, Optional, Union, Callable
import streamlit as st
from serpapi import GoogleSearch
import pandas as pd
import json
import os
from datetime import datetime, timedelta

# Try to import global configuration
try:
    from config_manager import get_config
    config = get_config()
except ImportError:
    config = {}


class APIKeyRotator:
    """
    Manages multiple API keys to distribute queries and stay within rate limits
    """
    
    def __init__(self, api_keys: List[str], rate_limits: Optional[Dict[str, float]] = None):
        """
        Initialize with a list of API keys
        
        Args:
            api_keys: List of API keys to rotate through
            rate_limits: Optional dictionary mapping keys to their rate limits
        """
        self.api_keys = api_keys
        self.current_index = 0
        self.usage_count = {key: 0 for key in api_keys}
        self.last_used = {key: datetime.min for key in api_keys}
        
        # Rate limits (queries per second)
        if rate_limits:
            self.rate_limits = rate_limits
        else:
            # Default rate limit from config or fallback to 1.0
            default_rate = config.get('api', {}).get('serpapi', {}).get('rate_limit', 1.0)
            self.rate_limits = {key: default_rate for key in api_keys}
    
    def get_next_key(self) -> str:
        """
        Returns the next available API key, respecting rate limits
        
        Returns:
            API key string
        """
        if not self.api_keys:
            raise ValueError("No API keys available")
        
        # Try each key in sequence
        for _ in range(len(self.api_keys)):
            key = self.api_keys[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.api_keys)
            
            # Check if this key is within rate limits
            time_since_last_use = (datetime.now() - self.last_used[key]).total_seconds()
            required_interval = 1.0 / self.rate_limits[key]
            
            if time_since_last_use >= required_interval:
                # Key is available
                self.usage_count[key] += 1
                self.last_used[key] = datetime.now()
                return key
        
        # If we get here, all keys are at rate limit, so wait and use the least-used key
        least_used_key = min(self.usage_count, key=self.usage_count.get)
        required_interval = 1.0 / self.rate_limits[least_used_key]
        time.sleep(required_interval)
        
        self.usage_count[least_used_key] += 1
        self.last_used[least_used_key] = datetime.now()
        return least_used_key
    
    def get_usage_stats(self) -> Dict[str, int]:
        """
        Returns usage statistics for all keys
        
        Returns:
            Dictionary mapping keys to usage counts
        """
        return self.usage_count


# Cache for API results to prevent duplicate calls
api_cache = {}


def fetch_serp_results_optimized(keyword: str, params: Dict[str, Any], 
                               use_cache: bool = True, cache_ttl: int = 86400,
                               api_key_rotator: Optional[APIKeyRotator] = None,
                               max_retries: int = 3) -> Dict[str, Any]:
    """
    Fetch SERP results with optimizations for caching, rate limiting, and error handling
    
    Args:
        keyword: Search keyword
        params: API parameters dictionary
        use_cache: Whether to use caching
        cache_ttl: Cache time-to-live in seconds (default: 24 hours)
        api_key_rotator: Optional APIKeyRotator instance for key rotation
        max_retries: Maximum number of retries on failure
    
    Returns:
        Dictionary with SERP results
    """
    # Standardize the keyword for cache lookup
    cache_key = f"{keyword}_{json.dumps(params, sort_keys=True)}"
    
    # Check cache first if enabled
    if use_cache and cache_key in api_cache:
        cache_entry = api_cache[cache_key]
        cache_time = cache_entry.get('timestamp', 0)
        if time.time() - cache_time < cache_ttl:
            return cache_entry.get('data', {})
    
    # If using key rotation, get a key from the rotator
    if api_key_rotator:
        params['api_key'] = api_key_rotator.get_next_key()
    
    # Get retry delay from config or use default
    retry_delay = config.get('api', {}).get('serpapi', {}).get('retry_delay', 2.0)
    
    # Attempt the API call with retries
    for attempt in range(max_retries):
        try:
            # Add jitter to avoid rate limit issues if multiple requests start at once
            jitter = random.uniform(0, 0.5)
            if attempt > 0:
                time.sleep(retry_delay * attempt + jitter)
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            # Cache the results if caching is enabled
            if use_cache:
                api_cache[cache_key] = {
                    'data': results,
                    'timestamp': time.time()
                }
            
            return results
        except Exception as e:
            logging.warning(f"API call failed (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                # Last attempt failed, raise the exception
                raise
    
    # This should not be reached due to the raise in the except block
    return {}


def clear_api_cache() -> None:
    """
    Clear the in-memory API cache
    """
    global api_cache
    api_cache = {}


def save_api_cache(cache_file: str = "cache/api_cache.json") -> None:
    """
    Save the current API cache to a file
    
    Args:
        cache_file: Path to save the cache
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    
    # Convert cache to serializable format
    serializable_cache = {}
    for key, value in api_cache.items():
        serializable_cache[key] = {
            'data': value.get('data', {}),
            'timestamp': value.get('timestamp', 0)
        }
    
    # Save to file
    with open(cache_file, 'w') as f:
        json.dump(serializable_cache, f)


def load_api_cache(cache_file: str = "cache/api_cache.json", max_age: int = 86400 * 7) -> None:
    """
    Load API cache from a file
    
    Args:
        cache_file: Path to the cache file
        max_age: Maximum age of cache entries in seconds (default: 7 days)
    """
    global api_cache
    
    if not os.path.exists(cache_file):
        return
    
    try:
        with open(cache_file, 'r') as f:
            loaded_cache = json.load(f)
        
        # Filter out expired entries
        current_time = time.time()
        api_cache = {
            key: value for key, value in loaded_cache.items()
            if current_time - value.get('timestamp', 0) < max_age
        }
    except Exception as e:
        logging.error(f"Error loading API cache: {str(e)}")


class APIQuotaManager:
    """
    Manages API quota usage to prevent exceeding limits
    """
    
    def __init__(self, monthly_quota: int, start_date: Optional[datetime] = None,
               warning_threshold: float = 0.8):
        """
        Initialize with quota information
        
        Args:
            monthly_quota: Maximum monthly API calls
            start_date: Start date of the billing cycle (defaults to 1st of current month)
            warning_threshold: Threshold for warnings (0.0-1.0)
        """
        self.monthly_quota = monthly_quota
        
        if start_date is None:
            # Default to 1st of current month
            today = datetime.now()
            self.start_date = datetime(today.year, today.month, 1)
        else:
            self.start_date = start_date
            
        self.warning_threshold = warning_threshold
        self.usage_count = 0
        self.last_reset = self.start_date
    
    def check_quota(self) -> bool:
        """
        Check if there is remaining quota
        
        Returns:
            True if quota available, False if exceeded
        """
        self._check_reset()
        return self.usage_count < self.monthly_quota
    
    def increment_usage(self, count: int = 1) -> bool:
        """
        Increment usage count and check quota
        
        Args:
            count: Number of API calls to add to usage
        
        Returns:
            True if still within quota, False if exceeded
        """
        self._check_reset()
        self.usage_count += count
        
        # Check if we've crossed the warning threshold
        if self.usage_count > self.monthly_quota * self.warning_threshold:
            warning_pct = (self.usage_count / self.monthly_quota) * 100
            logging.warning(f"API quota usage: {warning_pct:.1f}% ({self.usage_count}/{self.monthly_quota})")
        
        return self.usage_count <= self.monthly_quota
    
    def _check_reset(self) -> None:
        """
        Check if quota should be reset based on billing cycle
        """
        now = datetime.now()
        
        # Calculate next reset date
        if self.start_date.day == 1:
            # Monthly on the 1st
            next_reset = datetime(now.year, now.month, 1)
            if next_reset < self.start_date:
                # Ensure we're not before the initial start date
                next_reset = self.start_date
            
            # If we're in a new month from the last reset
            if now.year > self.last_reset.year or now.month > self.last_reset.month:
                self.usage_count = 0
                self.last_reset = next_reset
        else:
            # Monthly on a specific day
            # Find the next occurrence of that day
            next_reset = datetime(now.year, now.month, self.start_date.day)
            if next_reset < now:
                # Move to next month
                if now.month == 12:
                    next_reset = datetime(now.year + 1, 1, self.start_date.day)
                else:
                    next_reset = datetime(now.year, now.month + 1, self.start_date.day)
            
            # If we've passed a reset date
            if now - self.last_reset >= timedelta(days=28):  # Simplistic approach
                self.usage_count = 0
                self.last_reset = next_reset
    
    def get_remaining_quota(self) -> int:
        """
        Get number of API calls remaining in the quota
        
        Returns:
            Number of remaining API calls
        """
        self._check_reset()
        return max(0, self.monthly_quota - self.usage_count)
    
    def get_usage_percentage(self) -> float:
        """
        Get percentage of quota used
        
        Returns:
            Percentage of quota used (0.0-100.0)
        """
        self._check_reset()
        return (self.usage_count / self.monthly_quota) * 100 if self.monthly_quota > 0 else 100.0


def batch_process_keywords(keywords: List[str], process_func: Callable[[str], Any], 
                         batch_size: int = 5, progress_bar=None) -> List[Any]:
    """
    Process a list of keywords in batches with a progress bar
    
    Args:
        keywords: List of keywords to process
        process_func: Function to call for each keyword
        batch_size: Size of batches to process
        progress_bar: Optional Streamlit progress bar
    
    Returns:
        List of results from processing
    """
    results = []
    
    for i in range(0, len(keywords), batch_size):
        batch = keywords[i:i+batch_size]
        
        # Process the batch
        batch_results = [process_func(keyword) for keyword in batch]
        results.extend(batch_results)
        
        # Update progress
        if progress_bar:
            progress_bar.progress(min(1.0, (i + batch_size) / len(keywords)))
        
        # Pause between batches to respect API limits if not the last batch
        if i + batch_size < len(keywords):
            # Dynamic pause based on batch size and config
            base_pause = config.get('api', {}).get('serpapi', {}).get('batch_pause', 0.2)
            pause_time = min(2.0, base_pause * batch_size)
            time.sleep(pause_time)
    
    return results
