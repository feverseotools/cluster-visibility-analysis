"""
Utilities for the SEO Visibility Estimator Pro application.

This package contains modules for cost optimization, data processing,
API management, and SEO analysis.
"""

# Package version
__version__ = '1.0.0'

# Import main functions to make them easier to use
from .seo_calculator import (
    detailed_cost_calculator,
    display_cost_breakdown
)

from .optimization import (
    cluster_representative_keywords,
    group_similar_keywords,
    calculate_api_cost
)

from .api_manager import (
    APIKeyRotator,
    fetch_serp_results_optimized
)

from .data_processing import (
    process_large_dataset,
    save_results_to_cache,
    load_results_from_cache
)

# Define what gets imported with "from utils import *"
__all__ = [
    # From seo_calculator
    'detailed_cost_calculator',
    'display_cost_breakdown',
    
    # From optimization
    'cluster_representative_keywords',
    'group_similar_keywords',
    'calculate_api_cost',
    
    # From api_manager
    'APIKeyRotator',
    'fetch_serp_results_optimized',
    
    # From data_processing
    'process_large_dataset',
    'save_results_to_cache',
    'load_results_from_cache'
]
