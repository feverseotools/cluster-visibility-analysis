"""
Utilidades para la aplicación SEO Visibility Estimator Pro.

Este paquete contiene módulos para optimización de costes, procesamiento de datos,
gestión de APIs y análisis SEO.
"""

# Versión del paquete
__version__ = '1.0.0'

# Importar funciones principales para facilitar su uso
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

# Definir lo que se importa con "from utils import *"
__all__ = [
    # De seo_calculator
    'detailed_cost_calculator',
    'display_cost_breakdown',
    
    # De optimization
    'cluster_representative_keywords',
    'group_similar_keywords',
    'calculate_api_cost',
    
    # De api_manager
    'APIKeyRotator',
    'fetch_serp_results_optimized',
    
    # De data_processing
    'process_large_dataset',
    'save_results_to_cache',
    'load_results_from_cache'
]
