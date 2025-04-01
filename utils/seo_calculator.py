"""
Cost calculators for SEO analysis.

This module provides functions to calculate and visualize API costs
for SEO analysis, with different optimization scenarios.
"""

import streamlit as st
import pandas as pd
import math
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional, Tuple

# Try to import global configuration
try:
    from config_manager import get_config
    config = get_config()
except ImportError:
    config = {}

# Import necessary functions from other modules
# Use try/except blocks to handle circular imports or unavailable modules
try:
    from .optimization import cluster_representative_keywords, calculate_api_cost
except (ImportError, ModuleNotFoundError):
    # Fallback definitions in case modules are not available
    def cluster_representative_keywords(keywords_df, max_representatives=100, advanced_sampling=True):
        """Fallback version of cluster_representative_keywords"""
        return keywords_df.head(max_representatives)
    
    def calculate_api_cost(num_keywords, api_plan="basic", batch_optimization=True, sampling_rate=1.0):
        """Fallback version of calculate_api_cost"""
        # Get pricing from configuration or use default values
        pricing = config.get('api', {}).get('serpapi', {}).get('plans', {}).get(api_plan.lower(), {})
        if not pricing:
            pricing = {
                "basic": {"monthly_cost": 50, "searches": 5000},
                "business": {"monthly_cost": 250, "searches": 30000},
                "enterprise": {"monthly_cost": 500, "searches": 70000}
            }.get(api_plan.lower(), {"monthly_cost": 50, "searches": 5000})
        
        # Calculate cost per search
        monthly_cost = pricing.get("monthly_cost", 50)
        searches = pricing.get("searches", 5000)
        cost_per_search = monthly_cost / searches
        
        # Apply optimizations
        effective_queries = num_keywords
        effective_queries = math.ceil(effective_queries * sampling_rate)
        if batch_optimization and effective_queries > 10:
            effective_queries = math.ceil(effective_queries * 0.95)
        
        # Calculate estimated cost
        estimated_cost = effective_queries * cost_per_search
        
        # Calculate percentage of monthly quota
        quota_percentage = (effective_queries / searches) * 100
        
        return {
            "num_queries": effective_queries,
            "original_queries": num_keywords,
            "reduction_percentage": round((1 - (effective_queries / num_keywords)) * 100, 2) if num_keywords > 0 else 0,
            "estimated_cost": round(estimated_cost, 2),
            "quota_percentage": round(quota_percentage, 2),
            "plan_details": f"{monthly_cost}$ for {searches} searches"
        }


def detailed_cost_calculator(keywords_df, optimization_settings=None):
    """
    Detailed calculator that shows costs for different optimization scenarios
    
    Args:
        keywords_df (DataFrame): DataFrame with keywords
        optimization_settings (dict, optional): Custom optimization settings
    
    Returns:
        dict: Detailed cost information for different scenarios
    """
    if optimization_settings is None:
        optimization_settings = {
            "use_representatives": True,
            "advanced_sampling": True,
            "batch_optimization": True,
            "max_keywords": min(100, len(keywords_df) // 2)
        }
    
    original_count = len(keywords_df)
    results = {}
    
    # Calculate different scenarios
    
    # Scenario 1: No optimization
    basic_cost = calculate_api_cost(original_count, "basic", False, 1.0)
    results["no_optimization"] = {
        "queries": original_count,
        "cost": basic_cost["estimated_cost"],
        "savings_pct": 0,
        "quota_pct": basic_cost["quota_percentage"]
    }
    
    # Scenario 2: Batch optimization only
    batch_cost = calculate_api_cost(original_count, "basic", True, 1.0)
    results["batch_only"] = {
        "queries": original_count,
        "cost": batch_cost["estimated_cost"],
        "savings_pct": ((basic_cost["estimated_cost"] - batch_cost["estimated_cost"]) / basic_cost["estimated_cost"] * 100) if basic_cost["estimated_cost"] > 0 else 0,
        "quota_pct": batch_cost["quota_percentage"]
    }
    
    # Scenario 3: Simple sampling only
    if optimization_settings["use_representatives"]:
        sampling_keywords = cluster_representative_keywords(
            keywords_df, 
            optimization_settings["max_keywords"],
            advanced_sampling=False
        )
        sampling_count = len(sampling_keywords)
        sampling_cost = calculate_api_cost(sampling_count, "basic", False, sampling_count/original_count)
        results["sampling_only"] = {
            "queries": sampling_count,
            "cost": sampling_cost["estimated_cost"],
            "savings_pct": ((basic_cost["estimated_cost"] - sampling_cost["estimated_cost"]) / basic_cost["estimated_cost"] * 100) if basic_cost["estimated_cost"] > 0 else 0,
            "quota_pct": sampling_cost["quota_percentage"]
        }
    
    # Scenario 4: Full optimization
    if optimization_settings["use_representatives"]:
        full_opt_keywords = cluster_representative_keywords(
            keywords_df, 
            optimization_settings["max_keywords"],
            advanced_sampling=optimization_settings["advanced_sampling"]
        )
        full_opt_count = len(full_opt_keywords)
        full_opt_cost = calculate_api_cost(
            full_opt_count, 
            "basic", 
            optimization_settings["batch_optimization"], 
            full_opt_count/original_count
        )
        results["full_optimization"] = {
            "queries": full_opt_count,
            "cost": full_opt_cost["estimated_cost"],
            "savings_pct": ((basic_cost["estimated_cost"] - full_opt_cost["estimated_cost"]) / basic_cost["estimated_cost"] * 100) if basic_cost["estimated_cost"] > 0 else 0,
            "quota_pct": full_opt_cost["quota_percentage"]
        }
    
    # Calculate total savings
    if "full_optimization" in results:
        total_savings = basic_cost["estimated_cost"] - results["full_optimization"]["cost"]
        total_savings_pct = results["full_optimization"]["savings_pct"]
    else:
        total_savings = 0
        total_savings_pct = 0
    
    results["total_savings"] = {
        "amount": total_savings,
        "percentage": total_savings_pct
    }
    
    return results


def display_cost_breakdown(cost_results):
    """
    Display a detailed cost breakdown in Streamlit
    
    Args:
        cost_results (dict): Results from the detailed calculator
    """
    st.subheader("Cost Breakdown by Strategy")
    
    # Create columns for scenarios
    cols = st.columns(4)
    
    # Scenario 1: No optimization
    with cols[0]:
        st.metric(
            "No optimization",
            f"${cost_results['no_optimization']['cost']:.2f}",
            f"{cost_results['no_optimization']['queries']} queries"
        )
        st.progress(min(1.0, cost_results['no_optimization']['quota_pct'] / 100))
        st.caption(f"Quota: {cost_results['no_optimization']['quota_pct']:.1f}%")
    
    # Scenario 2: Batch optimization only
    with cols[1]:
        st.metric(
            "Batching only",
            f"${cost_results['batch_only']['cost']:.2f}",
            f"-{cost_results['batch_only']['savings_pct']:.1f}%"
        )
        st.progress(min(1.0, cost_results['batch_only']['quota_pct'] / 100))
        st.caption(f"Quota: {cost_results['batch_only']['quota_pct']:.1f}%")
    
    # Scenario 3: Simple sampling only
    if "sampling_only" in cost_results:
        with cols[2]:
            st.metric(
                "Sampling only",
                f"${cost_results['sampling_only']['cost']:.2f}",
                f"-{cost_results['sampling_only']['savings_pct']:.1f}%"
            )
            st.progress(min(1.0, cost_results['sampling_only']['quota_pct'] / 100))
            st.caption(f"Quota: {cost_results['sampling_only']['quota_pct']:.1f}%")
    
    # Scenario 4: Full optimization
    if "full_optimization" in cost_results:
        with cols[3]:
            st.metric(
                "Full optimization",
                f"${cost_results['full_optimization']['cost']:.2f}",
                f"-{cost_results['full_optimization']['savings_pct']:.1f}%"
            )
            st.progress(min(1.0, cost_results['full_optimization']['quota_pct'] / 100))
            st.caption(f"Quota: {cost_results['full_optimization']['quota_pct']:.1f}%")
    
    # Savings summary
    st.subheader("Savings Summary")
    savings_col1, savings_col2 = st.columns(2)
    
    with savings_col1:
        st.metric(
            "Estimated total savings",
            f"${cost_results['total_savings']['amount']:.2f}",
            f"{cost_results['total_savings']['percentage']:.1f}% less"
        )
    
    with savings_col2:
        # Estimate time saved
        time_per_query = 1.5  # seconds
        if "full_optimization" in cost_results:
            time_saved = ((cost_results['no_optimization']['queries'] - 
                        cost_results['full_optimization']['queries']) * time_per_query) / 60
            queries_text = f"{cost_results['full_optimization']['queries']} queries vs {cost_results['no_optimization']['queries']}"
        else:
            time_saved = 0
            queries_text = "No optimization applied"
        
        st.metric(
            "Estimated time saved",
            f"{time_saved:.1f} minutes",
            queries_text
        )
    
    # Recommendations
    st.subheader("Recommendations")
    
    if cost_results['no_optimization']['quota_pct'] > 80:
        st.warning("⚠️ Analysis without optimization would consume most of your monthly quota. Optimizations are recommended.")
    
    if cost_results['total_savings']['percentage'] > 50:
        st.success("✅ Optimizations can significantly reduce cost (>50%). We recommend enabling them.")
    elif cost_results['total_savings']['percentage'] > 25:
        st.info("ℹ️ Optimizations offer moderate savings (>25%). Consider enabling them.")
    else:
        st.info("ℹ️ The keyword set is small, optimizations offer limited savings.")
    
    # Comparison chart
    import plotly.graph_objects as go
    
    strategies = []
    costs = []
    
    strategies.append("No optimization")
    costs.append(cost_results['no_optimization']['cost'])
    
    strategies.append("Batching only")
    costs.append(cost_results['batch_only']['cost'])
    
    if "sampling_only" in cost_results:
        strategies.append("Sampling only")
        costs.append(cost_results['sampling_only']['cost'])
    
    if "full_optimization" in cost_results:
        strategies.append("Full optimization")
        costs.append(cost_results['full_optimization']['cost'])
    
    fig = go.Figure(data=[
        go.Bar(
            x=strategies,
            y=costs,
            marker_color=['#ff9999', '#ffd699', '#99ff99', '#66b3ff'],
            text=[f"${cost:.2f}" for cost in costs],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Cost Comparison by Strategy',
        xaxis_title='Strategy',
        yaxis_title='Cost ($)',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def preliminary_cost_calculator(keywords_df):
    """
    Basic preliminary cost calculator for quick estimates
    
    Args:
        keywords_df (DataFrame): DataFrame with keywords
    
    Returns:
        tuple: (cost_data, recommended_settings)
    """
    # Get default settings from config
    settings = config.get('optimization', {})
    sampling_config = settings.get('sampling', {})
    
    # Default settings if not in config
    default_max_representatives = sampling_config.get('default_max_representatives', 100)
    use_representatives = sampling_config.get('enabled', True)
    advanced_sampling = sampling_config.get('advanced', True)
    
    if use_representatives:
        sample_df = cluster_representative_keywords(
            keywords_df, 
            min(default_max_representatives, len(keywords_df) // 2),
            advanced_sampling=advanced_sampling
        )
        num_queries = len(sample_df)
        sampling_ratio = num_queries / len(keywords_df)
    else:
        num_queries = len(keywords_df)
        sampling_ratio = 1.0
    
    # Calculate cost with batch optimization
    cost_data = calculate_api_cost(
        num_queries, 
        "basic", 
        batch_optimization=True, 
        sampling_rate=sampling_ratio
    )
    
    # Recommended settings
    recommended_settings = {
        "use_representatives": use_representatives,
        "advanced_sampling": advanced_sampling,
        "batch_optimization": True,
        "max_keywords": default_max_representatives
    }
    
    return cost_data, recommended_settings


def display_preliminary_calculator(keywords_df):
    """
    Display preliminary cost calculator in Streamlit
    
    Args:
        keywords_df (DataFrame): DataFrame with keywords
    
    Returns:
        tuple: (optimized_df, cost_data, optimization_settings)
    """
    st.subheader("Preliminary Cost Calculator")
    st.write("Estimate cost and optimize query before running full analysis")
    
    col1, col2 = st.columns(2)
    
    # Get API plan options from config
    api_plans = list(config.get('api', {}).get('serpapi', {}).get('plans', {}).keys())
    if not api_plans:
        api_plans = ["Basic", "Business", "Enterprise"]
    else:
        # Capitalize first letter
        api_plans = [plan.capitalize() for plan in api_plans]
    
    with col1:
        # API plan selection
        api_plan = st.selectbox(
            "SerpAPI Plan",
            options=api_plans,
            index=0
        )
        
        # Analysis frequency
        analysis_frequency = st.selectbox(
            "Analysis Frequency",
            options=["One-time", "Weekly", "Monthly", "Quarterly"],
            index=2
        )
        
        # Number of months
        months = st.slider("Analysis Period (months)", 1, 12, 3)
    
    with col2:
        # Keyword sampling strategy
        use_representatives = st.checkbox("Use representative keywords only", value=True)
        advanced_sampling = st.checkbox("Use advanced sampling (more accurate)", value=True)
        
        if use_representatives:
            max_keywords = st.slider(
                "Maximum number of representative keywords", 
                min_value=10, 
                max_value=min(500, len(keywords_df)), 
                value=min(100, len(keywords_df) // 2)
            )
        else:
            max_keywords = len(keywords_df)
        
        # Batch optimization
        batch_optimization = st.checkbox("Apply batch optimization", value=True,
                                       help="Reduces the number of API calls by grouping queries")
    
    # Calculate frequency multiplier
    frequency_multipliers = {
        "One-time": 1,
        "Weekly": 4 * months,  # ~4 weeks per month
        "Monthly": months,
        "Quarterly": math.ceil(months / 3)
    }
    
    multiplier = frequency_multipliers[analysis_frequency]
    
    # Calculate number of API calls
    if use_representatives:
        sample_df = cluster_representative_keywords(
            keywords_df, 
            max_keywords,
            advanced_sampling=advanced_sampling
        )
        num_queries = len(sample_df)
        sampling_ratio = num_queries / len(keywords_df)
    else:
        sample_df = keywords_df
        num_queries = len(keywords_df)
        sampling_ratio = 1.0
    
    # Calculate total cost
    total_queries = num_queries * multiplier
    cost_data = calculate_api_cost(
        total_queries, 
        api_plan.lower(), 
        batch_optimization=batch_optimization, 
        sampling_rate=sampling_ratio
    )
    
    # Display results
    st.subheader("Cost Estimate")
    
    cost_col1, cost_col2, cost_col3 = st.columns(3)
    
    with cost_col1:
        st.metric("Total Keywords", f"{len(keywords_df):,}")
        st.metric("Keywords Analyzed", f"{num_queries:,}")
        st.metric("Sampling Rate", f"{sampling_ratio:.1%}")
    
    with cost_col2:
        st.metric("Total API Queries", f"{total_queries:,}")
        st.metric("Estimated Cost", f"${cost_data['estimated_cost']:.2f}")
        st.metric("Monthly Quota Used", f"{cost_data['quota_percentage']:.1f}%")
    
    with cost_col3:
        st.metric("Savings with Optimizations", f"{cost_data['reduction_percentage']}%")
        st.metric("Frequency", f"{analysis_frequency}")
        st.metric("SerpAPI Plan", f"{api_plan}")
    
    # Display optimization metrics
    st.subheader("Applied Saving Strategies")
    
    saving_col1, saving_col2 = st.columns(2)
    
    with saving_col1:
        st.write("**Keyword sampling:**")
        if use_representatives:
            st.success(f"✅ Enabled - Reduces required queries by {(1 - sampling_ratio) * 100:.1f}%")
            if advanced_sampling:
                st.success("✅ Advanced sampling - Improves representativeness by cluster")
        else:
            st.error("❌ Disabled - All keywords will be analyzed")
    
    with saving_col2:
        st.write("**Batch optimization:**")
        if batch_optimization:
            st.success("✅ Enabled - Groups queries to reduce API calls")
        else:
            st.error("❌ Disabled - Batches will not be optimized")
    
    # Calculate estimated time
    avg_time_per_query = 1.5  # seconds per query with batching
    estimated_time = (total_queries / 5) * avg_time_per_query  # Assuming batch_size = 5
    
    # Convert to minutes
    estimated_minutes = estimated_time / 60
    
    st.info(f"⏱️ Estimated time for complete analysis: approximately {estimated_minutes:.1f} minutes")
    
    st.markdown("---")
    
    # Return optimization settings
    optimization_settings = {
        "use_representatives": use_representatives,
        "advanced_sampling": advanced_sampling,
        "batch_optimization": batch_optimization,
        "max_keywords": max_keywords
    }
    
    return sample_df, cost_data, optimization_settings
