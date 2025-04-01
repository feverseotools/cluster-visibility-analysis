import streamlit as st
import pandas as pd
import math
import os
import json
import time
from datetime import datetime

# ============================
# Basic Utility Functions
# ============================

def extract_domain(url):
    """Extract main domain from a URL"""
    import re
    pattern = r'(?:https?:\/\/)?(?:www\.)?([^\/\n]+)'
    match = re.search(pattern, url)
    return match.group(1) if match else url

def get_ctr_by_position(position):
    """Return estimated CTR by search result position"""
    ctr_model = {
        1: 0.316,  # 31.6% for position 1
        2: 0.158,
        3: 0.096,
        4: 0.072,
        5: 0.0596,
        6: 0.0454,
        7: 0.0379,
        8: 0.0312,
        9: 0.0278,
        10: 0.0236
    }
    return ctr_model.get(position, 0.01)  # Default 1% for positions beyond 10

# ============================
# API Cost Calculation
# ============================

def calculate_api_cost(num_keywords, api_plan="basic", batch_optimization=True, sampling_rate=1.0):
    """
    Calculate the estimated SerpAPI cost based on the number of queries and optimization settings.
    
    Args:
        num_keywords (int): Total number of keyword queries.
        api_plan (str): API pricing tier ("basic", "business", "enterprise").
        batch_optimization (bool): Whether batch optimization is applied.
        sampling_rate (float): Ratio of queries after sampling.
    
    Returns:
        dict: Estimated cost, quota percentage, and related details.
    """
    pricing = {
        "basic": {"monthly_cost": 50, "searches": 5000},
        "business": {"monthly_cost": 250, "searches": 30000},
        "enterprise": {"monthly_cost": 500, "searches": 70000}
    }
    plan = pricing.get(api_plan.lower(), pricing["basic"])
    
    effective_queries = math.ceil(num_keywords * sampling_rate)
    if batch_optimization and effective_queries > 10:
        # Estimate a conservative 5% reduction due to batching
        effective_queries = math.ceil(effective_queries * 0.95)
    
    cost_per_search = plan["monthly_cost"] / plan["searches"]
    estimated_cost = effective_queries * cost_per_search
    quota_percentage = (effective_queries / plan["searches"]) * 100
    
    return {
        "num_queries": effective_queries,
        "original_queries": num_keywords,
        "reduction_percentage": round((1 - (effective_queries / num_keywords)) * 100, 2) if num_keywords > 0 else 0,
        "estimated_cost": round(estimated_cost, 2),
        "quota_percentage": round(quota_percentage, 2),
        "plan_details": f"${plan['monthly_cost']} for {plan['searches']} searches"
    }

# ============================
# Keyword Sampling & Grouping
# ============================

def cluster_representative_keywords(keywords_df, max_representatives=100, advanced_sampling=True):
    """
    Select representative keywords from each cluster to reduce API calls.
    
    Args:
        keywords_df (DataFrame): DataFrame containing keyword data with a 'cluster_name' column.
        max_representatives (int): Maximum number of representative keywords.
        advanced_sampling (bool): Whether to use advanced sampling for improved representativity.
    
    Returns:
        DataFrame: A subset of keywords representing each cluster.
    """
    if keywords_df.empty or 'cluster_name' not in keywords_df.columns:
        return keywords_df
    
    grouped = keywords_df.groupby('cluster_name')
    
    # Advanced sampling: calculate diversity per cluster based on search volume
    if advanced_sampling:
        cluster_diversities = {}
        for cluster_name, group in grouped:
            vol_range = group['Avg. monthly searches'].max() - group['Avg. monthly searches'].min()
            vol_std = group['Avg. monthly searches'].std()
            cluster_diversities[cluster_name] = (vol_range + vol_std) / 2
        total_diversity = sum(cluster_diversities.values())
        if total_diversity > 0:
            for cluster in cluster_diversities:
                cluster_diversities[cluster] /= total_diversity
    
    # Compute importance based on cluster size and total search volume
    cluster_sizes = grouped.size()
    cluster_volumes = grouped['Avg. monthly searches'].sum()
    if advanced_sampling:
        cluster_importance = (
            0.3 * (cluster_sizes / cluster_sizes.sum()) +
            0.5 * (cluster_volumes / cluster_volumes.sum()) +
            0.2 * pd.Series(cluster_diversities)
        )
    else:
        cluster_importance = (
            0.4 * (cluster_sizes / cluster_sizes.sum()) +
            0.6 * (cluster_volumes / cluster_volumes.sum())
        )
    
    reps_per_cluster = (cluster_importance * max_representatives).apply(lambda x: max(1, round(x)))
    # Adjust if total representatives exceed max_representatives
    while reps_per_cluster.sum() > max_representatives:
        max_cluster = reps_per_cluster.idxmax()
        reps_per_cluster[max_cluster] -= 1
    
    selected_keywords = []
    for cluster, count in reps_per_cluster.items():
        cluster_data = keywords_df[keywords_df['cluster_name'] == cluster]
        if advanced_sampling and len(cluster_data) > count:
            sorted_data = cluster_data.sort_values('Avg. monthly searches')
            step = len(sorted_data) / count
            indices = [int(i * step) for i in range(count)]
            selected_keywords.append(sorted_data.iloc[indices])
        else:
            sorted_data = cluster_data.sort_values('Avg. monthly searches', ascending=False)
            selected_keywords.append(sorted_data.head(int(count)))
    
    representative_df = pd.concat(selected_keywords)
    return representative_df

def group_similar_keywords(keywords_df, similarity_threshold=0.8):
    """
    Group very similar keywords to reduce the number of API queries.
    
    Args:
        keywords_df (DataFrame): DataFrame with a 'keyword' column.
        similarity_threshold (float): Threshold for cosine similarity (0 to 1).
    
    Returns:
        DataFrame: A DataFrame containing only representative keywords.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if len(keywords_df) < 100:
        return keywords_df

    vectorizer = TfidfVectorizer()
    try:
        X = vectorizer.fit_transform(keywords_df['keyword'])
        similarity_matrix = cosine_similarity(X)
        groups = {}
        processed = set()
        for i in range(len(keywords_df)):
            if i in processed:
                continue
            similar_indices = [
                j for j in range(len(keywords_df))
                if similarity_matrix[i, j] > similarity_threshold and j not in processed
            ]
            if similar_indices:
                group_df = keywords_df.iloc[[i] + similar_indices]
                representative_idx = group_df['Avg. monthly searches'].idxmax()
                processed.update([i] + similar_indices)
                groups[representative_idx] = [i] + similar_indices
        representative_indices = list(groups.keys()) + [i for i in range(len(keywords_df)) if i not in processed]
        return keywords_df.iloc[representative_indices].copy()
    except Exception as e:
        print(f"Error grouping similar keywords: {str(e)}")
        return keywords_df

# ============================
# Detailed Cost Calculator & Breakdown
# ============================

def detailed_cost_calculator(keywords_df, optimization_settings=None):
    """
    Detailed cost calculator that displays API costs for different optimization scenarios.
    
    Args:
        keywords_df (DataFrame): DataFrame with keyword data.
        optimization_settings (dict, optional): Custom optimization settings.
        
    Returns:
        dict: Detailed cost information for different scenarios.
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
        "savings_pct": ((basic_cost["estimated_cost"] - batch_cost["estimated_cost"]) / basic_cost["estimated_cost"] * 100)
                      if basic_cost["estimated_cost"] > 0 else 0,
        "quota_pct": batch_cost["quota_percentage"]
    }
    
    # Scenario 3: Sampling only (without advanced sampling)
    if optimization_settings["use_representatives"]:
        sampling_keywords = cluster_representative_keywords(
            keywords_df,
            optimization_settings["max_keywords"],
            advanced_sampling=False
        )
        sampling_count = len(sampling_keywords)
        sampling_cost = calculate_api_cost(sampling_count, "basic", False, sampling_count / original_count)
        results["sampling_only"] = {
            "queries": sampling_count,
            "cost": sampling_cost["estimated_cost"],
            "savings_pct": ((basic_cost["estimated_cost"] - sampling_cost["estimated_cost"]) / basic_cost["estimated_cost"] * 100)
                          if basic_cost["estimated_cost"] > 0 else 0,
            "quota_pct": sampling_cost["quota_percentage"]
        }
    
    # Scenario 4: Full optimization (sampling with advanced options and batching)
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
            full_opt_count / original_count
        )
        results["full_optimization"] = {
            "queries": full_opt_count,
            "cost": full_opt_cost["estimated_cost"],
            "savings_pct": ((basic_cost["estimated_cost"] - full_opt_cost["estimated_cost"]) / basic_cost["estimated_cost"] * 100)
                          if basic_cost["estimated_cost"] > 0 else 0,
            "quota_pct": full_opt_cost["quota_percentage"]
        }
    
    # Total savings summary
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
    Display a detailed cost breakdown in Streamlit.
    
    Args:
        cost_results (dict): Results from the detailed cost calculator.
    """
    st.subheader("Cost Breakdown by Strategy")
    cols = st.columns(4)
    
    # Scenario 1: No Optimization
    with cols[0]:
        st.metric(
            "No Optimization",
            f"${cost_results['no_optimization']['cost']:.2f}",
            f"{cost_results['no_optimization']['queries']} queries"
        )
        st.progress(min(1.0, cost_results['no_optimization']['quota_pct'] / 100))
        st.caption(f"Quota: {cost_results['no_optimization']['quota_pct']:.1f}%")
    
    # Scenario 2: Batching Only
    with cols[1]:
        st.metric(
            "Batching Only",
            f"${cost_results['batch_only']['cost']:.2f}",
            f"-{cost_results['batch_only']['savings_pct']:.1f}%"
        )
        st.progress(min(1.0, cost_results['batch_only']['quota_pct'] / 100))
        st.caption(f"Quota: {cost_results['batch_only']['quota_pct']:.1f}%")
    
    # Scenario 3: Sampling Only
    if "sampling_only" in cost_results:
        with cols[2]:
            st.metric(
                "Sampling Only",
                f"${cost_results['sampling_only']['cost']:.2f}",
                f"-{cost_results['sampling_only']['savings_pct']:.1f}%"
            )
            st.progress(min(1.0, cost_results['sampling_only']['quota_pct'] / 100))
            st.caption(f"Quota: {cost_results['sampling_only']['quota_pct']:.1f}%")
    
    # Scenario 4: Full Optimization
    if "full_optimization" in cost_results:
        with cols[3]:
            st.metric(
                "Full Optimization",
                f"${cost_results['full_optimization']['cost']:.2f}",
                f"-{cost_results['full_optimization']['savings_pct']:.1f}%"
            )
            st.progress(min(1.0, cost_results['full_optimization']['quota_pct'] / 100))
            st.caption(f"Quota: {cost_results['full_optimization']['quota_pct']:.1f}%")
    
    st.subheader("Savings Summary")
    savings_col1, savings_col2 = st.columns(2)
    with savings_col1:
        st.metric(
            "Estimated Total Savings",
            f"${cost_results['total_savings']['amount']:.2f}",
            f"{cost_results['total_savings']['percentage']:.1f}% less"
        )
    with savings_col2:
        time_per_query = 1.5  # seconds per query
        time_saved = ((cost_results['no_optimization']['queries'] - cost_results['full_optimization']['queries']) * time_per_query) / 60
        st.metric(
            "Estimated Time Saved",
            f"{time_saved:.1f} minutes",
            f"{cost_results['full_optimization']['queries']} queries vs {cost_results['no_optimization']['queries']}"
        )
    
    st.subheader("Recommendations")
    if cost_results['no_optimization']['quota_pct'] > 80:
        st.warning("⚠️ Analysis without optimization would use a large portion of your monthly quota. Optimizations are recommended.")
    if cost_results['total_savings']['percentage'] > 50:
        st.success("✅ Optimizations can significantly reduce cost (>50%). We recommend enabling them.")
    elif cost_results['total_savings']['percentage'] > 25:
        st.info("ℹ️ Optimizations offer moderate savings (>25%). Consider enabling them.")
    else:
        st.info("ℹ️ The keyword set is small; optimizations offer limited savings.")
    
    # Comparison chart using Plotly
    import plotly.graph_objects as go
    strategies = []
    costs = []
    
    strategies.append("No Optimization")
    costs.append(cost_results['no_optimization']['cost'])
    strategies.append("Batching Only")
    costs.append(cost_results['batch_only']['cost'])
    if "sampling_only" in cost_results:
        strategies.append("Sampling Only")
        costs.append(cost_results['sampling_only']['cost'])
    if "full_optimization" in cost_results:
        strategies.append("Full Optimization")
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

# ============================
# Large Dataset Processing & Caching
# ============================

def process_large_dataset(keywords_df, max_per_session=5000):
    """
    Divide very large datasets into manageable chunks.
    
    Args:
        keywords_df (DataFrame): The complete keyword dataset.
        max_per_session (int): Maximum keywords to process in one session.
        
    Returns:
        list: A list of DataFrame fragments.
    """
    total_keywords = len(keywords_df)
    if total_keywords <= max_per_session:
        return [keywords_df]
    
    num_sessions = (total_keywords + max_per_session - 1) // max_per_session
    
    if 'cluster_name' in keywords_df.columns:
        clusters = keywords_df['cluster_name'].unique()
        clusters_per_session = (len(clusters) + num_sessions - 1) // num_sessions
        fragments = []
        for i in range(0, len(clusters), clusters_per_session):
            session_clusters = clusters[i:i+clusters_per_session]
            fragment = keywords_df[keywords_df['cluster_name'].isin(session_clusters)]
            fragments.append(fragment)
        return fragments
    else:
        return [keywords_df.iloc[i:i+max_per_session] for i in range(0, total_keywords, max_per_session)]

def save_results_to_cache(results_df, cache_id):
    """
    Save analysis results to persistent cache.
    
    Args:
        results_df (DataFrame): Results DataFrame to cache.
        cache_id (str): Unique identifier for the cache file.
    
    Returns:
        str: The path to the cache file.
    """
    cache_dir = "cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    cache_file = f"{cache_dir}/results_{cache_id}.pkl"
    results_df.to_pickle(cache_file)
    
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "num_keywords": len(results_df),
        "domains": results_df['Domain'].unique().tolist()
    }
    with open(f"{cache_dir}/metadata_{cache_id}.json", 'w') as f:
        json.dump(metadata, f)
    
    return cache_file

def load_results_from_cache(cache_id):
    """
    Load analysis results from persistent cache.
    
    Args:
        cache_id (str): Unique identifier for the cache file.
    
    Returns:
        tuple: (DataFrame, bool) where bool indicates whether the cache was found.
    """
    cache_file = f"cache/results_{cache_id}.pkl"
    if os.path.exists(cache_file):
        return pd.read_pickle(cache_file), True
    return pd.DataFrame(), False

# ============================
# API Key Rotation
# ============================

class APIKeyRotator:
    """Manages multiple API keys to distribute queries."""
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.current_index = 0
        self.usage_count = {key: 0 for key in api_keys}
    
    def get_next_key(self):
        """Return the next available API key."""
        key = self.api_keys[self.current_index]
        self.usage_count[key] += 1
        self.current_index = (self.current_index + 1) % len(self.api_keys)
        return key
    
    def get_usage_stats(self):
        """Return usage statistics for each API key."""
        return self.usage_count

# ============================
# Email Report Notification
# ============================

def send_email_report(email, report_data, api_usage):
    """
    Send an email report when a long analysis completes.
    
    Args:
        email (str): Recipient email address.
        report_data (dict): Data from the analysis report.
        api_usage (dict): API usage details.
    
    Returns:
        bool: True if the email was sent successfully, False otherwise.
    """
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    # Configure your email settings (customize as needed)
    sender_email = "youremail@example.com"
    receiver_email = email
    subject = "SEO Analysis Report Completed"
    
    # Create HTML content for the email
    html_content = f"""
    <h1>SEO Analysis Completed</h1>
    <p>The analysis has finished with the following results:</p>
    
    <h2>Summary</h2>
    <ul>
        <li>Keywords analyzed: {report_data['keywords_analyzed']}</li>
        <li>Domains analyzed: {', '.join(report_data['domains'])}</li>
        <li>Total visibility: {report_data['total_visibility']}</li>
    </ul>
    
    <h2>API Usage</h2>
    <ul>
        <li>Queries made: {api_usage['queries_made']}</li>
        <li>Estimated cost: ${api_usage['estimated_cost']}</li>
        <li>Savings: ${api_usage['savings']}</li>
    </ul>
    
    <p>Please log in to the application to view the full report.</p>
    """
    
    # Create the email message
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email
    part = MIMEText(html_content, 'html')
    msg.attach(part)
    
    # Send the email (adjust SMTP settings accordingly)
    try:
        with smtplib.SMTP('smtp.example.com', 587) as server:
            server.starttls()
            server.login(sender_email, "your_password")
            server.sendmail(sender_email, receiver_email, msg.as_string())
        return True
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False

# ============================
# Practical Implementation Notes:
# ============================
# - Replace the original cost calculator functions with these improved versions.
# - Use 'display_cost_breakdown' to show detailed cost metrics in your Streamlit app.
# - Update your keyword processing functions (e.g., process_keywords_in_batches, analyze_competitors)
#   to accept and use optimization parameters.
# - For very large datasets, use process_large_dataset to split the work.
# - Use save_results_to_cache and load_results_from_cache to persist results between sessions.
# - Optionally, use group_similar_keywords to further reduce API queries.
# - If you have multiple API keys, utilize APIKeyRotator.
# - For long-running analyses, call send_email_report to notify users via email.
