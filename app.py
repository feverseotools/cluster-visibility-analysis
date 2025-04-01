import streamlit as st
import pandas as pd
import math
import os
import json
import time
from datetime import datetime
import concurrent.futures
import re
import plotly.graph_objects as go

# -----------------------------
# Page Configuration and Caching
# -----------------------------
st.set_page_config(page_title='SEO Visibility Estimator Pro', layout='wide')

@st.cache_data(ttl=86400*7)  # Cache for 7 days
def fetch_serp_results(keyword, params):
    """Cache search results to avoid duplicate API calls using SerpAPI."""
    from serpapi import GoogleSearch
    search = GoogleSearch(params)
    return search.get_dict()

@st.cache_data(ttl=3600*24)  # Cache for 24 hours
def process_csv(uploaded_file):
    """Cache CSV processing."""
    return pd.read_csv(uploaded_file)

# -----------------------------
# Basic Utility Functions
# -----------------------------
def extract_domain(url):
    """Extract the main domain from a URL."""
    pattern = r'(?:https?:\/\/)?(?:www\.)?([^\/\n]+)'
    match = re.search(pattern, url)
    return match.group(1) if match else url

def get_ctr_by_position(position):
    """Return estimated CTR by search result position."""
    ctr_model = {
        1: 0.316,
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
    return ctr_model.get(position, 0.01)

# -----------------------------
# Keyword Cluster Representative Selection
# -----------------------------
def cluster_representative_keywords(keywords_df, max_keywords=100, advanced_sampling=True):
    """
    Select representative keywords from each cluster to reduce API calls while maintaining coverage.
    
    Parameters:
    keywords_df (DataFrame): DataFrame containing keywords data with 'cluster_name' and 'Avg. monthly searches'
    max_keywords (int): Maximum number of keywords to return
    advanced_sampling (bool): Whether to use advanced sampling techniques
    
    Returns:
    DataFrame: A subset of keywords representing each cluster
    """
    if keywords_df.empty:
        return keywords_df
    
    # If max_keywords is greater than or equal to the number of keywords, return all keywords
    if max_keywords <= 0 or max_keywords >= len(keywords_df):
        return keywords_df
    
    # Count clusters and determine allocation strategy
    clusters = keywords_df['cluster_name'].unique()
    num_clusters = len(clusters)
    
    # Base allocation: ensure at least one keyword per cluster
    keywords_per_cluster = max(1, max_keywords // num_clusters)
    
    # If advanced sampling is enabled, use a more sophisticated approach
    if advanced_sampling:
        # Calculate cluster sizes
        cluster_sizes = keywords_df.groupby('cluster_name').size()
        
        # Calculate volume importance for each cluster
        cluster_volumes = keywords_df.groupby('cluster_name')['Avg. monthly searches'].sum()
        total_volume = cluster_volumes.sum()
        volume_importance = cluster_volumes / total_volume if total_volume > 0 else 0
        
        # Calculate weighted importance (size + volume)
        size_importance = cluster_sizes / cluster_sizes.sum()
        weighted_importance = (size_importance + volume_importance) / 2
        
        # Allocate keywords based on weighted importance
        # Ensure each cluster gets at least one keyword
        allocated_keywords = {}
        remaining = max_keywords - num_clusters
        
        for cluster in clusters:
            allocated_keywords[cluster] = 1
        
        # Distribute remaining keywords proportionally
        if remaining > 0:
            for cluster in clusters:
                additional = int(remaining * weighted_importance[cluster])
                allocated_keywords[cluster] += additional
            
            # Allocate any remaining keywords to the most important clusters
            distributed = sum(allocated_keywords.values())
            still_remaining = max_keywords - distributed
            
            if still_remaining > 0:
                sorted_clusters = sorted(
                    weighted_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                for i in range(min(still_remaining, len(sorted_clusters))):
                    allocated_keywords[sorted_clusters[i][0]] += 1
    else:
        # Simple allocation: equal distribution
        allocated_keywords = {cluster: keywords_per_cluster for cluster in clusters}
    
    # Select representative keywords from each cluster
    result_df = pd.DataFrame()
    
    for cluster, num_keywords in allocated_keywords.items():
        cluster_keywords = keywords_df[keywords_df['cluster_name'] == cluster]
        
        # Sort by search volume and select top keywords
        top_keywords = cluster_keywords.sort_values('Avg. monthly searches', ascending=False).head(num_keywords)
        result_df = pd.concat([result_df, top_keywords])
    
    return result_df

# -----------------------------
# API Cost Calculation Functions
# -----------------------------
def calculate_api_cost(num_keywords, api_plan="basic", batch_optimization=True, sampling_rate=1.0):
    """
    Calculate estimated SerpAPI cost (in dollars) based on the number of queries and optimization settings.
    For this app, we use the number of queries as 'credits used'.
    """
    pricing = {
        "basic": {"monthly_cost": 50, "searches": 5000},
        "business": {"monthly_cost": 250, "searches": 30000},
        "enterprise": {"monthly_cost": 500, "searches": 70000}
    }
    plan = pricing.get(api_plan.lower(), pricing["basic"])
    effective_queries = math.ceil(num_keywords * sampling_rate)
    if batch_optimization and effective_queries > 10:
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

def calculate_openai_cost(num_analyses, model="gpt-3.5-turbo"):
    """
    Calculate an estimated OpenAI cost in euros.
    For example, assume each analysis costs €0.05 for GPT-3.5-turbo and €0.25 for GPT-4.
    """
    if model == "gpt-3.5-turbo":
        cost_per_analysis = 0.05
    else:
        cost_per_analysis = 0.25
    return round(num_analyses * cost_per_analysis, 2)

def detailed_cost_calculator(keywords_df, optimization_settings=None):
    """
    Detailed cost calculator that shows API costs for different optimization scenarios.
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
    # Scenario 2: Batching only
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
    # Scenario 4: Full optimization (advanced sampling with batching)
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

def display_cost_breakdown(cost_results, openai_cost, serpapi_credits_used):
    """
    Display a detailed cost breakdown in Streamlit.
    Shows SerpApi credit usage and OpenAI estimated cost in euros.
    """
    st.subheader("Cost Breakdown by Strategy (SerpApi Dollars)")
    cols = st.columns(4)
    with cols[0]:
        st.metric(
            "No Optimization",
            f"${cost_results['no_optimization']['cost']:.2f}",
            f"{cost_results['no_optimization']['queries']} queries"
        )
        st.progress(min(1.0, cost_results['no_optimization']['quota_pct'] / 100))
        st.caption(f"Quota: {cost_results['no_optimization']['quota_pct']:.1f}%")
    with cols[1]:
        st.metric(
            "Batching Only",
            f"${cost_results['batch_only']['cost']:.2f}",
            f"-{cost_results['batch_only']['savings_pct']:.1f}%"
        )
        st.progress(min(1.0, cost_results['batch_only']['quota_pct'] / 100))
        st.caption(f"Quota: {cost_results['batch_only']['quota_pct']:.1f}%")
    if "sampling_only" in cost_results:
        with cols[2]:
            st.metric(
                "Sampling Only",
                f"${cost_results['sampling_only']['cost']:.2f}",
                f"-{cost_results['sampling_only']['savings_pct']:.1f}%"
            )
            st.progress(min(1.0, cost_results['sampling_only']['quota_pct'] / 100))
            st.caption(f"Quota: {cost_results['sampling_only']['quota_pct']:.1f}%")
    if "full_optimization" in cost_results:
        with cols[3]:
            st.metric(
                "Full Optimization",
                f"${cost_results['full_optimization']['cost']:.2f}",
                f"-{cost_results['full_optimization']['savings_pct']:.1f}%"
            )
            st.progress(min(1.0, cost_results['full_optimization']['quota_pct'] / 100))
            st.caption(f"Quota: {cost_results['full_optimization']['quota_pct']:.1f}%")
    st.subheader("Savings Summary (SerpApi Dollars)")
    savings_col1, savings_col2 = st.columns(2)
    with savings_col1:
        st.metric(
            "Estimated Total Savings",
            f"${cost_results['total_savings']['amount']:.2f}",
            f"{cost_results['total_savings']['percentage']:.1f}% less"
        )
    with savings_col2:
        time_per_query = 1.5
        time_saved = ((cost_results['no_optimization']['queries'] - cost_results['full_optimization']['queries']) * time_per_query) / 60
        st.metric(
            "Estimated Time Saved",
            f"{time_saved:.1f} minutes",
            f"{cost_results['full_optimization']['queries']} vs {cost_results['no_optimization']['queries']} queries"
        )
    st.subheader("API Cost Summary")
    st.metric("SerpApi Credits Used", f"{serpapi_credits_used} credits")
    st.metric("OpenAI Estimated Cost (€)", f"€{openai_cost:.2f}")
    st.subheader("Recommendations")
    if cost_results['no_optimization']['quota_pct'] > 80:
        st.warning("⚠️ Analysis without optimization would use a large portion of your monthly quota. Optimizations are recommended.")
    if cost_results['total_savings']['percentage'] > 50:
        st.success("✅ Optimizations can significantly reduce cost (>50%). We recommend enabling them.")
    elif cost_results['total_savings']['percentage'] > 25:
        st.info("ℹ️ Optimizations offer moderate savings (>25%). Consider enabling them.")
    else:
        st.info("ℹ️ The keyword set is small; optimizations offer limited savings.")
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
        title='Cost Comparison by Strategy (SerpApi Dollars)',
        xaxis_title='Strategy',
        yaxis_title='Cost ($)',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Advanced Metrics Calculation
# -----------------------------
def calculate_advanced_metrics(results_df):
    """
    Calculate advanced visibility metrics, including Share of Voice (SOV) and Improvement Potential.
    """
    if results_df.empty:
        return pd.DataFrame()
    domain_metrics = results_df.groupby(['Domain', 'Cluster']).agg({
        'Keyword': 'count',
        'Search Volume': 'sum',
        'Visibility Score': 'sum',
        'Estimated_Traffic': 'sum',
        'Rank': ['mean', 'min', 'max']
    }).reset_index()
    domain_metrics.columns = ['_'.join(col).strip('_') for col in domain_metrics.columns.values]
    domain_metrics = domain_metrics.rename(columns={
        'Keyword_count': 'Keywords_Count',
        'Search Volume_sum': 'Total_Search_Volume',
        'Visibility Score_sum': 'Total_Visibility_Score',
        'Estimated_Traffic_sum': 'Total_Estimated_Traffic',
        'Rank_mean': 'Average_Position',
        'Rank_min': 'Best_Position',
        'Rank_max': 'Worst_Position'
    })
    total_visibility = results_df['Visibility Score'].sum()
    domain_metrics['SOV_Percentage'] = (domain_metrics['Total_Visibility_Score'] / total_visibility * 100).round(2)
    domain_metrics['Improvement_Potential'] = domain_metrics.apply(
        lambda x: (100 - (101 - x['Average_Position'])) * x['Total_Search_Volume'] if x['Average_Position'] > 3 else 0,
        axis=1
    )
    return domain_metrics

# -----------------------------
# Large Dataset Processing & Persistent Caching
# -----------------------------
def process_large_dataset(keywords_df, max_per_session=5000):
    """
    Divide very large datasets into manageable chunks.
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
    """
    cache_dir = "cache"
    cache_file = f"{cache_dir}/results_{cache_id}.pkl"
    if os.path.exists(cache_file):
        return pd.read_pickle(cache_file), True
    return pd.DataFrame(), False

# -----------------------------
# API Key Rotation
# -----------------------------
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

# -----------------------------
# Email Report Notification
# -----------------------------
def send_email_report(email, report_data, api_usage):
    """
    Send an email report when a long analysis completes.
    """
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    sender_email = "youremail@example.com"
    receiver_email = email
    subject = "SEO Analysis Report Completed"
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
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email
    part = MIMEText(html_content, 'html')
    msg.attach(part)
    try:
        with smtplib.SMTP('smtp.example.com', 587) as server:
            server.starttls()
            server.login(sender_email, "your_password")
            server.sendmail(sender_email, receiver_email, msg.as_string())
        return True
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False

# -----------------------------
# Optimized Keyword Processing Functions
# -----------------------------
def process_keywords_in_batches(keywords_df, domains, params_template, batch_size=5, optimize_batch=True, max_retries=2):
    """
    Process keywords in batches with improved efficiency and retry mechanism.
    Modified: if no SERP result matches the specified domains, include all organic results.
    """
    all_results = []
    progress_bar = st.progress(0)
    if optimize_batch and 'cluster_name' in keywords_df.columns:
        keywords_df = keywords_df.sort_values('cluster_name')
        cluster_sizes = keywords_df.groupby('cluster_name').size()
        avg_cluster_size = cluster_sizes.mean()
        if avg_cluster_size < 10:
            batch_size = min(10, max(5, int(50 / avg_cluster_size)))
    for i in range(0, len(keywords_df), batch_size):
        batch = keywords_df.iloc[i:i+batch_size]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for _, row in batch.iterrows():
                params = params_template.copy()
                params["q"] = row['keyword']
                future = executor.submit(fetch_serp_results, row['keyword'], params)
                futures.append((future, row))
            for future, row in futures:
                retry_count = 0
                success = False
                row_results = []
                while retry_count < max_retries and not success:
                    try:
                        results = future.result()
                        organic_results = results.get("organic_results", [])
                        # First, try to filter results matching the specified domains
                        for rank, result in enumerate(organic_results, 1):
                            if any(domain.lower() in result.get('link', '').lower() for domain in domains):
                                visibility_score = (101 - rank) * row['Avg. monthly searches']
                                ctr = get_ctr_by_position(rank)
                                est_traffic = ctr * row['Avg. monthly searches']
                                row_results.append({
                                    'Keyword': row['keyword'],
                                    'Cluster': row['cluster_name'],
                                    'Domain': extract_domain(result.get('link', '')),
                                    'Rank': rank,
                                    'Search Volume': row['Avg. monthly searches'],
                                    'Visibility Score': visibility_score,
                                    'CTR': ctr,
                                    'Estimated_Traffic': est_traffic
                                })
                        # If no result matched the domain filter, then include all results
                        if not row_results:
                            for rank, result in enumerate(organic_results, 1):
                                visibility_score = (101 - rank) * row['Avg. monthly searches']
                                ctr = get_ctr_by_position(rank)
                                est_traffic = ctr * row['Avg. monthly searches']
                                row_results.append({
                                    'Keyword': row['keyword'],
                                    'Cluster': row['cluster_name'],
                                    'Domain': extract_domain(result.get('link', '')),
                                    'Rank': rank,
                                    'Search Volume': row['Avg. monthly searches'],
                                    'Visibility Score': visibility_score,
                                    'CTR': ctr,
                                    'Estimated_Traffic': est_traffic
                                })
                        success = True
                    except Exception as e:
                        retry_count += 1
                        if retry_count >= max_retries:
                            st.warning(f"Error processing '{row['keyword']}' after {max_retries} attempts: {str(e)}")
                        time.sleep(2)
                all_results.extend(row_results)
        progress_bar.progress(min(1.0, (i + batch_size) / len(keywords_df)))
        if i + batch_size < len(keywords_df):
            pause_time = min(2.0, 0.2 * batch_size)
            time.sleep(pause_time)
    return pd.DataFrame(all_results) if all_results else pd.DataFrame()

def analyze_competitors(results_df, keywords_df, domains, params_template):
    """
    Analyze competitors using a reduced representative sample to minimize API calls.
    """
    if keywords_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    all_competitors = {}
    keyword_count = {}
    progress_bar = st.progress(0)
    analysis_sample_size = min(50, len(keywords_df) // 5)
    if 'cluster_name' in keywords_df.columns:
        clusters = keywords_df['cluster_name'].unique()
        samples_per_cluster = max(1, analysis_sample_size // len(clusters))
        analysis_sample = pd.DataFrame()
        for cluster in clusters:
            cluster_kws = keywords_df[keywords_df['cluster_name'] == cluster]
            top_kws = cluster_kws.sort_values('Avg. monthly searches', ascending=False).head(samples_per_cluster)
            analysis_sample = pd.concat([analysis_sample, top_kws])
    else:
        analysis_sample = keywords_df.sort_values('Avg. monthly searches', ascending=False).head(analysis_sample_size)
    for i, row in enumerate(analysis_sample.iterrows()):
        _, row_data = row
        params = params_template.copy()
        params["q"] = row_data['keyword']
        if "num" not in params:
            params["num"] = 10
        try:
            results = fetch_serp_results(row_data['keyword'], params)
            organic_results = results.get("organic_results", [])
            for rank, result in enumerate(organic_results, 1):
                domain = extract_domain(result.get('link', ''))
                if domain not in all_competitors:
                    all_competitors[domain] = {
                        'appearances': 0,
                        'total_visibility': 0,
                        'avg_position': 0,
                        'keywords': [],
                        'clusters': set()
                    }
                all_competitors[domain]['appearances'] += 1
                all_competitors[domain]['total_visibility'] += (101 - rank)
                all_competitors[domain]['avg_position'] += rank
                all_competitors[domain]['keywords'].append(row_data['keyword'])
                all_competitors[domain]['clusters'].add(row_data['cluster_name'])
                if row_data['keyword'] not in keyword_count:
                    keyword_count[row_data['keyword']] = {}
                if domain not in keyword_count[row_data['keyword']]:
                    keyword_count[row_data['keyword']][domain] = 0
                keyword_count[row_data['keyword']][domain] += 1
        except Exception as e:
            st.error(f"Error analyzing competitors for '{row_data['keyword']}': {str(e)}")
        progress_bar.progress((i + 1) / len(analysis_sample))
    competitors_df = []
    for domain, data in all_competitors.items():
        if any(target_domain in domain for target_domain in domains):
            continue
        competitors_df.append({
            'Domain': domain,
            'Appearances': data['appearances'],
            'SERP_Coverage': round(data['appearances'] / len(analysis_sample) * 100, 1),
            'Total_Visibility': data['total_visibility'],
            'Avg_Position': round(data['avg_position'] / data['appearances'], 2) if data['appearances'] > 0 else 0,
            'Keyword_Count': len(set(data['keywords'])),
            'Cluster_Count': len(data['clusters']),
            'Clusters': ', '.join(data['clusters'])
        })
    competitors_df = pd.DataFrame(competitors_df)
    if not competitors_df.empty:
        competitors_df = competitors_df.sort_values('Total_Visibility', ascending=False).head(20)
    opportunities = []
    ranking_keywords = set(results_df['Keyword'].unique()) if not results_df.empty else set()
    non_ranking_keywords = keywords_df[~keywords_df['keyword'].isin(ranking_keywords)].head(200)
    for _, row in non_ranking_keywords.iterrows():
        keyword = row['keyword']
        if keyword not in keyword_count:
            continue
        competitors = keyword_count[keyword]
        competitor_count = len(competitors)
        difficulty = min(competitor_count / 10, 1.0)
        opportunities.append({
            'Keyword': keyword,
            'Cluster': row['cluster_name'],
            'Search_Volume': row['Avg. monthly searches'],
            'Difficulty': round(difficulty, 2),
            'Opportunity_Score': round(row['Avg. monthly searches'] * (1 - difficulty), 2),
            'Competitor_Count': competitor_count
        })
    opportunities_df = pd.DataFrame(opportunities)
    if not opportunities_df.empty:
        opportunities_df = opportunities_df.sort_values('Opportunity_Score', ascending=False)
    return competitors_df, opportunities_df
