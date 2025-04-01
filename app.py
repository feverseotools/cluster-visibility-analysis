import streamlit as st
import pandas as pd
import plotly.express as px
import time
import os
import json
import yaml
from datetime import datetime
import concurrent.futures
from io import BytesIO
import re
import math
import openai

# Import custom modules
from utils.seo_calculator import (
    display_preliminary_calculator,
    detailed_cost_calculator,
    display_cost_breakdown
)
from utils.optimization import (
    cluster_representative_keywords,
    calculate_api_cost, 
    group_similar_keywords,
    estimate_processing_time
)
from utils.api_manager import (
    fetch_serp_results_optimized,
    APIKeyRotator,
    batch_process_keywords,
    APIQuotaManager
)
from utils.data_processing import (
    process_large_dataset,
    save_results_to_cache,
    load_results_from_cache,
    generate_cache_id,
    calculate_dataset_statistics,
    analyze_result_patterns
)

# Load configuration
def load_config():
    """Load configuration from config.yaml"""
    config_path = 'config.yaml'
    
    if not os.path.exists(config_path):
        st.warning("Configuration file not found. Using default values.")
        return {}
    
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except Exception as e:
            st.error(f"Error loading configuration: {str(e)}")
            return {}

# Load configuration at startup
config = load_config()

# Page configuration
st.set_page_config(page_title='SEO Visibility Estimator Pro', layout='wide')

# Caching functions
@st.cache_data(ttl=86400*7)  # Cache for 7 days
def fetch_serp_results(keyword, params):
    """Cache search results to avoid duplicate API calls"""
    return fetch_serp_results_optimized(
        keyword, 
        params, 
        use_cache=True, 
        cache_ttl=config.get('api', {}).get('serpapi', {}).get('cache_ttl', 86400)
    )

@st.cache_data(ttl=3600*24)  # Cache for 24 hours
def process_csv(uploaded_file):
    """Cache CSV processing"""
    return pd.read_csv(uploaded_file)

# Utility functions
def extract_domain(url):
    """Extract main domain from a URL"""
    pattern = r'(?:https?:\/\/)?(?:www\.)?([^\/\n]+)'
    match = re.search(pattern, url)
    return match.group(1) if match else url

def get_ctr_by_position(position):
    """Return estimated CTR by position using config"""
    ctr_model = config.get('ctr_model', {})
    default_ctr = ctr_model.get('default', 0.01)
    
    # Convert position to string for lookup if needed
    position_str = str(position)
    if position_str in ctr_model:
        return ctr_model[position_str]
    elif position in ctr_model:
        return ctr_model[position]
    else:
        return default_ctr

# Data analysis functions
def process_keywords_in_batches(keywords_df, domains, params_template, optimization_settings=None):
    """Process keywords in batches with optimized performance"""
    if optimization_settings is None:
        optimization_settings = {
            "batch_optimization": True,
            "max_retries": 3
        }
    
    # Get optimization settings
    batch_optimization = optimization_settings.get("batch_optimization", True)
    max_retries = optimization_settings.get("max_retries", 3)
    
    # Default batch size from config or fallback
    default_batch_size = config.get('optimization', {}).get('batching', {}).get('min_batch_size', 5)
    
    # Get batch size settings
    if batch_optimization and 'cluster_name' in keywords_df.columns:
        # Calculate optimized batch sizes by cluster
        from utils.optimization import optimize_batch_sizes
        batch_sizes = optimize_batch_sizes(keywords_df)
    else:
        # Use fixed batch size
        batch_sizes = {'default': default_batch_size}
    
    all_results = []
    progress_bar = st.progress(0)
    total_processed = 0
    
    # Process in batches to respect API limits
    if 'cluster_name' in keywords_df.columns and batch_optimization:
        # Process by cluster with optimized batch sizes
        clusters = keywords_df['cluster_name'].unique()
        
        for i, cluster in enumerate(clusters):
            cluster_data = keywords_df[keywords_df['cluster_name'] == cluster]
            batch_size = batch_sizes.get(cluster, batch_sizes.get('default', default_batch_size))
            
            # Process this cluster's keywords
            for j in range(0, len(cluster_data), batch_size):
                batch = cluster_data.iloc[j:j+batch_size]
                
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
                        
                        while retry_count < max_retries and not success:
                            try:
                                results = future.result()
                                organic_results = results.get("organic_results", [])
                                
                                for rank, result in enumerate(organic_results, 1):
                                    for domain in domains:
                                        if domain in result.get('link', ''):
                                            visibility_score = (101 - rank) * row['Avg. monthly searches']
                                            ctr = get_ctr_by_position(rank)
                                            est_traffic = ctr * row['Avg. monthly searches']
                                            
                                            all_results.append({
                                                'Keyword': row['keyword'],
                                                'Cluster': row['cluster_name'],
                                                'Domain': domain,
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
                                time.sleep(2)  # Wait before retry
                
                # Update progress
                total_processed += len(batch)
                progress_bar.progress(min(1.0, total_processed / len(keywords_df)))
                
                # Dynamic pause between batches based on batch size to respect API limits
                if j + batch_size < len(cluster_data):
                    pause_time = config.get('api', {}).get('serpapi', {}).get('batch_pause', 0.2)
                    pause_time = min(2.0, pause_time * batch_size)
                    time.sleep(pause_time)
            
            # Show progress message
            if len(clusters) > 5:
                st.info(f"Processed cluster '{cluster}' ({i+1}/{len(clusters)})")
    else:
        # Process all keywords with fixed batch size
        for i in range(0, len(keywords_df), default_batch_size):
            batch = keywords_df.iloc[i:i+default_batch_size]
            
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
                    
                    while retry_count < max_retries and not success:
                        try:
                            results = future.result()
                            organic_results = results.get("organic_results", [])
                            
                            for rank, result in enumerate(organic_results, 1):
                                for domain in domains:
                                    if domain in result.get('link', ''):
                                        visibility_score = (101 - rank) * row['Avg. monthly searches']
                                        ctr = get_ctr_by_position(rank)
                                        est_traffic = ctr * row['Avg. monthly searches']
                                        
                                        all_results.append({
                                            'Keyword': row['keyword'],
                                            'Cluster': row['cluster_name'],
                                            'Domain': domain,
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
                            time.sleep(2)  # Wait before retry
            
            # Update progress
            progress_bar.progress(min(1.0, (i + default_batch_size) / len(keywords_df)))
            
            # Pause between batches to respect API limits
            if i + default_batch_size < len(keywords_df):
                pause_time = config.get('api', {}).get('serpapi', {}).get('batch_pause', 0.2)
                pause_time = min(2.0, pause_time * default_batch_size)
                time.sleep(pause_time)
    
    return pd.DataFrame(all_results) if all_results else pd.DataFrame()

def calculate_advanced_metrics(results_df):
    """Calculate advanced visibility metrics"""
    if results_df.empty:
        return pd.DataFrame()
        
    # Calculate metrics by domain and cluster
    domain_metrics = results_df.groupby(['Domain', 'Cluster']).agg({
        'Keyword': 'count',
        'Search Volume': 'sum',
        'Visibility Score': 'sum',
        'Estimated_Traffic': 'sum',
        'Rank': ['mean', 'min', 'max']
    }).reset_index()
    
    # Flatten column levels
    domain_metrics.columns = ['_'.join(col).strip('_') for col in domain_metrics.columns.values]
    
    # Rename columns for clarity
    domain_metrics = domain_metrics.rename(columns={
        'Keyword_count': 'Keywords_Count',
        'Search Volume_sum': 'Total_Search_Volume',
        'Visibility_Score_sum': 'Total_Visibility_Score',
        'Estimated_Traffic_sum': 'Total_Estimated_Traffic',
        'Rank_mean': 'Average_Position',
        'Rank_min': 'Best_Position',
        'Rank_max': 'Worst_Position'
    })
    
    # Calculate "Share of Voice" (SOV) metrics
    total_visibility = results_df['Visibility Score'].sum()
    
    domain_metrics['SOV_Percentage'] = (domain_metrics['Total_Visibility_Score'] / total_visibility * 100).round(2)
    
    # Calculate improvement potential based on positions
    domain_metrics['Improvement_Potential'] = domain_metrics.apply(
        lambda x: (100 - (101 - x['Average_Position'])) * x['Total_Search_Volume'] if x['Average_Position'] > 3 else 0,
        axis=1
    )
    
    return domain_metrics

def analyze_competitors(results_df, keywords_df, domains, params_template, optimization_settings=None):
    """Analyze competitors with optimized API usage"""
    if keywords_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    if optimization_settings is None:
        optimization_settings = {
            "limit_analysis": True,
            "sample_size": 50,
            "serp_depth": 10
        }
    
    # Get optimization settings
    limit_analysis = optimization_settings.get("limit_analysis", True)
    sample_size = optimization_settings.get("sample_size", 50)
    serp_depth = optimization_settings.get("serp_depth", 10)
    
    # Extract competitors from SERP results
    all_competitors = {}
    keyword_count = {}
    progress_bar = st.progress(0)
    
    # Use a subset of keywords for competitive analysis
    if limit_analysis:
        # Get sample size from config if available
        config_sample_size = config.get('optimization', {}).get('competitor_analysis', {}).get('max_sample_size')
        if config_sample_size:
            sample_size = min(config_sample_size, len(keywords_df))
        
        # Select representative keywords for analysis
        if 'cluster_name' in keywords_df.columns:
            clusters = keywords_df['cluster_name'].unique()
            samples_per_cluster = max(1, sample_size // len(clusters))
            
            analysis_sample = pd.DataFrame()
            for cluster in clusters:
                cluster_kws = keywords_df[keywords_df['cluster_name'] == cluster]
                # Take top keywords by volume in each cluster
                top_kws = cluster_kws.sort_values('Avg. monthly searches', ascending=False).head(samples_per_cluster)
                analysis_sample = pd.concat([analysis_sample, top_kws])
                
            # Ensure we don't exceed the sample size
            if len(analysis_sample) > sample_size:
                analysis_sample = analysis_sample.sort_values('Avg. monthly searches', ascending=False).head(sample_size)
        else:
            # If no clusters, use the method from the original code
            analysis_sample = keywords_df.sort_values('Avg. monthly searches', ascending=False).head(sample_size)
    else:
        # Use all keywords (may be expensive)
        analysis_sample = keywords_df
    
    # Get SERP depth from config if available
    config_serp_depth = config.get('optimization', {}).get('competitor_analysis', {}).get('default_serp_depth')
    if config_serp_depth:
        serp_depth = config_serp_depth
    
    for i, row in enumerate(analysis_sample.iterrows()):
        _, row_data = row
        params = params_template.copy()
        params["q"] = row_data['keyword']
        params["num"] = serp_depth  # Use configured SERP depth
        
        try:
            results = fetch_serp_results(row_data['keyword'], params)
            organic_results = results.get("organic_results", [])
            
            # Record all domains in the results
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
                
                # Count domain by keyword for later analysis
                if row_data['keyword'] not in keyword_count:
                    keyword_count[row_data['keyword']] = {}
                
                if domain not in keyword_count[row_data['keyword']]:
                    keyword_count[row_data['keyword']][domain] = 0
                
                keyword_count[row_data['keyword']][domain] += 1
        except Exception as e:
            st.error(f"Error analyzing competitors for '{row_data['keyword']}': {str(e)}")
        
        # Update progress
        progress_bar.progress((i + 1) / len(analysis_sample))
    
    # Calculate final metrics
    competitors_df = []
    for domain, data in all_competitors.items():
        # Skip domains being analyzed
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
    
    # Convert to DataFrame and sort by visibility
    competitors_df = pd.DataFrame(competitors_df)
    if not competitors_df.empty:
        competitors_df = competitors_df.sort_values('Total_Visibility', ascending=False).head(20)
    
    # Identify opportunities
    opportunities = []
    
    # Keywords already ranking for target domains
    ranking_keywords = set(results_df['Keyword'].unique()) if not results_df.empty else set()
    
    # Keywords not ranking for the analyzed domains - LIMIT to 200 for efficiency
    non_ranking_limit = 200
    non_ranking_keywords = keywords_df[~keywords_df['keyword'].isin(ranking_keywords)]
    if len(non_ranking_keywords) > non_ranking_limit:
        non_ranking_keywords = non_ranking_keywords.sort_values('Avg. monthly searches', ascending=False).head(non_ranking_limit)
    
    for _, row in non_ranking_keywords.iterrows():
        keyword = row['keyword']
        
        # If no data for this keyword, continue
        if keyword not in keyword_count:
            continue
        
        # Analyze competitors for this keyword
        competitors = keyword_count[keyword]
        competitor_count = len(competitors)
        
        # Calculate "difficulty" based on number of competitors
        difficulty = min(competitor_count / 10, 1.0)  # Normalized between 0-1
        
        opportunities.append({
            'Keyword': keyword,
            'Cluster': row['cluster_name'],
            'Search_Volume': row['Avg. monthly searches'],
            'Difficulty': round(difficulty, 2),
            'Opportunity_Score': round(row['Avg. monthly searches'] * (1 - difficulty), 2),
            'Competitor_Count': competitor_count
        })
    
    # Convert to DataFrame and sort by opportunity score
    opportunities_df = pd.DataFrame(opportunities)
    
    if not opportunities_df.empty:
        opportunities_df = opportunities_df.sort_values('Opportunity_Score', ascending=False)
    
    return competitors_df, opportunities_df

# Historical tracking functions
def save_historical_data(results_df):
    """Save results for historical tracking with enhanced metadata"""
    if results_df.empty:
        return None
        
    # Get historical data directory from config or use default
    historical_dir = config.get('performance', {}).get('historical_data_dir', 'historical_data')
    
    # Create directory if it doesn't exist
    if not os.path.exists(historical_dir):
        os.makedirs(historical_dir)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{historical_dir}/seo_data_{timestamp}.json"
    
    # Create summary by domain and cluster
    summary = {}
    for domain in results_df['Domain'].unique():
        domain_data = results_df[results_df['Domain'] == domain]
        
        summary[domain] = {
            'date': datetime.now().strftime("%Y-%m-%d"),
            'total_keywords': len(domain_data['Keyword'].unique()),
            'total_volume': int(domain_data['Search Volume'].sum()),
            'total_visibility': int(domain_data['Visibility Score'].sum()),
            'avg_position': float(domain_data['Rank'].mean()),
            'clusters': {}
        }
        
        # Data by cluster
        for cluster in domain_data['Cluster'].unique():
            cluster_data = domain_data[domain_data['Cluster'] == cluster]
            
            summary[domain]['clusters'][cluster] = {
                'keywords': len(cluster_data['Keyword'].unique()),
                'volume': int(cluster_data['Search Volume'].sum()),
                'visibility': int(cluster_data['Visibility Score'].sum()),
                'avg_position': float(cluster_data['Rank'].mean())
            }
    
    # Add metadata
    summary['_meta'] = {
        'timestamp': datetime.now().isoformat(),
        'version': '1.0',
        'total_domains': len(results_df['Domain'].unique()),
        'total_keywords': len(results_df['Keyword'].unique()),
        'total_clusters': len(results_df['Cluster'].unique()),
    }
    
    # Save as JSON
    with open(filename, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return filename

def load_historical_data():
    """Load all available historical data"""
    # Get historical data directory from config or use default
    historical_dir = config.get('performance', {}).get('historical_data_dir', 'historical_data')
    
    if not os.path.exists(historical_dir):
        return []
    
    historical_data = []
    for filename in os.listdir(historical_dir):
        if filename.startswith('seo_data_') and filename.endswith('.json'):
            with open(os.path.join(historical_dir, filename), 'r') as f:
                try:
                    data = json.load(f)
                    # Extract date from filename
                    date_str = filename.replace('seo_data_', '').replace('.json', '')
                    date = datetime.strptime(date_str.split('_')[0], "%Y%m%d")
                    
                    # Add date and file to data if not already there
                    if '_meta' not in data:
                        data['_meta'] = {}
                        
                    data['_meta']['date'] = date.strftime("%Y-%m-%d")
                    data['_meta']['filename'] = filename
                    
                    historical_data.append(data)
                except Exception as e:
                    st.error(f"Error loading historical data file {filename}: {str(e)}")
                    continue
    
    # Sort by date
    historical_data.sort(key=lambda x: x['_meta'].get('date', ''))
    return historical_data

# ChatGPT integration
class ChatGPTAnalyzer:
    """Class to handle ChatGPT API interactions for SEO analysis with cost optimization"""
    
    def __init__(self, api_key, use_gpt35=True, limit_analysis=True):
        """Initialize with OpenAI API key and optimization settings"""
        self.api_key = api_key
        openai.api_key = api_key
        self.use_gpt35 = use_gpt35  # Use GPT-3.5 (more economical)
        self.limit_analysis = limit_analysis  # Limit number of analyses
        
        # Counter to limit analyses if enabled
        self.analysis_count = 0
        self.max_analyses = config.get('api', {}).get('openai', {}).get('max_analyses_per_session', 5)
    
    def analyze_serp_competitors(self, keyword, competitors):
        """
        Analyze SERP competitors for a specific keyword
        Returns insights on content strategy based on top-ranking pages
        """
        # Check limits if analysis limiting is enabled
        if self.limit_analysis:
            if self.analysis_count >= self.max_analyses:
                return "Analysis limit reached. To perform more analyses, disable limiting in settings."
            self.analysis_count += 1
        
        if not self.api_key or not competitors:
            return "API key or competitor data missing"
        
        # Reduce to 3 competitors instead of 5 to save tokens
        competitors_to_analyze = 3 if self.limit_analysis else 5
        
        # Format competitor data
        competitor_text = "\n".join([
            f"{i+1}. {comp['Domain']} (Position: {comp['Rank']})"
            for i, comp in enumerate(competitors[:competitors_to_analyze])
        ])
        
        # Optimize prompt to use fewer tokens
        prompt = f"""
        Analyze competitors for keyword "{keyword}":
        
        {competitor_text}
        
        Provide brief insights on:
        1. Key content elements
        2. Content length
        3. Structure recommendations
        4. Unique angle
        
        Use bullet points. Be concise.
        """
        
        try:
            # Select model based on configuration
            model = "gpt-3.5-turbo" if self.use_gpt35 else "gpt-4"
            
            # Reduce tokens and temperature for greater efficiency
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an SEO content strategy expert. Be concise."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Reduced for more consistent responses
                max_tokens=400 if self.limit_analysis else 500  # Reduced to save tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate_cluster_content_brief(self, cluster_name, keywords, competition_level="medium"):
        """
        Generate a content brief for a keyword cluster
        Provides recommendations for content that could rank for multiple keywords
        """
        # Check limits if analysis limiting is enabled
        if self.limit_analysis:
            if self.analysis_count >= self.max_analyses:
                return "Analysis limit reached. To perform more analyses, disable limiting in settings."
            self.analysis_count += 1
        
        if not self.api_key:
            return "API key missing"
        
        # Limit keywords to reduce token usage
        keyword_limit = 5 if self.limit_analysis else 10
        
        # Format keywords with volumes
        keywords_text = "\n".join([
            f"- {kw['Keyword']} ({kw['Search_Volume']} searches/month)"
            for kw in keywords[:keyword_limit]
        ])
        
        # Optimize prompt
        prompt = f"""
        Create a content brief for topic cluster: "{cluster_name}"
        
        Top keywords:
        {keywords_text}
        
        Competition: {competition_level}
        
        Provide:
        1. Title (include main keyword)
        2. Meta description
        3. Content outline (H2s and H3s)
        4. Key points
        5. Media suggestions
        6. Word count
        
        Focus on ranking for multiple keywords in this cluster.
        """
        
        try:
            # Select model based on configuration
            model = "gpt-3.5-turbo" if self.use_gpt35 else "gpt-4"
            
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an SEO content strategist who creates briefs."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=800 if self.limit_analysis else 1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def analyze_opportunity_gaps(self, domain, competitors, cluster_data):
        """
        Analyze gaps and opportunities compared to competitors
        Provides strategic recommendations for improvements
        """
        # Check limits if analysis limiting is enabled
        if self.limit_analysis:
            if self.analysis_count >= self.max_analyses:
                return "Analysis limit reached. To perform more analyses, disable limiting in settings."
            self.analysis_count += 1
        
        if not self.api_key:
            return "API key missing"
        
        # Limit competitors to reduce token usage
        competitor_limit = 3
        
        # Format competitor data
        competitor_text = "\n".join([
            f"- {comp['Domain']} (Visibility: {comp['Total_Visibility']}, Keywords: {comp['Keyword_Count']})"
            for comp in competitors[:competitor_limit]
        ])
        
        # Limit clusters to reduce token usage
        cluster_limit = 10 if self.limit_analysis else 20
        
        # Format cluster performance (limit to top clusters)
        sorted_clusters = sorted(
            cluster_data.items(), 
            key=lambda x: x[1].get('Total_Visibility_Score', 0), 
            reverse=True
        )[:cluster_limit]
        
        cluster_text = "\n".join([
            f"- {cluster}: Keywords: {data['Keywords_Count']}, Avg Position: {data['Average_Position']:.1f}"
            for cluster, data in sorted_clusters
        ])
        
        # Optimize prompt
        prompt = f"""
        Analyze SEO opportunities for: {domain}
        
        Top competitors:
        {competitor_text}
        
        Current performance by cluster:
        {cluster_text}
        
        Provide strategic recommendations:
        1. Key opportunity areas
        2. Priority clusters
        3. Content improvements
        4. Technical SEO considerations
        5. Quick wins vs long-term strategies
        
        Focus on actionable insights.
        """
        
        try:
            # Always use GPT-4 for strategic analysis if not limited
            model = "gpt-3.5-turbo" if self.use_gpt35 else "gpt-4"
            
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an SEO strategy consultant with expertise in competitive analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=1000 if self.limit_analysis else 1200
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

# Integration with Streamlit UI
def display_chatgpt_features(openai_api_key, results_df, competitor_df, domain_metrics, opportunity_df, 
                           use_gpt35=True, limit_analysis=True):
    """Display ChatGPT-powered analysis features in Streamlit with cost optimization"""
    st.header("AI-Powered SEO Insights")
    
    if not openai_api_key:
        st.warning("Enter your OpenAI API key in the sidebar to unlock AI-powered insights")
        return
    
    # Show information about cost savings
    if use_gpt35:
        st.success("✅ Using GPT-3.5 Turbo for analysis (more economical)")
    else:
        st.info("ℹ️ Using GPT-4 for analysis (higher quality, higher cost)")
    
    if limit_analysis:
        st.success("✅ Limiting number of analyses to reduce costs")
    
    # Initialize ChatGPT analyzer with optimization settings
    analyzer = ChatGPTAnalyzer(openai_api_key, use_gpt35=use_gpt35, limit_analysis=limit_analysis)
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Content Strategy", "Cluster Briefs", "Strategic Opportunities"])
    
    # Tab 1: Content Strategy by Keyword
    with tab1:
        st.subheader("Competitor Content Analysis")
        st.write("Analyze top-ranking competitors for specific keywords to inform your content strategy")
        
        # Keyword selector
        if not results_df.empty:
            selected_keyword = st.selectbox(
                "Select a keyword to analyze:",
                options=results_df['Keyword'].unique(),
                index=0
            )
            
            # Get competitor data for this keyword
            keyword_data = results_df[results_df['Keyword'] == selected_keyword]
            competitors = [
                {"Domain": row['Domain'], "Rank": row['Rank']}
                for _, row in keyword_data.iterrows()
            ]
            
            if st.button("Generate Content Strategy", key="content_strategy"):
                with st.spinner("Analyzing competitor content..."):
                    analysis = analyzer.analyze_serp_competitors(selected_keyword, competitors)
                    st.markdown(analysis)
        else:
            st.info("Run analysis first to see keywords")
    
    # Tab 2: Cluster Content Briefs
    with tab2:
        st.subheader("Cluster Content Briefs")
        st.write("Generate comprehensive content briefs for keyword clusters")
        
        # Cluster selector
        if not results_df.empty:
            selected_cluster = st.selectbox(
                "Select a cluster:",
                options=results_df['Cluster'].unique(),
                index=0
            )
            
            # Get cluster data
            cluster_keywords = results_df[results_df['Cluster'] == selected_cluster]
            
            # Convert to format needed for the brief
            keywords_for_brief = [
                {"Keyword": row['Keyword'], "Search_Volume": row['Search Volume']}
                for _, row in cluster_keywords.iterrows()
            ]
            
            competition = st.select_slider(
                "Estimated competition level:",
                options=["low", "medium", "high"],
                value="medium"
            )
            
            if st.button("Generate Content Brief", key="cluster_brief"):
                with st.spinner("Creating content brief..."):
                    brief = analyzer.generate_cluster_content_brief(
                        selected_cluster, 
                        keywords_for_brief,
                        competition
                    )
                    st.markdown(brief)
        else:
            st.info("Run analysis first to see clusters")
    
    # Tab 3: Strategic Opportunities
    with tab3:
        st.subheader("Strategic Opportunity Analysis")
        st.write("Get AI-powered strategic recommendations based on competitive analysis")
        
        if not results_df.empty and not competitor_df.empty and not domain_metrics.empty:
            # Domain selector
            selected_domain = st.selectbox(
                "Select your domain:",
                options=results_df['Domain'].unique(),
                index=0
            )
            
            # Create cluster performance summary
            cluster_performance = {}
            for _, row in domain_metrics[domain_metrics['Domain'] == selected_domain].iterrows():
                cluster_performance[row['Cluster']] = {
                    "Keywords_Count": row['Keywords_Count'],
                    "Average_Position": row['Average_Position'],
                    "Total_Visibility_Score": row['Total_Visibility_Score']
                }
            
            if st.button("Generate Strategic Analysis", key="strategy"):
                with st.spinner("Analyzing strategic opportunities..."):
                    strategy = analyzer.analyze_opportunity_gaps(
                        selected_domain,
                        competitor_df.to_dict('records'),
                        cluster_performance
                    )
                    st.markdown(strategy)
        else:
            st.info("Run a complete analysis first to access strategic insights")
