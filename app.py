# At the beginning of your file, change the imports section:
import streamlit as st
import pandas as pd
import numpy as np
import logging
import os
import json
import hashlib
import time
import glob
from datetime import datetime, timedelta
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import openai  # For AI-based SEO recommendations
import requests  # For fallback SERPAPI queries

# Try to import aiohttp, but provide a fallback if it's not available
try:
    import asyncio
    import aiohttp
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False
    logging.warning("aiohttp package not available. Using synchronous processing only.")


# Configure logging for debugging
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)

# -------------------------------------------
# UTILITY MODULES
# -------------------------------------------
class DataProcessing:
    @staticmethod
    def clean_domain(domain: str) -> str:
        """Clean a domain string by removing protocols and www prefix."""
        domain = domain.strip().replace("http://", "").replace("https://", "")
        domain = domain.split("/")[0]
        return domain[4:] if domain.startswith("www.") else domain.lower()

    @staticmethod
    def extract_domain(url: str) -> str:
        """Extract domain from a URL."""
        if not url:
            return ""
        try:
            netloc = urlparse(url).netloc.lower()
            return netloc[4:] if netloc.startswith("www.") else netloc
        except Exception as e:
            logging.error(f"Error extracting domain from {url}: {e}")
            return ""

    @staticmethod
    def clean_csv_columns(df):
        """Standardize CSV column names and handle common issues."""
        if df.empty:
            return df
            
        df.columns = [col.strip().lower() for col in df.columns]
        
        # Find and rename common columns
        keyword_cols = [col for col in df.columns if 'keyword' in col.lower()]
        cluster_cols = [col for col in df.columns if 'cluster' in col.lower()]
        volume_cols = [col for col in df.columns if 'search' in col.lower() or 'volume' in col.lower() or 'monthly' in col.lower()]
        
        if keyword_cols:
            df = df.rename(columns={keyword_cols[0]: 'keyword'})
        if cluster_cols:
            df = df.rename(columns={cluster_cols[0]: 'cluster_name'})
        if volume_cols:
            df = df.rename(columns={volume_cols[0]: 'avg_monthly_searches'})
            
        # Handle volume data conversion
        try:
            df['avg_monthly_searches'] = (df['avg_monthly_searches'].astype(str)
                                         .str.replace('K', '000')
                                         .str.replace(',', ''))
            df['avg_monthly_searches'] = pd.to_numeric(df['avg_monthly_searches'], errors='coerce').fillna(10)
        except Exception as vol_err:
            logging.warning(f"Error converting volume data: {vol_err}. Using fallback values.")
            df['avg_monthly_searches'] = 10
            
        return df
    
    @staticmethod
    def select_representative_keywords(df, samples_per_cluster=3, min_volume_samples=1):
        """Select representative keywords from each cluster based on volume."""
        selected_keywords = []
        for cluster in df['cluster_name'].unique():
            cluster_df = df[df['cluster_name'] == cluster].copy()
            if len(cluster_df) <= samples_per_cluster:
                selected_keywords.append(cluster_df)
                continue
            cluster_df = cluster_df.sort_values('avg_monthly_searches', ascending=False)
            top_keywords = cluster_df.head(samples_per_cluster // 2)
            selected_keywords.append(top_keywords)
            if len(cluster_df) > samples_per_cluster:
                mid_start = len(cluster_df) // 2 - samples_per_cluster // 4
                mid_keywords = cluster_df.iloc[mid_start:mid_start + samples_per_cluster // 4]
                selected_keywords.append(mid_keywords)
            if len(cluster_df) > samples_per_cluster * 2:
                low_keywords = cluster_df.tail(min_volume_samples)
                selected_keywords.append(low_keywords)
        result = pd.concat(selected_keywords)
        result = result.drop_duplicates(subset=['keyword'])
        return result

    @staticmethod
    def filter_results(results, min_volume=0, ranking_filter="All", sort_by="Volume", target_domains=None):
        """Filter results based on user criteria."""
        filtered = []
        for entry in results:
            # Apply volume filter
            if float(entry.get("volume", 0)) < min_volume:
                continue
                
            # Apply ranking filter
            if ranking_filter != "All":
                domain_positions = [pos for domain, pos in entry["rankings"].items() 
                                   if domain in target_domains and pos is not None]
                if not domain_positions:
                    if ranking_filter == "Ranking":
                        continue
                elif ranking_filter == "Not Ranking":
                    continue
                elif ranking_filter == "Top 3" and min(domain_positions) > 3:
                    continue
                elif ranking_filter == "Top 10" and min(domain_positions) > 10:
                    continue
                    
            filtered.append(entry)
            
        # Apply sorting
        if sort_by == "Volume":
            filtered.sort(key=lambda x: float(x.get("volume", 0)), reverse=True)
        elif sort_by == "Position":
            def get_best_position(entry):
                positions = [pos for domain, pos in entry["rankings"].items() 
                            if domain in target_domains and pos is not None]
                return min(positions) if positions else float('inf')
            filtered.sort(key=get_best_position)
        elif sort_by == "Opportunity":
            for entry in filtered:
                entry["opportunity_score"] = SEOCalculator.calculate_opportunity_score(entry, target_domains)
            filtered.sort(key=lambda x: x.get("opportunity_score", 0), reverse=True)
            
        return filtered

class ApiCache:
    """Cache system for SERPAPI results to reuse previous queries and reduce API usage."""
    def __init__(self, cache_dir='.cache', expiry_days=7):
        self.cache_dir = cache_dir
        self.expiry_days = expiry_days
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_key(self, keyword, api_params=None):
        cache_str = keyword.lower()
        if api_params:
            cache_str += json.dumps(api_params, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def get_cache_path(self, cache_key):
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def get(self, keyword, api_params=None):
        cache_key = self.get_cache_key(keyword, api_params)
        cache_path = self.get_cache_path(cache_key)
        if not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            cache_date = datetime.fromisoformat(cache_data.get('timestamp', '2000-01-01'))
            if datetime.now() - cache_date > timedelta(days=self.expiry_days):
                return None
            return cache_data.get('data')
        except Exception as e:
            logging.warning(f"Error reading cache for '{keyword}': {e}")
            return None
    
    def set(self, keyword, data, api_params=None):
        if not data:
            return False
        cache_key = self.get_cache_key(keyword, api_params)
        cache_path = self.get_cache_path(cache_key)
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'keyword': keyword,
                'data': data
            }
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
            return True
        except Exception as e:
            logging.warning(f"Error saving cache for '{keyword}': {e}")
            return False

class ApiManager:
    """Handles all external API interactions."""
    @staticmethod
    async def search_query_async(session, keyword, api_key, params):
        """Asynchronous SERPAPI query."""
        try:
            async with session.get("https://serpapi.com/search", params=params) as response:
                if response.status == 200:
                    return await response.json(), keyword
                else:
                    error_text = await response.text()
                    logging.error(f"SERPAPI request error for '{keyword}': {response.status} - {error_text}")
                    return None, keyword
        except Exception as e:
            logging.error(f"Error during SERPAPI request for '{keyword}': {e}")
            return None, keyword
    
    @staticmethod
    def search_query(keyword, api_key, params):
        """Synchronous fallback for SERPAPI query."""
        try:
            response = requests.get("https://serpapi.com/search", params=params)
            if response.status_code == 200:
                return response.json()
            else:
                logging.error(f"SERPAPI request error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logging.error(f"Error during SERPAPI request: {e}")
            return None
            
    @staticmethod
    async def process_keywords_batch(df, api_key, api_params_base, start_idx, end_idx, api_cache=None):
        """Process a batch of keywords asynchronously."""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for idx in range(start_idx, min(end_idx, len(df))):
                row = df.iloc[idx]
                keyword = str(row["keyword"]).strip()
                if not keyword:
                    continue
                
                # Check cache first
                cached_data = None
                if api_cache:
                    api_params = api_params_base.copy()
                    api_params['q'] = keyword
                    cached_data = api_cache.get(keyword, api_params)
                
                if cached_data:
                    # Skip API call if cached data exists
                    tasks.append((cached_data, keyword))
                else:
                    # Make API call
                    api_params = api_params_base.copy()
                    api_params['q'] = keyword
                    task = ApiManager.search_query_async(session, keyword, api_key, api_params)
                    tasks.append(asyncio.ensure_future(task))
            
            # Wait for all tasks to complete
            results = []
            for task in tasks:
                if isinstance(task, tuple):
                    # This is cached data
                    results.append(task)
                else:
                    # This is an API call task
                    result = await task
                    results.append(result)
            
            return results
    
    @staticmethod
    def generate_ai_content(prompt, model="gpt-4", temperature=0.7):
        """Generic function to generate content using OpenAI API."""
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            return "AI content generation unavailable."

class SEOCalculator:
    """Handles all SEO metric calculations."""
    # CTR values by position
    CTR_MAP = {
        1: 0.3042, 2: 0.1559, 3: 0.0916, 4: 0.0651, 5: 0.0478,
        6: 0.0367, 7: 0.0289, 8: 0.0241, 9: 0.0204, 10: 0.0185,
        11: 0.0156, 12: 0.0138, 13: 0.0122, 14: 0.0108, 15: 0.0096
    }
    
    @staticmethod
    def simple_score(position):
        """Calculate a simple visibility score based on position."""
        return 31 - position if (position is not None and position <= 30) else 0
    
    @staticmethod
    def calculate_simple_visibility_for_domain(results, domain):
        """Calculate simple visibility score for a domain across all keywords."""
        score_sum = 0
        for entry in results:
            pos = entry["rankings"].get(domain)
            score_sum += SEOCalculator.simple_score(pos)
        return score_sum
    
    @staticmethod
    def calculate_weighted_visibility(filtered_results):
        """Calculate weighted visibility based on CTR and search volume."""
        total_volume = sum(float(entry.get("volume", 0)) for entry in filtered_results)
        captured_clicks = 0
        for entry in filtered_results:
            vol = float(entry.get("volume", 0))
            pos = entry.get("domain_rank")
            if pos is not None and int(pos) in SEOCalculator.CTR_MAP:
                captured_clicks += vol * SEOCalculator.CTR_MAP[int(pos)]
        max_clicks = total_volume * SEOCalculator.CTR_MAP[1] if total_volume > 0 else 0
        return round((captured_clicks / max_clicks * 100) if max_clicks > 0 else 0, 2)
    
    @staticmethod
    def calculate_weighted_visibility_for_domain(results, domain):
        """Calculate weighted visibility for a specific domain."""
        filtered = []
        for entry in results:
            pos = entry["rankings"].get(domain)
            if pos is not None:
                filtered.append({
                    "keyword": entry["keyword"],
                    "cluster": entry["cluster"],
                    "volume": entry["volume"],
                    "domain_rank": pos
                })
        return SEOCalculator.calculate_weighted_visibility(filtered)
    
    @staticmethod
    def calculate_opportunity_score(keyword_data, target_domains=None):
        """Calculate an opportunity score based on volume and current rankings."""
        volume = float(keyword_data.get("volume", 0))
        
        # Get best position among target domains
        if target_domains:
            positions = [pos for domain, pos in keyword_data["rankings"].items() 
                        if domain in target_domains and pos is not None]
        else:
            positions = [pos for pos in keyword_data["rankings"].values() if pos is not None]
            
        best_position = min(positions) if positions else None
        
        if best_position is None:
            opportunity = volume * 1.0  # High opportunity if not ranking
        elif best_position <= 3:
            opportunity = volume * 0.1  # Low opportunity if already ranking well
        elif best_position <= 10:
            opportunity = volume * 0.5  # Medium opportunity
        else:
            opportunity = volume * 0.8  # High opportunity for positions beyond page 1
            
        return round(opportunity, 2)
    
    @staticmethod
    def analyze_top_ranking_domains(results, target_domains):
        """Identify top-ranking domains that are not your target domains."""
        top_domains = {}
        for entry in results:
            serp_results = entry.get("serp_results", [])
            if not serp_results:
                continue
                
            for idx, result in enumerate(serp_results[:10]):
                url = result.get("link", "")
                domain = DataProcessing.extract_domain(url)
                if domain and domain not in target_domains:
                    top_domains[domain] = top_domains.get(domain, 0) + (10 - idx)
                    
        return sorted(top_domains.items(), key=lambda x: x[1], reverse=True)
    
    @staticmethod
    def identify_content_gaps(results, target_domains, competitor_domains):
        """Identify keywords where competitors rank but target domains don't."""
        gaps = []
        for entry in results:
            rankings = entry["rankings"]
            has_competitor = any(rankings.get(domain) is not None and rankings.get(domain) <= 10 
                                for domain in competitor_domains)
            has_target = any(rankings.get(domain) is not None and rankings.get(domain) <= 10 
                             for domain in target_domains)
            if has_competitor and not has_target:
                gaps.append(entry)
        return gaps

class SEORecommendations:
    """Generates SEO recommendations using AI."""
    @staticmethod
    def generate_seo_suggestions(keyword):
        """Generate SEO suggestions for a keyword."""
        prompt = (
            f"Keyword: \"{keyword}\"\n"
            "1. **Intent**: Identify the search intent (Informational, Navigational, Transactional, or Commercial) and briefly explain why.\n"
            "2. **Content Type**: Recommend the best type of content (e.g., blog article, product page, landing page).\n"
            "3. **Title Tag**: Propose an SEO-optimized title tag (<= 60 characters) for this keyword.\n"
            "4. **Meta Description**: Draft a meta description (<= 155 characters) incorporating the keyword and enticing clicks.\n"
            "5. **CTA**: Provide a short call-to-action suitable for this page.\n"
            "6. **Content Outline**: Outline key sections for a comprehensive article or landing page based on the keyword's intent.\n"
        )
        return ApiManager.generate_ai_content(prompt)
    
    @staticmethod
    def generate_cluster_strategy(cluster_name, keywords):
        """Generate an SEO strategy for a cluster of keywords."""
        prompt = (
            f"Cluster: {cluster_name}\n"
            f"Keywords: {', '.join(keywords)}\n"
            "You are an SEO strategist. Provide a summary SEO content strategy for these keywords, including search intent analysis, recommended content types, meta title and description guidelines, and call-to-action suggestions."
        )
        return ApiManager.generate_ai_content(prompt)
    
    @staticmethod
    def analyze_serp_features(keyword, features):
        """Analyze SERP features for a keyword and provide recommendations."""
        prompt = (
            f"Analyze the following SERP features for the keyword '{keyword}':\n{json.dumps(features)}\n"
            "Based on these features (such as organic results, local pack, knowledge graph, video results, ads, etc.), "
            "provide an analysis of the SERP result types and actionable SEO recommendations for improving ranking and click-through."
        )
        return ApiManager.generate_ai_content(prompt)
    
    @staticmethod
    def generate_actionable_recommendations(keyword, current_position, competitor_positions):
        """Generate specific actionable recommendations to improve ranking."""
        prompt = (
            f"Keyword: '{keyword}'\n"
            f"Current position: {current_position}\n"
            f"Competitor positions: {competitor_positions}\n\n"
            "Based on this information, provide 3 specific, actionable recommendations to improve ranking for this keyword. "
            "Include recommendations about content structure, internal linking, and technical SEO factors."
        )
        return ApiManager.generate_ai_content(prompt)
    
    @staticmethod
    def generate_action_plan(results, target_domains):
        """Generate a prioritized action plan based on opportunity scores."""
        # Calculate opportunity scores
        for entry in results:
            entry["opportunity_score"] = SEOCalculator.calculate_opportunity_score(entry, target_domains)
        
        # Sort by opportunity
        prioritized = sorted(results, key=lambda x: x["opportunity_score"], reverse=True)
        
        # Generate cluster-level action plans
        cluster_plans = {}
        for entry in prioritized[:50]:  # Top 50 opportunities
            cluster = entry["cluster"]
            if cluster not in cluster_plans:
                cluster_plans[cluster] = {
                    "total_opportunity": 0,
                    "keywords": [],
                    "actions": []
                }
            
            cluster_plans[cluster]["total_opportunity"] += entry["opportunity_score"]
            cluster_plans[cluster]["keywords"].append(entry["keyword"])
            
            # Add to actions if this is a high-value keyword
            if entry["opportunity_score"] > 100:  # Arbitrary threshold
                positions = {d: entry["rankings"].get(d) for d in target_domains if entry["rankings"].get(d) is not None}
                best_domain = min(positions.items(), key=lambda x: x[1])[0] if positions else None
                
                if best_domain:
                    action = f"Optimize {best_domain} for '{entry['keyword']}' (currently position {positions[best_domain]})"
                else:
                    action = f"Create new content targeting '{entry['keyword']}'"
                    
                cluster_plans[cluster]["actions"].append({
                    "action": action,
                    "opportunity": entry["opportunity_score"],
                    "keyword": entry["keyword"]
                })
        
        # Sort clusters by total opportunity
        return sorted(cluster_plans.items(), key=lambda x: x[1]["total_opportunity"], reverse=True)

class HistoricalData:
    """Handles saving and loading historical data."""
    @staticmethod
    def save_historical_data(results, filename=None):
        """Save current results for historical tracking."""
        if filename is None:
            filename = f"visibility_history_{datetime.now().strftime('%Y%m%d')}.json"
        
        os.makedirs("history", exist_ok=True)
        data = {
            "date": datetime.now().isoformat(),
            "results": results
        }
        
        with open(os.path.join("history", filename), "w") as f:
            json.dump(data, f)
            
    @staticmethod
    def load_historical_data(days_back=30):
        """Load historical data for the specified period."""
        if not os.path.exists("history"):
            return []
            
        history_files = sorted(glob.glob("history/visibility_history_*.json"))
        history_files = [f for f in history_files if (
            datetime.now() - datetime.strptime(os.path.basename(f).split("_")[-1].split(".")[0], "%Y%m%d")
        ).days <= days_back]
        
        history = []
        for file in history_files:
            try:
                with open(file, "r") as f:
                    history.append(json.load(f))
            except Exception as e:
                logging.error(f"Error loading historical data from {file}: {e}")
                
        return history

class Visualization:
    """Handles generation of visualizations."""
    @staticmethod
    def create_ranking_distribution_chart(results, all_domains):
        """Create a ranking position distribution chart."""
        fig, ax = plt.subplots(figsize=(10, 6))
        for domain in all_domains:
            domain_data = [entry["rankings"].get(domain) for entry in results 
                          if entry["rankings"].get(domain) is not None]
            if domain_data:
                sns.histplot(domain_data, bins=list(range(1, 16)), alpha=0.5, label=domain, ax=ax)
        ax.set_title('SERP Position Distribution')
        ax.set_xlabel('Position')
        ax.set_ylabel('Number of Keywords')
        ax.legend()
        return fig
    
    @staticmethod
    def create_cluster_visibility_chart(cluster_stats, all_domains):
        """Create a cluster visibility comparison chart."""
        cluster_vis_data = []
        for cluster, stats in cluster_stats.items():
            row = {"Cluster": cluster}
            for domain in all_domains:
                pct = (stats["weighted"][domain] / stats["total_volume"] * 100) if stats["total_volume"] > 0 else 0
                row[domain] = round(pct, 2)
            cluster_vis_data.append(row)
        cluster_vis_df = pd.DataFrame(cluster_vis_data)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        x = range(len(cluster_vis_df['Cluster']))
        width = 0.8 / len(all_domains)
        for i, domain in enumerate(all_domains):
            positions = [j + i * width for j in x]
            ax.bar(positions, cluster_vis_df[domain], width, label=domain)
        ax.set_ylabel('Visibility (%)')
        ax.set_title('Cluster Visibility Comparison (Weighted)')
        ax.set_xticks([j + (width*(len(all_domains)-1)/2) for j in x])
        ax.set_xticklabels(cluster_vis_df['Cluster'], rotation=45, ha='right')
        ax.legend()
        fig.tight_layout()
        return fig
    
    @staticmethod
    def create_interactive_position_chart(results, domains):
        """Create an interactive position distribution chart."""
        data = []
        for domain in domains:
            domain_positions = [r["rankings"].get(domain) for r in results 
                              if r["rankings"].get(domain) is not None]
            if domain_positions:
                data.append(go.Histogram(
                    x=domain_positions,
                    name=domain,
                    nbinsx=20,
                    opacity=0.7
                ))
        
        fig = go.Figure(data=data)
        fig.update_layout(
            title="SERP Position Distribution",
            xaxis_title="Position",
            yaxis_title="Number of Keywords",
            barmode='overlay'
        )
        return fig
    
    @staticmethod
    def create_historical_trend_chart(history_data, domains):
        """Create a chart showing visibility trends over time."""
        if not history_data:
            return None
            
        trend_data = []
        for entry in history_data:
            date = datetime.fromisoformat(entry["date"]).strftime("%Y-%m-%d")
            visibility = {}
            for domain in domains:
                domain_results = [r for r in entry["results"] if r["rankings"].get(domain) is not None]
                visibility[domain] = SEOCalculator.calculate_weighted_visibility_for_domain(entry["results"], domain)
            trend_data.append({"date": date, **visibility})
            
        df = pd.DataFrame(trend_data)
        
        fig = go.Figure()
        for domain in domains:
            fig.add_trace(go.Scatter(
                x=df["date"],
                y=df[domain],
                mode='lines+markers',
                name=domain
            ))
            
        fig.update_layout(
            title="Visibility Trend Over Time",
            xaxis_title="Date",
            yaxis_title="Weighted Visibility (%)"
        )
        return fig

# -------------------------------------------
# MAIN APPLICATION
# -------------------------------------------
def main():
    st.set_page_config(page_title="Enhanced SEO Visibility Estimator", layout="wide")
    
    st.title("Enhanced SEO Visibility Estimator")
    st.markdown("""
    Upload a CSV of keywords (with **keyword**, **cluster_name**, **Avg. monthly searches** columns) and configure options to estimate SEO visibility.
    This tool compares your target domains against competitor domains by querying Google SERPs via SERPAPI and provides AI-driven recommendations.
    """)
    
    # -------------------------------------------
    # Domain Inputs
    # -------------------------------------------
    col1, col2 = st.columns(2)
    with col1:
        target_domains_input = st.text_input("Target Domains (comma separated)", "")
        target_domains = [DataProcessing.clean_domain(d) for d in target_domains_input.split(",") if d.strip()]
    
    with col2:
        competitor_domains_input = st.text_input("Competitor Domains (comma separated)", "")
        competitor_domains = [DataProcessing.clean_domain(d) for d in competitor_domains_input.split(",") if d.strip()]

    # Combine into one list for analysis
    all_domains = target_domains + competitor_domains

    if not target_domains:
        st.warning("Please provide at least one target domain.")
        st.stop()

    # -------------------------------------------
    # File Input
    # -------------------------------------------
    uploaded_file = st.file_uploader("Upload Keyword CSV", type=["csv"])
    if not uploaded_file:
        st.info("Please upload a CSV file to continue.")
        st.stop()

    try:
        df = pd.read_csv(uploaded_file)
        if df.empty or len(df.columns) <= 1:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=';')
        
        # Clean and standardize columns
        df = DataProcessing.clean_csv_columns(df)
        
        required_cols = {'keyword', 'cluster_name'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            st.error(f"CSV is missing required columns: {missing}. Found columns: {list(df.columns)}.")
            st.stop()
        
        df = df.dropna(subset=['keyword'])
        
        # Display basic statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Keywords", len(df))
        with col2:
            st.metric("Total Clusters", len(df['cluster_name'].unique()))
        with col3:
            st.metric("Total Search Volume", f"{int(df['avg_monthly_searches'].sum()):,}")
        
        with st.expander("Preview of loaded data"):
            st.write(df.head())
    except Exception as e:
        st.error(f"Error processing CSV file: {e}")
        st.stop()

    # -------------------------------------------
    # Additional Inputs
    # -------------------------------------------
    st.markdown("#### API & Localization Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        country_code = st.text_input("Country Code (gl)", "us")
    with col2:
        language_code = st.text_input("Language Code (hl)", "en")
    with col3:
        location_input = st.text_input("City/Location (optional)", "")
    
    col1, col2 = st.columns(2)
    with col1:
        api_key = st.text_input("SERPAPI API Key", type="password", help="Your SERPAPI key is required.")
    with col2:
        openai_api_key = st.text_input("OpenAI API Key (optional)", type="password", 
                                       help="Provide your OpenAI key for AI-driven suggestions.")
    
    if not api_key:
        st.warning("Please provide a SERPAPI API key to run the analysis.")
        st.stop()
        
    if openai_api_key:
        openai.api_key = openai_api_key
    else:
        st.info("OpenAI API key not provided. AI-based suggestions will be skipped.")
    
    # -------------------------------------------
    #
# -------------------------------------------
    # Filtering Options
    # -------------------------------------------
    st.markdown("#### Filtering Options")
    
    col1, col2 = st.columns(2)
    with col1:
        cluster_names = sorted(df["cluster_name"].unique())
        selected_clusters = st.multiselect("Filter by Clusters (optional)", cluster_names)
        if selected_clusters:
            df = df[df["cluster_name"].isin(selected_clusters)]
            st.write(f"Filtered to {len(selected_clusters)} clusters. Keywords remaining: {len(df)}")
    
    with col2:
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            min_volume = st.number_input("Min Search Volume", min_value=0, value=0)
        with filter_col2:
            ranking_filter = st.selectbox("Ranking Status (for filtering results)", 
                                       ["All", "Ranking", "Not Ranking", "Top 3", "Top 10"])
    
    if df.empty:
        st.warning("No keywords to process after applying cluster filter.")
        st.stop()
    
    # -------------------------------------------
    # Cost Optimization Options
    # -------------------------------------------
    st.markdown("#### Cost & Performance Options")
    
    col1, col2 = st.columns(2)
    with col1:
        use_cost_opt = st.checkbox("Use cost optimization (query only representative keywords per cluster)", value=False)
        if use_cost_opt:
            col1a, col1b = st.columns(2)
            with col1a:
                samples_per_cluster = st.slider("Keywords per cluster", min_value=1, max_value=10, value=3)
            with col1b:
                min_volume_keywords = st.slider("Min low-volume keywords", min_value=1, max_value=5, value=1)
            
            try:
                original_count = len(df)
                df = DataProcessing.select_representative_keywords(df, samples_per_cluster, min_volume_keywords)
                st.write(f"Cost Optimization enabled: reduced {original_count} keywords to {len(df)} representative keywords.")
                st.write(df.groupby('cluster_name').size().reset_index(name='keyword_count'))
            except Exception as opt_e:
                st.error(f"Error during cost optimization: {opt_e}. Proceeding with all keywords.")
    
    with col2:
        cache_control = st.checkbox("Use cache for API requests (reduces API usage)", value=True)
        if cache_control:
            col2a, col2b = st.columns(2)
            with col2a:
                cache_days = st.slider("Cache expiry (days)", min_value=1, max_value=30, value=7)
            with col2b:
                clear_cache = st.button("Clear existing cache")
            
            api_cache = ApiCache(expiry_days=cache_days)
            if clear_cache:
                try:
                    import shutil
                    shutil.rmtree(api_cache.cache_dir, ignore_errors=True)
                    os.makedirs(api_cache.cache_dir, exist_ok=True)
                    st.success("Cache cleared successfully.")
                except Exception as cache_err:
                    st.error(f"Error clearing cache: {cache_err}")
        else:
            api_cache = None
    
    if ASYNC_AVAILABLE:
        use_async = st.checkbox("Use asynchronous processing (faster but more resource-intensive)", value=True)
        if use_async:
            batch_size = st.slider("Batch size for async requests", min_value=5, max_value=50, value=20)
        else:
            batch_size = 1
    else:
        use_async = False
        st.info("Asynchronous processing not available. Install 'aiohttp' package for this feature.")
        batch_size = 1
    
# -------------------------------------------
# Button to Run Analysis
# -------------------------------------------
    historical_data = None
    if os.path.exists("history"):
        show_history = st.checkbox("Load historical data for trend analysis", value=False)
        if show_history:
            days_back = st.slider("Days to look back", min_value=1, max_value=90, value=30)
            historical_data = HistoricalData.load_historical_data(days_back)
            if historical_data:
                st.write(f"Loaded {len(historical_data)} historical data points")
            else:
                st.info("No historical data found")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        run_analysis = st.button("Run Analysis", type="primary")
    with col2:
        save_history = st.checkbox("Save results as historical data", value=True)
    with col3:
        sort_results_by = st.selectbox("Sort Results By", ["Volume", "Position", "Opportunity"])
    
    if not run_analysis:
        st.info("Click 'Run Analysis' to start processing the data.")
        st.stop()
    
    # -------------------------------------------
    # SERPAPI Query Execution
    # -------------------------------------------
    st.subheader("SERPAPI Data Retrieval")
    progress_bar = st.progress(0)
    results = []
    debug_messages = []
    success_count = 0
    no_result_count = 0
    fail_count = 0
    cache_hits = 0
    
    # API parameters
    api_params_base = {
        'engine': 'google',
        'api_key': api_key,
        'gl': country_code,
        'hl': language_code
    }
    if location_input:
        api_params_base['location'] = location_input
    
    # Process keywords - either asynchronously or synchronously
    if use_async and ASYNC_AVAILABLE:
        # Process in batches asynchronously
        with st.spinner("Processing keywords asynchronously in batches..."):
            for start_idx in range(0, len(df), batch_size):
                end_idx = start_idx + batch_size
                progress_percentage = min(start_idx / len(df), 1.0)
                progress_bar.progress(progress_percentage)
                
                batch_results = asyncio.run(
                    ApiManager.process_keywords_batch(df, api_key, api_params_base, start_idx, end_idx, api_cache)
                )
                
                for result, keyword in batch_results:
                    # Get corresponding row data
                    row_data = df[df["keyword"] == keyword].iloc[0] if not df[df["keyword"] == keyword].empty else None
                    
                    if row_data is None:
                        continue
                        
                    cluster = str(row_data["cluster_name"]).strip()
                    volume = float(row_data["avg_monthly_searches"])
                    
                    # Check if this was a cache hit
                    using_cache = isinstance(result, dict) and api_cache is not None
                    if using_cache:
                        cache_hits += 1
                        debug_messages.append(f"Keyword '{keyword}': Using cached data.")
                        serp_data = result
                    else:
                        serp_data = result
                    
                    # Extract SERP features
                    features = {}
                    if isinstance(serp_data, dict):
                        for feature_key in ["local_results", "knowledge_graph", "video_results", "ads", "inline_related_results"]:
                            if feature_key in serp_data:
                                features[feature_key] = serp_data[feature_key]
                    
                    if serp_data is None:
                        no_result_count += 1
                        debug_messages.append(f"Keyword '{keyword}': No results returned.")
                        results.append({
                            "keyword": keyword,
                            "cluster": cluster,
                            "volume": volume,
                            "rankings": {domain: None for domain in all_domains},
                            "features": features
                        })
                    else:
                        serp_results = serp_data.get("organic_results", serp_data.get("results", [])) if isinstance(serp_data, dict) else []
                        if isinstance(serp_data, dict) and serp_data.get("error"):
                            fail_count += 1
                            debug_messages.append(f"Keyword '{keyword}': API error - {serp_data.get('error')}")
                            serp_results = []
                        if serp_results is None:
                            serp_results = []
                            
                        # Extract rankings for all domains
                        keyword_rankings = {domain: None for domain in all_domains}
                        for res_index, res in enumerate(serp_results):
                            url = res if isinstance(res, str) else (res.get("link") or res.get("displayed_link") or "")
                            if not url:
                                continue
                            netloc = DataProcessing.extract_domain(url)
                            for domain in all_domains:
                                if keyword_rankings[domain] is None and netloc.endswith(domain):
                                    keyword_rankings[domain] = res.get("position", res_index + 1)
                            if all(val is not None for val in keyword_rankings.values()):
                                break
                                
                        if serp_results:
                            success_count += 1
                            debug_messages.append(f"Keyword '{keyword}': Retrieved {len(serp_results)} results. Rankings: {keyword_rankings}")
                        else:
                            no_result_count += 1
                            debug_messages.append(f"Keyword '{keyword}': No organic results found.")
                            
                        results.append({
                            "keyword": keyword,
                            "cluster": cluster,
                            "volume": volume,
                            "rankings": keyword_rankings,
                            "features": features,
                            "serp_results": serp_results[:10]  # Store top 10 results for analysis
                        })
                
                # Small delay to avoid hitting rate limits
                time.sleep(0.5)
    else:
        # Process sequentially
        for idx, row in df.iterrows():
            keyword = str(row["keyword"]).strip()
            cluster = str(row["cluster_name"]).strip()
            try:
                volume = float(row["avg_monthly_searches"])
            except Exception:
                volume = 0.0
            if not keyword:
                logging.warning(f"Skipping empty keyword at index {idx}.")
                continue
    
            serp_data = None
            using_cache = False
            api_params = api_params_base.copy()
            api_params['q'] = keyword
    
            if api_cache:
                cached_data = api_cache.get(keyword, api_params)
                if cached_data:
                    serp_data = cached_data
                    using_cache = True
                    cache_hits += 1
                    debug_messages.append(f"Keyword '{keyword}': Using cached data.")
            if not using_cache:
                try:
                    serp_data = ApiManager.search_query(keyword, api_key=api_key, params=api_params)
                    if api_cache and serp_data:
                        api_cache.set(keyword, serp_data, api_params)
                except Exception as e:
                    fail_count += 1
                    debug_messages.append(f"Keyword '{keyword}': API request failed - {e}")
                    serp_data = None
    
            # Extract additional SERP features if available
            features = {}
            if isinstance(serp_data, dict):
                for feature_key in ["local_results", "knowledge_graph", "video_results", "ads", "inline_related_results"]:
                    if feature_key in serp_data:
                        features[feature_key] = serp_data[feature_key]
    
            if serp_data is None:
                no_result_count += 1
                debug_messages.append(f"Keyword '{keyword}': No results returned.")
                results.append({
                    "keyword": keyword,
                    "cluster": cluster,
                    "volume": volume,
                    "rankings": {domain: None for domain in all_domains},
                    "features": features
                })
            else:
                serp_results = serp_data.get("organic_results", serp_data.get("results", [])) if isinstance(serp_data, dict) else []
                if isinstance(serp_data, dict) and serp_data.get("error"):
                    fail_count += 1
                    debug_messages.append(f"Keyword '{keyword}': API error - {serp_data.get('error')}")
                    serp_results = []
                if serp_results is None:
                    serp_results = []
                    
                keyword_rankings = {domain: None for domain in all_domains}
                for res_index, res in enumerate(serp_results):
                    url = res if isinstance(res, str) else (res.get("link") or res.get("displayed_link") or "")
                    if not url:
                        continue
                    netloc = DataProcessing.extract_domain(url)
                    for domain in all_domains:
                        if keyword_rankings[domain] is None and netloc.endswith(domain):
                            keyword_rankings[domain] = res.get("position", res_index + 1)
                    if all(val is not None for val in keyword_rankings.values()):
                        break
                        
                if serp_results:
                    success_count += 1
                    debug_messages.append(f"Keyword '{keyword}': Retrieved {len(serp_results)} results. Rankings: {keyword_rankings}")
                else:
                    no_result_count += 1
                    debug_messages.append(f"Keyword '{keyword}': No organic results found.")
                    
                results.append({
                    "keyword": keyword,
                    "cluster": cluster,
                    "volume": volume,
                    "rankings": keyword_rankings,
                    "features": features,
                    "serp_results": serp_results[:10]  # Store top 10 results for analysis
                })
                
            progress_bar.progress(min((idx + 1) / len(df), 1.0))
    
    progress_bar.empty()
    logging.info(f"Querying complete. Success: {success_count}, No results: {no_result_count}, Failures: {fail_count}, Cache hits: {cache_hits}")
    
    # Save historical data if requested
    if save_history and (success_count > 0 or cache_hits > 0):
        HistoricalData.save_historical_data(results)
        st.success("Results saved to historical data")
    
    # Display query execution summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Keywords Processed", len(results))
    with col2:
        st.metric("Successful API Fetches", success_count)
    with col3:
        st.metric("Cache Hits", cache_hits)
    with col4:
        st.metric("API Failures", fail_count)
        
    if fail_count > 0:
        st.warning(f"{fail_count} API queries failed. Check your SERPAPI key or network.")
    if success_count == 0 and cache_hits == 0:
        st.error("No successful results. Unable to calculate visibility.")
        st.expander("Debug Details").write("\n".join(debug_messages))
        st.stop()
    with st.expander("Detailed Debug Information"):
        st.write("\n".join(debug_messages))
    
    # -------------------------------------------
    # Visibility Calculation
    # -------------------------------------------
    # Calculate visibility metrics
    overall_totals = {domain: {"weighted": 0.0, "simple": 0} for domain in all_domains}
    cluster_stats = {}
    
    for entry in results:
        cluster = entry["cluster"]
        vol = float(entry.get("volume", 0))
        
        if cluster not in cluster_stats:
            cluster_stats[cluster] = {
                "total_volume": 0.0,
                "keywords_count": 0,
                "weighted": {domain: 0.0 for domain in all_domains},
                "simple": {domain: 0 for domain in all_domains}
            }
            
        cluster_stats[cluster]["total_volume"] += vol
        cluster_stats[cluster]["keywords_count"] += 1
        
        for domain in all_domains:
            pos = entry["rankings"].get(domain)
            weighted = vol * SEOCalculator.CTR_MAP.get(int(pos), 0) if (pos is not None and int(pos) in SEOCalculator.CTR_MAP) else 0
            simple = SEOCalculator.simple_score(pos)
            
            cluster_stats[cluster]["weighted"][domain] += weighted
            cluster_stats[cluster]["simple"][domain] += simple
            overall_totals[domain]["weighted"] += weighted
            overall_totals[domain]["simple"] += simple
    
    # Filter results if requested
    if min_volume > 0 or ranking_filter != "All":
        filtered_results = DataProcessing.filter_results(
            results, min_volume, ranking_filter, sort_results_by, target_domains
        )
        st.write(f"Applied filters: {len(filtered_results)} keywords remaining")
    else:
        filtered_results = sorted(results, key=lambda x: float(x.get("volume", 0)), reverse=True)
    
    # -------------------------------------------
    # Dashboard Overview
    # -------------------------------------------
    st.subheader("SEO Visibility Dashboard")
    
    # Summary metrics
    total_volume_all = sum(float(e.get("volume", 0)) for e in results)
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        # Top ranked keywords
        top_ranked = [r for r in results if any(pos is not None and pos <= 3 for domain, pos in r["rankings"].items() 
                                              if domain in target_domains)]
        st.metric("Keywords in Top 3", len(top_ranked))
        
    with metrics_col2:
        # Keywords on page 1
        page1_ranked = [r for r in results if any(pos is not None and pos <= 10 for domain, pos in r["rankings"].items() 
                                               if domain in target_domains)]
        st.metric("Keywords on Page 1", len(page1_ranked))
        
    with metrics_col3:
        # Not ranking keywords
        not_ranking = [r for r in results if all(pos is None for domain, pos in r["rankings"].items() 
                                               if domain in target_domains)]
        st.metric("Keywords Not Ranking", len(not_ranking))
    
    # Domain visibility comparison
    st.subheader("Domain Visibility Summary")
    summary_rows = []
    for domain in all_domains:
        weighted_visibility = (overall_totals[domain]["weighted"] / (total_volume_all * SEOCalculator.CTR_MAP[1]) * 100) if total_volume_all > 0 else 0
        
        is_target = domain in target_domains
        domain_type = "🎯 Target" if is_target else "🔍 Competitor"
        
        summary_rows.append({
            "Domain": f"{domain_type}: {domain}",
            "Simple Visibility": overall_totals[domain]["simple"],
            "Weighted Visibility (%)": round(weighted_visibility, 2),
            "Top 3 Keywords": len([r for r in results if r["rankings"].get(domain) is not None and r["rankings"].get(domain) <= 3]),
            "Page 1 Keywords": len([r for r in results if r["rankings"].get(domain) is not None and r["rankings"].get(domain) <= 10]),
            "Not Ranking": len([r for r in results if r["rankings"].get(domain) is None])
        })
    
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df)
    
    # -------------------------------------------
    # Visualization Tabs
    # -------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(["Rankings Distribution", "Cluster Analysis", "Domain Comparison", "Historical Trends"])
    
    with tab1:
        st.subheader("Ranking Position Distribution")
        col1, col2 = st.columns([2, 1])
        with col1:
            # Interactive position chart
            position_chart = Visualization.create_interactive_position_chart(results, all_domains)
            st.plotly_chart(position_chart, use_container_width=True)
        with col2:
            # Position breakdown table
            position_breakdown = []
            for domain in all_domains:
                domain_positions = [r["rankings"].get(domain) for r in results if r["rankings"].get(domain) is not None]
                top3 = len([p for p in domain_positions if p <= 3])
                top10 = len([p for p in domain_positions if p <= 10])
                top20 = len([p for p in domain_positions if p <= 20])
                top30 = len([p for p in domain_positions if p <= 30])
                position_breakdown.append({
                    "Domain": domain,
                    "Top 3": top3,
                    "Top 10": top10,
                    "Top 20": top20,
                    "Top 30": top30,
                    "Not Ranking": len(results) - len(domain_positions)
                })
            position_df = pd.DataFrame(position_breakdown)
            st.dataframe(position_df)
    
    with tab2:
        st.subheader("Cluster Visibility Analysis")
        
        # Per-cluster summary table
        cluster_summary = []
        for cluster, stats in cluster_stats.items():
            row = {"Cluster": cluster, "Keywords": stats["keywords_count"], "Total Volume": stats["total_volume"]}
            for domain in all_domains:
                row[f"{domain} Simple"] = stats["simple"][domain]
                pct = (stats["weighted"][domain] / stats["total_volume"] * 100) if stats["total_volume"] > 0 else 0
                row[f"{domain} Weighted (%)"] = round(pct, 2)
            cluster_summary.append(row)
        
        cluster_df = pd.DataFrame(cluster_summary)
        st.dataframe(cluster_df)
        
        # Cluster visibility chart
        cluster_vis_chart = Visualization.create_cluster_visibility_chart(cluster_stats, all_domains)
        st.pyplot(cluster_vis_chart)
    
    with tab3:
        st.subheader("Domain Comparison")
        
        # Domain comparison metrics
        if target_domains and competitor_domains:
            # Content gap analysis
            content_gaps = SEOCalculator.identify_content_gaps(results, target_domains, competitor_domains)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Content Gap Keywords", len(content_gaps))
                
                if content_gaps:
                    gap_df = pd.DataFrame([{
                        "Keyword": gap["keyword"],
                        "Cluster": gap["cluster"],
                        "Volume": gap["volume"],
                        **{domain: gap["rankings"].get(domain) for domain in competitor_domains[:3]}
                    } for gap in content_gaps])
                    st.dataframe(gap_df)
            
            with col2:
                # Top competitor domains analysis
                top_domains = SEOCalculator.analyze_top_ranking_domains(results, target_domains)
                st.write("Top Ranking Domains (excluding your domains)")
                
                top_domains_df = pd.DataFrame(top_domains[:10], columns=["Domain", "Score"])
                st.dataframe(top_domains_df)
    
    with tab4:
        st.subheader("Visibility Trend Over Time")
        if historical_data:
            trend_chart = Visualization.create_historical_trend_chart(historical_data, all_domains)
            if trend_chart:
                st.plotly_chart(trend_chart, use_container_width=True)
            else:
                st.info("Not enough historical data to generate trend chart")
        else:
            st.info("No historical data available. Enable 'Save results as historical data' to start tracking.")
    
    # -------------------------------------------
    # Keyword Ranking Table
    # -------------------------------------------
    st.subheader("Keyword Ranking Table")
    ranking_data = []
    for entry in filtered_results:
        # Calculate opportunity score
        opportunity_score = SEOCalculator.calculate_opportunity_score(entry, target_domains)
        
        row = {
            "Keyword": entry["keyword"], 
            "Cluster": entry["cluster"], 
            "Volume": entry["volume"],
            "Opportunity": opportunity_score
        }
        for domain in all_domains:
            row[domain] = entry["rankings"].get(domain)
        ranking_data.append(row)
    
    ranking_df = pd.DataFrame(ranking_data)
    st.dataframe(ranking_df)
    
    # Export options
    csv_full = ranking_df.to_csv(index=False)
    st.download_button("Download keyword rankings (CSV)", data=csv_full, file_name="seo_keyword_rankings.csv", mime="text/csv")
    
    # -------------------------------------------
    # SEO Recommendations
    # -------------------------------------------
    if openai_api_key:
        st.subheader("AI-Driven Recommendations")
        
        recommendation_tabs = st.tabs(["Actionable Plan", "Keyword Analysis", "Cluster Strategy"])
        
        with recommendation_tabs[0]:
            st.subheader("Prioritized Action Plan")
            if st.button("Generate Action Plan"):
                with st.spinner("Generating prioritized action plan..."):
                    action_plan = SEORecommendations.generate_action_plan(results, target_domains)
                    
                    for cluster, plan in action_plan:
                        with st.expander(f"Cluster: {cluster} - Opportunity Score: {int(plan['total_opportunity'])}"):
                            st.write(f"Top keywords: {', '.join(plan['keywords'][:5])}")
                            st.write("### Recommended Actions")
                            for action in plan['actions']:
                                st.write(f"- **{action['action']}** (Opportunity: {action['opportunity']})")
        
        with recommendation_tabs[1]:
            st.subheader("Individual Keyword Analysis")
            selected_keyword = st.selectbox("Select a keyword for detailed analysis", 
                                           [entry["keyword"] for entry in results])
            
            if selected_keyword and st.button("Analyze Selected Keyword"):
                selected_entry = next((entry for entry in results if entry["keyword"] == selected_keyword), None)
                
                if selected_entry:
                    with st.spinner("Analyzing keyword..."):
                        # Get current positions
                        positions = {domain: pos for domain, pos in selected_entry["rankings"].items() 
                                   if pos is not None}
                        best_target_pos = min([pos for domain, pos in positions.items() 
                                             if domain in target_domains], default=None)
                        competitor_positions = {domain: pos for domain, pos in positions.items() 
                                             if domain in competitor_domains}
                        
                        # Generate recommendations
                        ai_recommendations = SEORecommendations.generate_actionable_recommendations(
                            selected_keyword, best_target_pos, competitor_positions
                        )
                        
                        # Display results
                        st.write(f"### Keyword: {selected_keyword}")
                        st.write(f"**Cluster:** {selected_entry['cluster']}")
                        st.write(f"**Search Volume:** {selected_entry['volume']}")
                        st.write(f"**Best Current Position:** {best_target_pos if best_target_pos else 'Not ranking'}")
                        
                        st.write("### SEO Recommendations")
                        st.write(ai_recommendations)
                        
                        # SERP feature analysis if available
                        if selected_entry.get("features"):
                            st.write("### SERP Feature Analysis")
                            serp_analysis = SEORecommendations.analyze_serp_features(
                                selected_keyword, selected_entry["features"]
                            )
                            st.write(serp_analysis)
        
        with recommendation_tabs[2]:
            st.subheader("Cluster-Level Content Strategy")
            selected_cluster = st.selectbox("Select a cluster for content strategy", 
                                          sorted(cluster_stats.keys()))
            
            if selected_cluster and st.button("Generate Cluster Strategy"):
                with st.spinner("Generating cluster content strategy..."):
                    # Get all keywords in this cluster
                    cluster_keywords = [entry["keyword"] for entry in results 
                                      if entry["cluster"] == selected_cluster]
                    
                    # Generate strategy
                    cluster_strategy = SEORecommendations.generate_cluster_strategy(
                        selected_cluster, cluster_keywords[:30]  # Limit to 30 keywords to avoid token limits
                    )
                    
                    # Display strategy
                    st.write(f"### Cluster: {selected_cluster}")
                    st.write(f"**Keywords:** {len(cluster_keywords)}")
                    st.write(f"**Total Volume:** {cluster_stats[selected_cluster]['total_volume']}")
                    
                    st.write("### Content Strategy")
                    st.write(cluster_strategy)
    
    # -------------------------------------------
    # App Footer
    # -------------------------------------------
    st.markdown("""
    ---
    ### Enhanced SEO Visibility Estimator v3.0
    An advanced tool for analyzing SEO visibility with AI-driven content strategy recommendations.
    """)
    logging.info("App execution completed successfully.")

if __name__ == "__main__":
    main()
