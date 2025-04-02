# -------------------------------------------
# SEO Visibility Estimator - Complete App
# -------------------------------------------
import streamlit as st
import pandas as pd
import logging
import os
import json
import hashlib
from datetime import datetime, timedelta
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import seaborn as sns
import openai  # For AI-based SEO recommendations
import requests  # Needed for fallback SERPAPI queries

# Configure logging for debugging
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)

# Import utility modules from the project
try:
    from utils import data_processing, seo_calculator, optimization, api_manager
except ImportError:
    # Create dummy modules if not found (for standalone operation)
    class DummyModule:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    api_manager = DummyModule()
    optimization = DummyModule()
    data_processing = DummyModule()
    seo_calculator = DummyModule()
    logging.warning("Using dummy utility modules. Please ensure the utils/ folder is correctly set up.")

# -------------------------------------------
# Fallback for SERPAPI query if api_manager.search_query is missing
# -------------------------------------------
if not hasattr(api_manager, 'search_query'):
    def search_query(keyword, api_key, params):
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
    api_manager.search_query = search_query

# -------------------------------------------
# API Cache System Implementation
# -------------------------------------------
class ApiCache:
    """
    Cache system for SERPAPI results to reuse previous queries and reduce API usage.
    """
    def __init__(self, cache_dir='.cache', expiry_days=7):
        self.cache_dir = cache_dir
        self.expiry_days = expiry_days
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_key(self, keyword, api_params=None):
        # Incorporate localization parameters into the cache key
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

# -------------------------------------------
# Keyword Intent Analysis System
# -------------------------------------------
def analyze_keyword_intent(keyword):
    """
    Classify keyword intent into Informational, Transactional, Commercial, or Navigational.
    """
    keyword = keyword.lower()
    info_patterns = ['what', 'how', 'when', 'where', 'why', 'who', 'which',
                     'vs', 'versus', 'difference', 'mean', 'means', 'meaning',
                     'examples', 'definition', 'guide', 'tutorial', 'learn',
                     'explain', 'explained', 'understanding', 'understand',
                     'help', 'ideas', 'tips', '?', 'instructions']
    trans_patterns = ['buy', 'price', 'cheap', 'deal', 'discount', 'sale',
                      'shipping', 'order', 'purchase', 'coupon', 'shop',
                      'booking', 'book', 'subscribe', 'subscription', 'download',
                      'free', 'trial', 'demo', 'get', 'affordable', 'cost',
                      'pay', 'payment', 'checkout', 'cart']
    commercial_patterns = ['best', 'top', 'review', 'reviews', 'compare', 'comparison',
                           'vs', 'features', 'model', 'models', 'alternative',
                           'alternatives', 'recommended', 'premium', 'pros and cons',
                           'advantages', 'disadvantages', 'worth it']
    nav_patterns = ['.com', '.org', '.net', '.io', '.co', 'login', 'sign in',
                    'account', 'register', 'signup', 'sign up', 'website',
                    'official', 'homepage', 'home page', 'customer service']
    
    for pattern in info_patterns:
        if pattern in keyword.split() or pattern in keyword:
            return "Informational"
    for pattern in trans_patterns:
        if pattern in keyword.split() or pattern in keyword:
            return "Transactional"
    for pattern in commercial_patterns:
        if pattern in keyword.split() or pattern in keyword:
            return "Commercial"
    for pattern in nav_patterns:
        if pattern in keyword:
            return "Navigational"
    return "Undetermined"

# -------------------------------------------
# Representative Keywords Selection Function
# -------------------------------------------
def select_representative_keywords(df, samples_per_cluster=3, min_volume_samples=1):
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

# -------------------------------------------
# Improved CTR Model for Visibility Calculation
# -------------------------------------------
def get_improved_ctr_map():
    base_ctr = {
        1: 0.3042,
        2: 0.1559,
        3: 0.0916,
        4: 0.0651,
        5: 0.0478,
        6: 0.0367,
        7: 0.0289,
        8: 0.0241,
        9: 0.0204,
        10: 0.0185,
        11: 0.0156,
        12: 0.0138,
        13: 0.0122,
        14: 0.0108,
        15: 0.0096
    }
    return base_ctr

def calculate_weighted_visibility(results, volume_weight=0.7, cluster_importance=None):
    if not results:
        return 0.0
    ctr_map = get_improved_ctr_map()
    total_volume = sum(float(entry.get("volume", 0)) for entry in results)
    total_potential_clicks = 0
    total_captured_clicks = 0
    for entry in results:
        volume = float(entry.get("volume", 0))
        rank = entry.get("domain_rank")
        cluster = entry.get("cluster", "")
        cluster_factor = 1.0
        if cluster_importance and cluster in cluster_importance:
            cluster_factor = cluster_importance[cluster]
        max_potential = volume * ctr_map[1] * cluster_factor
        total_potential_clicks += max_potential
        if rank is not None and rank in ctr_map:
            captured = volume * ctr_map[rank] * cluster_factor
            total_captured_clicks += captured
    if total_potential_clicks > 0:
        visibility_score = (total_captured_clicks / total_potential_clicks) * 100
    else:
        visibility_score = 0.0
    return round(visibility_score, 2)

def calculate_weighted_visibility_for_domain(results, domain):
    filtered = []
    for entry in results:
        rank = entry["rankings"].get(domain)
        if rank is not None:
            filtered.append({
                "keyword": entry["keyword"],
                "cluster": entry["cluster"],
                "volume": entry["volume"],
                "domain_rank": rank
            })
    return calculate_weighted_visibility(filtered)

# -------------------------------------------
# OpenAI API Integration for SEO Suggestions
# -------------------------------------------
def generate_seo_suggestions(keyword):
    prompt = (
        f"Keyword: \"{keyword}\"\n"
        "1. **Intent**: Identify the search intent (Informational, Navigational, Transactional, or Commercial) and briefly explain why.\n"
        "2. **Content Type**: Recommend the best type of content to target this keyword (e.g., blog article, product page, landing page).\n"
        "3. **Title Tag**: Propose an SEO-optimized title tag (<= 60 characters) for a page targeting this keyword.\n"
        "4. **Meta Description**: Draft a meta description (<= 155 characters) that incorporates the keyword and entices clicks.\n"
        "5. **CTA**: Provide a short call-to-action suitable for this page.\n"
        "6. **Content Outline**: If the keyword is informational or broad, outline key sections for a comprehensive article; if transactional, outline key selling points.\n"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or "gpt-3.5-turbo" for lower cost
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        logging.error(f"OpenAI API error for keyword '{keyword}': {e}")
        return "AI suggestion unavailable."

def generate_cluster_strategy(cluster_name, keywords):
    prompt = (
        f"Cluster: {cluster_name}\n"
        f"Keywords: {', '.join(keywords)}\n"
        "You are an SEO strategist. Provide a summary SEO content strategy for these keywords, including search intent analysis, recommended content types, meta title and description guidelines, and call-to-action suggestions."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        logging.error(f"OpenAI API error for cluster '{cluster_name}': {e}")
        return "AI cluster strategy unavailable."

# -------------------------------------------
# Utility Functions
# -------------------------------------------
def clean_domain(domain: str) -> str:
    domain = domain.strip()
    if not domain:
        return ""
    domain = domain.replace("http://", "").replace("https://", "")
    domain = domain.split("/")[0]
    if domain.startswith("www."):
        domain = domain[4:]
    return domain.lower()

# -------------------------------------------
# Streamlit App Title and Description
# -------------------------------------------
st.title("SEO Visibility Estimator")
st.markdown("""
Upload a CSV of keywords (with **keyword**, **cluster_name**, **Avg. monthly searches** columns) and configure options to estimate SEO visibility.
This tool calculates visibility scores for multiple target domains across keyword clusters by querying Google SERPs via SERPAPI.
""")

# -------------------------------------------
# CSV File Input with Improved Error Handling
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
    df.columns = [col.strip().lower() for col in df.columns]
    keyword_cols = [col for col in df.columns if 'keyword' in col.lower()]
    cluster_cols = [col for col in df.columns if 'cluster' in col.lower()]
    volume_cols = [col for col in df.columns if 'search' in col.lower() or 'volume' in col.lower() or 'monthly' in col.lower()]
    if keyword_cols:
        df = df.rename(columns={keyword_cols[0]: 'keyword'})
    if cluster_cols:
        df = df.rename(columns={cluster_cols[0]: 'cluster_name'})
    if volume_cols:
        df = df.rename(columns={volume_cols[0]: 'avg_monthly_searches'})
    required_cols = {'keyword', 'cluster_name'}
    if not any(col in df.columns for col in ['avg_monthly_searches', 'avg. monthly searches']):
        if volume_cols:
            df = df.rename(columns={volume_cols[0]: 'avg_monthly_searches'})
        else:
            st.warning("No volume/search data found. Using default values of 10 for all keywords.")
            df['avg_monthly_searches'] = 10
    else:
        if 'avg. monthly searches' in df.columns and 'avg_monthly_searches' not in df.columns:
            df = df.rename(columns={'avg. monthly searches': 'avg_monthly_searches'})
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        st.error(f"CSV is missing required columns: {missing}. Found columns: {list(df.columns)}.")
        st.write("First few rows of your CSV:")
        st.write(df.head())
        st.stop()
    df = df.dropna(subset=['keyword'])
    try:
        if 'avg_monthly_searches' in df.columns:
            df['avg_monthly_searches'] = df['avg_monthly_searches'].astype(str).str.replace('K', '000')
            df['avg_monthly_searches'] = df['avg_monthly_searches'].astype(str).str.replace(',', '')
            df['avg_monthly_searches'] = pd.to_numeric(df['avg_monthly_searches'], errors='coerce').fillna(10)
    except Exception as vol_err:
        st.warning(f"Error converting volume data: {vol_err}. Using fallback values.")
        df['avg_monthly_searches'] = 10
    if df.empty:
        st.error("No valid keyword data found in the CSV file after processing.")
        st.stop()
    num_keywords = len(df)
    st.write(f"**Total keywords loaded:** {num_keywords}")
    st.write(f"**CSV columns detected:** {', '.join(df.columns)}")
    with st.expander("Preview of loaded data"):
        st.write(df.head())
except Exception as e:
    st.error(f"Error processing CSV file: {e}")
    st.write("### Troubleshooting suggestions:")
    st.write("1. Check CSV format for required columns ('keyword', 'cluster_name', and search volume).")
    st.write("2. Ensure your CSV uses a standard delimiter (comma or semicolon).")
    st.write("3. Check for special characters or encoding issues.")
    st.stop()

# -------------------------------------------
# User Inputs: Multi-Domain, Localization, API Keys & Filters
# -------------------------------------------
# Multi-domain input (comma separated)
domains_input = st.text_input("Target Domains (comma separated)", "example.com, competitor1.com")
target_domains = [clean_domain(d) for d in domains_input.split(",") if clean_domain(d)]
if not target_domains:
    st.warning("No valid target domains provided. Please enter at least one domain.")
    st.stop()

# Localization options for SERPAPI
st.markdown("#### SERPAPI Localization Options")
country_code = st.text_input("Country Code (gl)", "us")
language_code = st.text_input("Language Code (hl)", "en")
location_input = st.text_input("City/Location (optional)", "")

# SERPAPI API key input
api_key = st.text_input("SERPAPI API Key", type="password", help="Your SERPAPI key is required to fetch Google results.")
if not api_key:
    st.warning("Please provide a SERPAPI API key to run the analysis.")
    st.stop()

# OpenAI API key input for AI suggestions
openai_api_key = st.text_input("OpenAI API Key (optional)", type="password", help="Provide your OpenAI key to get AI-driven SEO recommendations.")
if openai_api_key:
    openai.api_key = openai_api_key
else:
    st.info("OpenAI API key not provided. AI-based SEO recommendations will be skipped.")

# Cluster filtering (optional)
cluster_names = sorted(df["cluster_name"].unique())
selected_clusters = st.multiselect("Filter by Clusters (optional)", cluster_names)
if selected_clusters:
    df = df[df["cluster_name"].isin(selected_clusters)]
    st.write(f"Filtered to {len(selected_clusters)} selected clusters. Keywords remaining: {len(df)}")
if df.empty:
    st.warning("No keywords to process after applying cluster filter.")
    st.stop()

# Cost optimization options
use_cost_opt = st.checkbox("Use cost optimization (query only representative keywords per cluster)", value=False)
cost_opt_params = {}
if use_cost_opt:
    col1, col2 = st.columns(2)
    with col1:
        samples_per_cluster = st.slider("Keywords per cluster", min_value=1, max_value=10, value=3)
    with col2:
        min_volume_keywords = st.slider("Min low-volume keywords", min_value=1, max_value=5, value=1)
    cost_opt_params = {
        'samples_per_cluster': samples_per_cluster,
        'min_volume_samples': min_volume_keywords
    }
    try:
        original_count = len(df)
        df = select_representative_keywords(df, **cost_opt_params)
        optimized_count = len(df)
        st.write(f"Cost Optimization enabled: reduced {original_count} keywords to {optimized_count} representative keywords for querying.")
        cluster_counts = df.groupby('cluster_name').size().reset_index(name='keyword_count')
        st.write("Keywords selected per cluster:")
        st.write(cluster_counts)
    except Exception as opt_e:
        st.error(f"Error during cost optimization: {opt_e}. Proceeding with all keywords.")
        use_cost_opt = False

# Cache control
cache_control = st.checkbox("Use cache for API requests (reduces API usage)", value=True)
if cache_control:
    col1, col2 = st.columns(2)
    with col1:
        cache_days = st.slider("Cache expiry (days)", min_value=1, max_value=30, value=7)
    with col2:
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

# -------------------------------------------
# Button to Run Analysis
# -------------------------------------------
run_analysis = st.button("Run Analysis")
if not run_analysis:
    st.info("Click the 'Run Analysis' button to start processing the data.")
    st.stop()

# -------------------------------------------
# SERPAPI Queries Execution
# -------------------------------------------
st.write("Starting SERPAPI data retrieval... This may take a while for large keyword sets.")
progress_bar = st.progress(0)
results = []  # list to collect results for each keyword
debug_messages = []
success_count = 0
no_result_count = 0
fail_count = 0
cache_hits = 0

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
    # Build API parameters including localization options
    api_params = {
        'q': keyword,
        'engine': 'google',
        'api_key': api_key,
        'gl': country_code,
        'hl': language_code
    }
    if location_input:
        api_params['location'] = location_input

    if api_cache:
        cached_data = api_cache.get(keyword, api_params)
        if cached_data:
            serp_data = cached_data
            using_cache = True
            cache_hits += 1
            debug_messages.append(f"Keyword '{keyword}': Using cached data.")
            logging.info(f"Cache hit for keyword '{keyword}'.")
    if not using_cache:
        try:
            # Call the API via api_manager.search_query (fallback defined earlier if missing)
            serp_data = api_manager.search_query(keyword, api_key=api_key, params=api_params)
            if api_cache and serp_data:
                api_cache.set(keyword, serp_data, api_params)
        except Exception as e:
            fail_count += 1
            debug_messages.append(f"Keyword '{keyword}': API request failed - {e}")
            logging.error(f"API request error for '{keyword}': {e}")
            serp_data = None

    if serp_data is None:
        no_result_count += 1
        debug_messages.append(f"Keyword '{keyword}': No results returned from API.")
        logging.warning(f"No results for keyword '{keyword}'.")
        # For multi-domain, record empty rankings for all domains
        results.append({
            "keyword": keyword,
            "cluster": cluster,
            "volume": volume,
            "rankings": {domain: None for domain in target_domains}
        })
    else:
        # Parse the SERPAPI response
        serp_results = serp_data.get("organic_results", serp_data.get("results", [])) if isinstance(serp_data, dict) else []
        if isinstance(serp_data, dict) and serp_data.get("error"):
            fail_count += 1
            debug_messages.append(f"Keyword '{keyword}': API error - {serp_data.get('error')}")
            logging.error(f"API error for '{keyword}': {serp_data.get('error')}")
            serp_results = []
        if serp_results is None:
            serp_results = []

        # For each keyword, extract ranking positions for all target domains
        keyword_rankings = {domain: None for domain in target_domains}
        for res_index, res in enumerate(serp_results):
            url = ""
            if isinstance(res, str):
                url = res
            elif isinstance(res, dict):
                url = res.get("link") or res.get("displayed_link") or ""
            else:
                continue
            if not url:
                continue
            netloc = urlparse(url).netloc.lower()
            if netloc.startswith("www."):
                netloc = netloc[4:]
            for domain in target_domains:
                if keyword_rankings[domain] is None:
                    if netloc.endswith(domain):
                        keyword_rankings[domain] = res.get("position", res_index + 1)
            if all(rank is not None for rank in keyword_rankings.values()):
                break

        if serp_results:
            success_count += 1
            status_msg = f"Keyword '{keyword}': {len(serp_results)} results retrieved. Rankings: {keyword_rankings}"
            debug_messages.append(status_msg)
            logging.info(f"Results for '{keyword}': {keyword_rankings}")
        else:
            no_result_count += 1
            debug_messages.append(f"Keyword '{keyword}': No organic results returned.")
            logging.warning(f"No organic results for keyword '{keyword}'.")
        results.append({
            "keyword": keyword,
            "cluster": cluster,
            "volume": volume,
            "rankings": keyword_rankings
        })
    progress_bar.progress(min((idx + 1) / len(df), 1.0))

progress_bar.empty()
logging.info(f"SERPAPI querying complete. Success: {success_count}, No results: {no_result_count}, Failures: {fail_count}, Cache hits: {cache_hits}")

st.subheader("Query Execution Summary")
st.write(f"**Keywords processed:** {len(results)}")
st.write(f"**Successful API fetches:** {success_count}")
st.write(f"**Cache hits:** {cache_hits}")
st.write(f"**Keywords with no results:** {no_result_count}")
st.write(f"**API call failures:** {fail_count}")

if fail_count > 0:
    st.error(f"{fail_count} API queries failed. Please check your SERPAPI key or network connection.")
if success_count == 0 and cache_hits == 0:
    st.error("No successful results obtained from the API. Unable to calculate visibility.")
    st.expander("Debug Details").write("\n".join(debug_messages))
    st.stop()

with st.expander("Detailed Debug Information per Keyword"):
    for msg in debug_messages:
        st.write(msg)

# -------------------------------------------
# Visibility Calculation & Aggregation (Multi-Domain)
# -------------------------------------------
CTR_MAP = get_improved_ctr_map()
total_volume_all = 0.0
domain_totals = {domain: {"captured_volume": 0.0, "keywords_ranked": 0} for domain in target_domains}
cluster_stats = {}
for entry in results:
    cluster = entry["cluster"]
    vol = float(entry.get("volume", 0))
    total_volume_all += vol
    if cluster not in cluster_stats:
        cluster_stats[cluster] = {
            "total_volume": 0.0,
            "keywords_count": 0,
            "captured_volume": {domain: 0.0 for domain in target_domains},
            "keywords_ranked": {domain: 0 for domain in target_domains}
        }
    cluster_stats[cluster]["total_volume"] += vol
    cluster_stats[cluster]["keywords_count"] += 1
    for domain in target_domains:
        rank = entry["rankings"].get(domain)
        if rank is not None:
            captured = vol * CTR_MAP.get(int(rank), 0)
            cluster_stats[cluster]["captured_volume"][domain] += captured
            cluster_stats[cluster]["keywords_ranked"][domain] += 1
            domain_totals[domain]["captured_volume"] += captured
            domain_totals[domain]["keywords_ranked"] += 1

overall_visibility = {}
weighted_visibility = {}
for domain in target_domains:
    overall_visibility[domain] = (domain_totals[domain]["captured_volume"] / total_volume_all * 100) if total_volume_all > 0 else 0.0
    weighted_visibility[domain] = calculate_weighted_visibility_for_domain(results, domain)

# -------------------------------------------
# Results Output (Visibility Scores and Analysis)
# -------------------------------------------
st.subheader("SEO Visibility Results")
st.write("### Overall Visibility Scores")
for domain in target_domains:
    st.metric(f"Visibility Score ({domain})", value=f"{weighted_visibility[domain]:.1f}%",
              delta=f"{(weighted_visibility[domain] - overall_visibility[domain]):.1f}%")
    st.write(f"Basic Visibility for {domain}: {overall_visibility[domain]:.1f}%")

output_rows = []
for cluster_name, stats in sorted(cluster_stats.items()):
    row = {
        "Cluster": cluster_name,
        "Keywords": stats["keywords_count"],
        "Total Search Volume": int(stats["total_volume"])
    }
    for domain in target_domains:
        vis_pct = (stats["captured_volume"][domain] / stats["total_volume"] * 100) if stats["total_volume"] > 0 else 0.0
        row[f"{domain} Vis%"] = round(vis_pct, 1)
        row[f"{domain} Keywords Ranked"] = stats["keywords_ranked"][domain]
    output_rows.append(row)

if output_rows:
    cluster_df = pd.DataFrame(output_rows)
    base_cols = ["Cluster", "Keywords", "Total Search Volume"]
    domain_cols = []
    for domain in target_domains:
        domain_cols += [f"{domain} Vis%", f"{domain} Keywords Ranked"]
    cluster_df = cluster_df[base_cols + domain_cols]
    st.dataframe(cluster_df)
else:
    st.write("No cluster data to display.")

# -------------------------------------------
# Visualizations
# -------------------------------------------
def generate_visualizations(results, target_domains):
    results_df = pd.DataFrame(results)
    st.subheader("Ranking Position Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    for domain in target_domains:
        domain_ranks = results_df[results_df['rankings'].apply(lambda r: r.get(domain) is not None)]
        if not domain_ranks.empty:
            ranks = domain_ranks['rankings'].apply(lambda r: r.get(domain))
            sns.histplot(ranks, bins=list(range(1, 16)), alpha=0.5, label=domain, ax=ax)
    ax.set_title('SERP Position Distribution')
    ax.set_xlabel('Position')
    ax.set_ylabel('Number of keywords')
    ax.legend()
    st.pyplot(fig)

    st.subheader("Visibility by Cluster")
    cluster_vis_data = []
    for cluster, stats in cluster_stats.items():
        cluster_data = {'Cluster': cluster}
        for domain in target_domains:
            vis_pct = (stats["captured_volume"][domain] / stats["total_volume"] * 100) if stats["total_volume"] > 0 else 0.0
            cluster_data[domain] = round(vis_pct, 1)
        cluster_vis_data.append(cluster_data)
    cluster_vis_df = pd.DataFrame(cluster_vis_data)
    if len(target_domains) > 1:
        cluster_vis_df['Total_Vis'] = cluster_vis_df[target_domains[0]]
        for domain in target_domains[1:]:
            cluster_vis_df['Total_Vis'] += cluster_vis_df[domain]
        cluster_vis_df = cluster_vis_df.sort_values('Total_Vis', ascending=False)
    else:
        cluster_vis_df = cluster_vis_df.sort_values(target_domains[0], ascending=False)
    fig, ax = plt.subplots(figsize=(12, 8))
    x = range(len(cluster_vis_df['Cluster']))
    width = 0.8 / len(target_domains)
    for i, domain in enumerate(target_domains):
        positions = [j + i * width for j in x]
        ax.bar(positions, cluster_vis_df[domain], width, label=domain)
    ax.set_ylabel('Visibility (%)')
    ax.set_title('SEO Visibility by Cluster')
    ax.set_xticks([j + (width*(len(target_domains)-1)/2) for j in x])
    ax.set_xticklabels(cluster_vis_df['Cluster'], rotation=45, ha='right')
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

if success_count > 0 or cache_hits > 0:
    try:
        generate_visualizations(results, target_domains)
    except Exception as vis_error:
        st.error(f"Error generating visualizations: {vis_error}")
        logging.error(f"Visualization error: {vis_error}")

# -------------------------------------------
# Keyword Intent Analysis and Visualization
# -------------------------------------------
def analyze_keyword_intents(results):
    intent_counts = {"Informational": 0, "Transactional": 0, "Commercial": 0, "Navigational": 0, "Undetermined": 0}
    cluster_intents = {}
    for entry in results:
        keyword = entry.get("keyword", "")
        cluster = entry.get("cluster", "")
        intent = analyze_keyword_intent(keyword)
        intent_counts[intent] += 1
        if cluster not in cluster_intents:
            cluster_intents[cluster] = {"Informational": 0, "Transactional": 0, "Commercial": 0, "Navigational": 0, "Undetermined": 0}
        cluster_intents[cluster][intent] += 1
    return intent_counts, cluster_intents

def visualize_intents(intent_counts, cluster_intents):
    st.subheader("Search Intent Analysis")
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = list(intent_counts.keys())
    sizes = list(intent_counts.values())
    non_zero_labels = [label for label, size in zip(labels, sizes) if size > 0]
    non_zero_sizes = [size for size in sizes if size > 0]
    if non_zero_sizes:
        ax.pie(non_zero_sizes, labels=non_zero_labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        plt.title('Search Intent Distribution')
        st.pyplot(fig)
        st.write("### Intent by Cluster")
        intent_data = []
        for cluster, intents in cluster_intents.items():
            row = {"Cluster": cluster}
            row.update(intents)
            intent_data.append(row)
        intent_df = pd.DataFrame(intent_data)
        for intent_type in ["Informational", "Transactional", "Commercial", "Navigational"]:
            if intent_type in intent_df.columns:
                total = intent_df[list(intent_counts.keys())].sum(axis=1)
                intent_df[f"%{intent_type}"] = (intent_df[intent_type] / total * 100).round(1)
        intent_df = intent_df.sort_values("Cluster")
        st.dataframe(intent_df)
        st.write("### Intent-Based Recommendations")
        predominant_intent = max(intent_counts.items(), key=lambda x: x[1])[0]
        if predominant_intent == "Informational":
            st.write("""
            **Recommendations for Informational Keywords:**
            - Create detailed guides, tutorials, and FAQs.
            - Structure content to answer specific questions.
            - Optimize for featured snippets.
            - Use explanatory images, infographics, and videos.
            """)
        elif predominant_intent == "Transactional":
            st.write("""
            **Recommendations for Transactional Keywords:**
            - Optimize product/category pages.
            - Include clear and visible CTAs.
            - Display pricing, shipping, and availability information.
            - Use testimonials and reviews.
            """)
        elif predominant_intent == "Commercial":
            st.write("""
            **Recommendations for Commercial Keywords:**
            - Develop comparative content (tables, pros/cons).
            - Create detailed product/service reviews.
            - Use testimonials and case studies.
            - Consider “best of” or “alternatives” content.
            """)
        elif predominant_intent == "Navigational":
            st.write("""
            **Recommendations for Navigational Keywords:**
            - Optimize brand and landing pages.
            - Use schema markup to improve SERP presence.
            - Ensure strong presence on Google My Business.
            - Enhance overall site navigation.
            """)

if success_count > 0 or cache_hits > 0:
    try:
        intent_counts, cluster_intents = analyze_keyword_intents(results)
        visualize_intents(intent_counts, cluster_intents)
    except Exception as intent_error:
        st.error(f"Error in intent analysis: {intent_error}")
        logging.error(f"Intent analysis error: {intent_error}")

# -------------------------------------------
# Cluster Correlation Analysis
# -------------------------------------------
def analyze_cluster_correlations(results):
    clusters = sorted(set(entry.get("cluster", "") for entry in results))
    cluster_rankings = {cluster: [] for cluster in clusters}
    for entry in results:
        cluster = entry.get("cluster", "")
        # Use ranking from the first target domain (or adjust as needed)
        if target_domains:
            rank = entry["rankings"].get(target_domains[0])
            if cluster and rank is not None:
                cluster_rankings[cluster].append(rank)
    cluster_avg_rank = {}
    for cluster, ranks in cluster_rankings.items():
        if ranks:
            cluster_avg_rank[cluster] = sum(ranks) / len(ranks)
    correlations = []
    processed = set()
    for cluster1 in clusters:
        if cluster1 not in cluster_avg_rank:
            continue
        for cluster2 in clusters:
            if cluster2 not in cluster_avg_rank or cluster1 == cluster2:
                continue
            pair = tuple(sorted([cluster1, cluster2]))
            if pair in processed:
                continue
            processed.add(pair)
            rank_diff = abs(cluster_avg_rank[cluster1] - cluster_avg_rank[cluster2])
            if rank_diff < 3.0:
                correlations.append({
                    "cluster1": cluster1,
                    "cluster2": cluster2,
                    "rank_diff": round(rank_diff, 2),
                    "correlation": round(max(0, 1 - (rank_diff / 10)), 2)
                })
    correlations.sort(key=lambda x: x["correlation"], reverse=True)
    return correlations, cluster_avg_rank

def visualize_cluster_correlations(correlations, cluster_avg_rank):
    st.subheader("Cluster Correlation Analysis")
    if not correlations:
        st.write("No significant correlations found between clusters.")
        return
    st.write("### Average Position by Cluster")
    avg_rank_data = [{"Cluster": cluster, "Average Position": round(rank, 2)} for cluster, rank in cluster_avg_rank.items()]
    avg_rank_df = pd.DataFrame(avg_rank_data).sort_values("Average Position")
    st.dataframe(avg_rank_df)
    st.write("### Clusters with similar behavior (possible correlation)")
    corr_df = pd.DataFrame(correlations)
    if not corr_df.empty:
        corr_df = corr_df.rename(columns={
            "cluster1": "Cluster A", 
            "cluster2": "Cluster B",
            "rank_diff": "Ranking difference",
            "correlation": "Correlation index"
        })
        st.dataframe(corr_df)
        st.write("### Recommendations based on correlations")
        st.write("""
        Clusters with high correlation may indicate:
        - Internal linking opportunities.
        - Complementary content topics.
        - Possibilities for unified or consolidated content strategies.
        """)

if (success_count > 0 or cache_hits > 0) and len(set(entry.get("cluster", "") for entry in results)) > 1:
    try:
        correlations, cluster_avg_rank = analyze_cluster_correlations(results)
        visualize_cluster_correlations(correlations, cluster_avg_rank)
    except Exception as corr_error:
        st.error(f"Error in correlation analysis: {corr_error}")
        logging.error(f"Correlation analysis error: {corr_error}")

# -------------------------------------------
# AI-Driven SEO Content Strategy (Using OpenAI)
# -------------------------------------------
if openai_api_key:
    st.subheader("AI-Driven SEO Content Strategy")
    ai_section = st.expander("View AI-generated recommendations per cluster", expanded=False)
    with ai_section:
        ai_cluster_strategies = {}
        for cluster, stats in cluster_stats.items():
            cluster_keywords = [entry["keyword"] for entry in results if entry["cluster"] == cluster]
            if cluster_keywords:
                strategy = generate_cluster_strategy(cluster, cluster_keywords)
                ai_cluster_strategies[cluster] = strategy
                st.markdown(f"**Cluster: {cluster}**")
                st.write(strategy)
else:
    st.info("OpenAI API key not provided. Skipping AI-driven SEO strategy recommendations.")

# -------------------------------------------
# Optimization Suggestions (Optional)
# -------------------------------------------
def generate_optimization_suggestions(results, target_domains):
    suggestions = []
    results_df = pd.DataFrame(results)
    for domain in target_domains:
        domain_only = results_df[results_df['rankings'].apply(lambda r: r.get(domain) is not None)]
        if not domain_only.empty:
            below_top = domain_only[domain_only['rankings'].apply(lambda r: r.get(domain, 999) > 1)]
            if not below_top.empty:
                suggestions.append({
                    "type": "rank_improvement",
                    "title": f"Improve positions for {domain}",
                    "description": f"{len(below_top)} keywords where {domain} is not in position 1.",
                    "keywords": below_top.sort_values(lambda df: df['rankings'].apply(lambda r: r.get(domain))).head(5)['keyword'].tolist(),
                    "action": "Analyze and optimize these pages to boost rankings."
                })
    return suggestions

def display_optimization_suggestions(suggestions):
    st.subheader("SEO Optimization Suggestions")
    if not suggestions:
        st.write("Not enough comparative data to generate specific suggestions.")
        return
    for suggestion in suggestions:
        with st.expander(f"💡 {suggestion['title']}", expanded=True):
            st.write(f"**Description:** {suggestion['description']}")
            if 'keywords' in suggestion:
                st.write("**Key keywords:**")
                for kw in suggestion['keywords']:
                    st.write(f"- {kw}")
            st.write(f"**Recommended action:** {suggestion['action']}")
            if suggestion['type'] == 'rank_improvement':
                st.write("""
                **Additional tips:**
                - Update and expand existing content.
                - Improve page speed and mobile-friendliness.
                - Enhance internal linking.
                - Optimize title tags and meta descriptions.
                """)
                
if (success_count > 0 or cache_hits > 0) and len(target_domains) > 1:
    try:
        suggestions = generate_optimization_suggestions(results, target_domains)
        display_optimization_suggestions(suggestions)
    except Exception as sugg_error:
        st.error(f"Error generating optimization suggestions: {sugg_error}")
        logging.error(f"Suggestion generation error: {sugg_error}")

# -------------------------------------------
# Export Results Section
# -------------------------------------------
full_results_df = pd.DataFrame(results)
if not full_results_df.empty:
    st.subheader("Export Results")
    csv = full_results_df.to_csv(index=False)
    st.download_button(
        label="Download complete results (CSV)",
        data=csv,
        file_name="seo_visibility_results.csv",
        mime="text/csv"
    )
    if 'cluster_df' in locals() and not cluster_df.empty:
        csv_clusters = cluster_df.to_csv(index=False)
        st.download_button(
            label="Download cluster analysis (CSV)",
            data=csv_clusters,
            file_name="seo_cluster_analysis.csv",
            mime="text/csv"
        )

# -------------------------------------------
# App Footer
# -------------------------------------------
st.markdown("""
---
### SEO Visibility Estimator v2.0
An advanced tool for analyzing SEO visibility across multiple domains with AI-driven strategy recommendations.

**Features:**
- Multi-domain visibility analysis.
- SERPAPI localization (country, language, location).
- API caching system.
- Improved visibility calculation with a weighted CTR model.
- Search intent analysis.
- Cluster correlation detection.
- AI-driven SEO content and optimization suggestions.
""")
logging.info("App execution completed successfully.")
