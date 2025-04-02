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
    
    if 'api_manager' not in globals():
        api_manager = DummyModule()
        logging.warning("Using dummy api_manager. Please create utils/api_manager.py")
    
    if 'optimization' not in globals():
        optimization = DummyModule()
        logging.warning("Using dummy optimization. Please create utils/optimization.py")

# -------------------------------------------
# API Cache System Implementation
# -------------------------------------------
class ApiCache:
    """
    Cache system for SerpAPI results that allows reusing
    previous queries and reducing API credit consumption.
    """
    def __init__(self, cache_dir='.cache', expiry_days=7):
        self.cache_dir = cache_dir
        self.expiry_days = expiry_days
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_key(self, keyword, api_params=None):
        """Generate a unique key for each query."""
        # Combine keyword and parameters to generate unique key
        cache_str = keyword.lower()
        if api_params:
            cache_str += json.dumps(api_params, sort_keys=True)
        # Generate MD5 hash as cache key
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def get_cache_path(self, cache_key):
        """Get the cache file path for a key."""
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def get(self, keyword, api_params=None):
        """Retrieve cached results if they exist and are valid."""
        cache_key = self.get_cache_key(keyword, api_params)
        cache_path = self.get_cache_path(cache_key)
        
        # Check if cache file exists
        if not os.path.exists(cache_path):
            return None
        
        try:
            # Read cache data
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            
            # Check expiration date
            cache_date = datetime.fromisoformat(cache_data.get('timestamp', '2000-01-01'))
            if datetime.now() - cache_date > timedelta(days=self.expiry_days):
                # Cache expired
                return None
            
            # Return cached data
            return cache_data.get('data')
        except Exception as e:
            logging.warning(f"Error reading cache for '{keyword}': {e}")
            return None
    
    def set(self, keyword, data, api_params=None):
        """Save results to cache."""
        if not data:
            return False
        
        cache_key = self.get_cache_key(keyword, api_params)
        cache_path = self.get_cache_path(cache_key)
        
        try:
            # Prepare data with timestamp
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'keyword': keyword,
                'data': data
            }
            
            # Save to file
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
    Classify the intent of a keyword based on common patterns.
    No external API required.
    
    Categories:
    - Informational: information seeking
    - Transactional: purchase intent
    - Navigational: brand/specific site search
    - Commercial: pre-purchase research
    """
    keyword = keyword.lower()
    
    # Informational intent patterns
    info_patterns = [
        'what', 'how', 'when', 'where', 'why', 'who', 'which',
        'vs', 'versus', 'difference', 'mean', 'means', 'meaning',
        'examples', 'definition', 'guide', 'tutorial', 'learn',
        'explain', 'explained', 'understanding', 'understand',
        'help', 'ideas', 'tips', '?', 'instructions'
    ]
    
    # Transactional intent patterns
    trans_patterns = [
        'buy', 'price', 'cheap', 'deal', 'discount', 'sale',
        'shipping', 'order', 'purchase', 'coupon', 'shop',
        'booking', 'book', 'subscribe', 'subscription', 'download',
        'free', 'trial', 'demo', 'get', 'affordable', 'cost',
        'pay', 'payment', 'checkout', 'cart'
    ]
    
    # Commercial intent patterns (research)
    commercial_patterns = [
        'best', 'top', 'review', 'reviews', 'compare', 'comparison',
        'vs', 'features', 'model', 'models', 'alternative',
        'alternatives', 'recommended', 'premium', 'pros and cons',
        'advantages', 'disadvantages', 'worth it'
    ]
    
    # Check navigational intent (usually brand names/websites)
    nav_patterns = [
        '.com', '.org', '.net', '.io', '.co', 'login', 'sign in',
        'account', 'register', 'signup', 'sign up', 'website',
        'official', 'homepage', 'home page', 'customer service'
    ]
    
    # Determine the main intent
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
    
    # Default if no clear patterns
    return "Undetermined"

# -------------------------------------------
# Representative Keywords Selection Function
# -------------------------------------------
def select_representative_keywords(df, samples_per_cluster=3, min_volume_samples=1):
    """
    Select representative keywords per cluster to reduce API calls.
    
    Strategy: 
    1. For each cluster, select the top N keywords by volume
    2. Also select some medium and low volume keywords
    3. Ensure adequate distribution across the volume range
    """
    selected_keywords = []
    
    for cluster in df['cluster_name'].unique():
        cluster_df = df[df['cluster_name'] == cluster].copy()
        
        # If the cluster has few keywords, take them all
        if len(cluster_df) <= samples_per_cluster:
            selected_keywords.append(cluster_df)
            continue
        
        # Sort by volume
        cluster_df = cluster_df.sort_values('avg_monthly_searches', ascending=False)
        
        # Select the highest volume keywords
        top_keywords = cluster_df.head(samples_per_cluster // 2)
        selected_keywords.append(top_keywords)
        
        # Select some medium volume keywords
        if len(cluster_df) > samples_per_cluster:
            mid_start = len(cluster_df) // 2 - samples_per_cluster // 4
            mid_keywords = cluster_df.iloc[mid_start:mid_start + samples_per_cluster // 4]
            selected_keywords.append(mid_keywords)
        
        # Select some low volume keywords
        if len(cluster_df) > samples_per_cluster * 2:
            low_keywords = cluster_df.tail(min_volume_samples)
            selected_keywords.append(low_keywords)
    
    # Combine all selected DataFrames
    result = pd.concat(selected_keywords)
    
    # Ensure no duplicates
    result = result.drop_duplicates(subset=['keyword'])
    
    return result

# -------------------------------------------
# Improved CTR Model for Visibility Calculation
# -------------------------------------------
def get_improved_ctr_map():
    """
    Provide an improved CTR model based on recent CTR by position studies.
    Includes search intent variations and integrates a more granular model.
    """
    # Basic CTR by position (more granular and accurate)
    base_ctr = {
        1: 0.3042,   # 30.42% CTR for position 1
        2: 0.1559,   # 15.59% for position 2
        3: 0.0916,   # 9.16% for position 3
        4: 0.0651,   # 6.51% for position 4
        5: 0.0478,   # 4.78% for position 5
        6: 0.0367,   # 3.67% for position 6
        7: 0.0289,   # 2.89% for position 7
        8: 0.0241,   # 2.41% for position 8
        9: 0.0204,   # 2.04% for position 9
        10: 0.0185,  # 1.85% for position 10
        # Additional positions (SERP may show more results)
        11: 0.0156,
        12: 0.0138,
        13: 0.0122,
        14: 0.0108,
        15: 0.0096
    }
    
    return base_ctr

# Function to calculate weighted visibility with Sistrix-like model
def calculate_weighted_visibility(results, volume_weight=0.7, cluster_importance=None):
    """
    Calculate SEO visibility using a weighted approach similar to Sistrix.
    
    Parameters:
    - results: list of results per keyword
    - volume_weight: weight of volume in calculation (vs. equal distribution)
    - cluster_importance: optional dictionary with specific weights per cluster
    
    This function uses an approach similar to Sistrix/Semrush visibility index
    that weights positions and considers volume distribution.
    """
    if not results:
        return 0.0
    
    # Get improved CTR model
    ctr_map = get_improved_ctr_map()
    
    # Total possible clickthroughs across the keyword set
    total_volume = sum(float(entry.get("volume", 0)) for entry in results)
    total_potential_clicks = 0
    total_captured_clicks = 0
    
    # For each keyword
    for entry in results:
        keyword = entry.get("keyword", "")
        cluster = entry.get("cluster", "")
        volume = float(entry.get("volume", 0))
        rank = entry.get("domain_rank")
        
        # Cluster importance factor (default 1.0)
        cluster_factor = 1.0
        if cluster_importance and cluster in cluster_importance:
            cluster_factor = cluster_importance[cluster]
        
        # Maximum potential clicks (if ranked position 1)
        max_potential = volume * ctr_map[1] * cluster_factor
        total_potential_clicks += max_potential
        
        # Captured clicks at current position
        if rank is not None and rank in ctr_map:
            captured = volume * ctr_map[rank] * cluster_factor
            total_captured_clicks += captured
    
    # Visibility as percentage of potential clicks captured
    if total_potential_clicks > 0:
        visibility_score = (total_captured_clicks / total_potential_clicks) * 100
    else:
        visibility_score = 0.0
    
    return round(visibility_score, 2)

# -------------------------------------------
# Streamlit App Title and Description
# -------------------------------------------
st.title("SEO Visibility Estimator")
st.markdown("""
Upload a CSV of keywords (with **keyword**, **cluster_name**, **Avg. monthly searches** columns) and configure options to estimate SEO visibility.
This tool will calculate a visibility score for a given domain (and an optional competitor) across keyword clusters by querying Google SERPs via SerpAPI.
""")

# -------------------------------------------
# CSV File Input with Improved Error Handling
# -------------------------------------------
uploaded_file = st.file_uploader("Upload Keyword CSV", type=["csv"])
if not uploaded_file:
    st.info("Please upload a CSV file to continue.")
    st.stop()

# Try reading the CSV with multiple configurations
try:
    # First option: standard reading
    df = pd.read_csv(uploaded_file)
    
    # If empty or has issues, try with different separators
    if df.empty or len(df.columns) <= 1:
        uploaded_file.seek(0)  # Reset file pointer
        df = pd.read_csv(uploaded_file, sep=';')  # Try with semicolon separator (common in Europe)
    
    # Normalize column names (remove spaces, convert to lowercase)
    df.columns = [col.strip().lower() for col in df.columns]
    
    # Look for flexible columns (can be in different formats)
    keyword_cols = [col for col in df.columns if 'keyword' in col.lower()]
    cluster_cols = [col for col in df.columns if 'cluster' in col.lower()]
    volume_cols = [col for col in df.columns if 'search' in col.lower() or 'volume' in col.lower() or 'monthly' in col.lower()]
    
    # Map found columns to required ones
    if keyword_cols:
        df = df.rename(columns={keyword_cols[0]: 'keyword'})
    if cluster_cols:
        df = df.rename(columns={cluster_cols[0]: 'cluster_name'})
    if volume_cols:
        df = df.rename(columns={volume_cols[0]: 'avg_monthly_searches'})
    
    # Verify required columns
    required_cols = {'keyword', 'cluster_name'}
    
    # Check if volume column exists or create it with default value
    if not any(col in df.columns for col in ['avg_monthly_searches', 'avg. monthly searches']):
        if volume_cols:
            # Try to use an existing volume-related column
            df = df.rename(columns={volume_cols[0]: 'avg_monthly_searches'})
        else:
            # Create a default column
            st.warning("No volume/search data found. Using default values of 10 for all keywords.")
            df['avg_monthly_searches'] = 10
    else:
        # Ensure column name is consistent
        if 'avg. monthly searches' in df.columns and 'avg_monthly_searches' not in df.columns:
            df = df.rename(columns={'avg. monthly searches': 'avg_monthly_searches'})
    
    # Check if we have the minimum required columns
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        st.error(f"CSV is missing required columns: {missing}. Found columns: {list(df.columns)}.")
        logging.error(f"CSV missing columns. Found: {list(df.columns)}; Expected at least: {required_cols}")
        
        # Show first few rows to help diagnose
        st.write("First few rows of your CSV:")
        st.write(df.head())
        st.stop()
    
    # Clean and prepare data
    # Remove rows where 'keyword' is empty
    df = df.dropna(subset=['keyword'])
    
    # Convert volume to numeric, handling different formats
    try:
        if 'avg_monthly_searches' in df.columns:
            # Clean non-numeric values (like '1K' or '1,000')
            df['avg_monthly_searches'] = df['avg_monthly_searches'].astype(str).str.replace('K', '000')
            df['avg_monthly_searches'] = df['avg_monthly_searches'].astype(str).str.replace(',', '')
            df['avg_monthly_searches'] = pd.to_numeric(df['avg_monthly_searches'], errors='coerce').fillna(10)
    except Exception as vol_err:
        st.warning(f"Error converting volume data: {vol_err}. Using fallback values.")
        df['avg_monthly_searches'] = 10
    
    # If after cleaning there's no data, stop
    if df.empty:
        st.error("No valid keyword data found in the CSV file after processing.")
        logging.warning("CSV contains no valid data rows after cleaning.")
        st.stop()
    
    # Show info about loaded data
    num_keywords = len(df)
    st.write(f"**Total keywords loaded:** {num_keywords}")
    st.write(f"**CSV columns detected:** {', '.join(df.columns)}")
    logging.info(f"Loaded {num_keywords} keywords from CSV with columns: {list(df.columns)}")
    
    # Show a sample of data for verification
    with st.expander("Preview of loaded data"):
        st.write(df.head())

except Exception as e:
    st.error(f"Error processing CSV file: {e}")
    logging.error(f"CSV processing error: {e}")
    
    # Provide troubleshooting suggestions
    st.write("### Troubleshooting suggestions:")
    st.write("1. Check if your CSV has the proper format with columns for 'keyword', 'cluster_name', and search volume.")
    st.write("2. Make sure your CSV uses a standard delimiter (comma or semicolon).")
    st.write("3. Check for special characters or encoding issues in your file.")
    st.write("4. If possible, open the CSV in Excel, save again, and re-upload.")
    
    st.stop()

# -------------------------------------------
# User Inputs and Filters
# -------------------------------------------
# 1. Cluster filtering: allow user to filter specific clusters (optional)
cluster_names = sorted(df["cluster_name"].unique())
selected_clusters = st.multiselect("Filter by Clusters (optional)", cluster_names)
if selected_clusters:
    # Filter dataframe to only include selected clusters
    df = df[df["cluster_name"].isin(selected_clusters)]
    st.write(f"Filtered to {len(selected_clusters)} selected clusters. Keywords remaining: {len(df)}")
    logging.info(f"Applied cluster filter. Selected clusters: {selected_clusters}. Remaining keywords: {len(df)}")
if df.empty:
    st.warning("No keywords to process after applying cluster filter. Please adjust your selection.")
    logging.warning("All keywords filtered out. Stopping execution.")
    st.stop()

# 2. Domain inputs for analysis
col1, col2 = st.columns(2)
with col1:
    domain_filter_input = st.text_input("Target Domain (optional)", help="Focus visibility analysis on this domain (e.g. your website)")
with col2:
    competitor_domain_input = st.text_input("Competitor Domain (optional)", help="Optional competitor domain to compare against")

# Clean and normalize domain inputs
def clean_domain(domain: str) -> str:
    """Utility to normalize domain input (remove scheme, path, and www)."""
    domain = domain.strip()
    if not domain:
        return ""
    # Remove protocol if present
    domain = domain.replace("http://", "").replace("https://", "")
    # Remove any path or query string
    domain = domain.split("/")[0]
    # Remove "www." prefix for consistency
    if domain.startswith("www."):
        domain = domain[4:]
    return domain.lower()

domain_filter = clean_domain(domain_filter_input)
competitor_domain = clean_domain(competitor_domain_input)

if domain_filter_input and not domain_filter:
    st.warning("Target Domain input seems invalid. Please enter a valid domain (e.g. 'example.com').")
    logging.warning(f"Target domain input provided ('{domain_filter_input}') is invalid after cleaning.")
if competitor_domain_input and not competitor_domain:
    st.warning("Competitor Domain input seems invalid. Please enter a valid domain.")
    logging.warning(f"Competitor domain input provided ('{competitor_domain_input}') is invalid after cleaning.")

# Ensure at least one domain is provided for visibility analysis
if not domain_filter and not competitor_domain:
    st.warning("No domain provided for analysis. Please enter a target domain and/or a competitor domain.")
    logging.info("No domain or competitor provided. Stopping execution as there's nothing to analyze.")
    st.stop()

# 3. SerpAPI API key input
api_key = st.text_input("SerpAPI API Key", type="password", help="Your SerpAPI key is required to fetch Google results.")
if not api_key:
    st.warning("Please provide a SerpAPI API key to run the analysis.")
    logging.warning("No SerpAPI API key provided by user.")
    st.stop()

# 4. Cost optimization options
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
    
    # Apply optimization
    try:
        original_count = len(df)
        df = select_representative_keywords(df, **cost_opt_params)
        optimized_count = len(df)
        st.write(f"Cost Optimization enabled: reduced {original_count} keywords to {optimized_count} representative keywords for querying.")
        logging.info(f"Cost optimization applied. Keywords reduced from {original_count} to {optimized_count}.")
        
        # Show breakdown by cluster
        cluster_counts = df.groupby('cluster_name').size().reset_index(name='keyword_count')
        st.write("Keywords selected per cluster:")
        st.write(cluster_counts)
    except Exception as opt_e:
        st.error(f"Error during cost optimization: {opt_e}. Proceeding with all keywords.")
        logging.error(f"Cost optimization failed: {opt_e}. Using all keywords instead.")
        use_cost_opt = False

# 5. Cache control
cache_control = st.checkbox("Use cache for API requests (reduces API usage)", value=True)
if cache_control:
    col1, col2 = st.columns(2)
    with col1:
        cache_days = st.slider("Cache expiry (days)", min_value=1, max_value=30, value=7)
    with col2:
        clear_cache = st.button("Clear existing cache")
    
    # Initialize cache
    api_cache = ApiCache(expiry_days=cache_days)
    
    # Clear cache if requested
    if clear_cache:
        try:
            import shutil
            shutil.rmtree(api_cache.cache_dir, ignore_errors=True)
            os.makedirs(api_cache.cache_dir, exist_ok=True)
            st.success("Cache cleared successfully.")
            logging.info("Cache directory cleared by user request.")
        except Exception as cache_err:
            st.error(f"Error clearing cache: {cache_err}")
            logging.error(f"Cache clear error: {cache_err}")
else:
    api_cache = None

# -------------------------------------------
# SERP API Queries Execution
# -------------------------------------------
# Prepare for querying the API
st.write("Starting SERP data retrieval... This may take a while for large keyword sets.")
progress_bar = st.progress(0)
results = []  # to collect results for each keyword
debug_messages = []  # to collect debug info per keyword
success_count = 0
no_result_count = 0
fail_count = 0
cache_hits = 0  # Track cache usage

# Loop through each keyword and perform API query
for idx, row in df.iterrows():
    keyword = str(row["keyword"]).strip()
    cluster = str(row["cluster_name"]).strip()
    try:
        volume = float(row["avg_monthly_searches"])
    except Exception:
        volume = 0.0

    # Defensive check: skip if keyword is empty after stripping
    if not keyword:
        logging.warning(f"Skipping empty keyword at index {idx}.")
        continue

    # Call SerpAPI (via utils.api_manager)
    serp_data = None
    
    # Check cache first if enabled
    using_cache = False
    if api_cache:
        api_params = {'api_key': api_key}
        cached_data = api_cache.get(keyword, api_params)
        if cached_data:
            serp_data = cached_data
            using_cache = True
            cache_hits += 1
            status_msg = f"Keyword '{keyword}': Using cached data."
            debug_messages.append(status_msg)
            logging.info(f"Cache hit for keyword '{keyword}'.")
    
    # If not found in cache, call API
    if not using_cache:
        try:
            # Call the API function from api_manager
            serp_data = api_manager.search_query(keyword, api_key=api_key)
            
            # Save to cache if enabled
            if api_cache and serp_data:
                api_cache.set(keyword, serp_data, {'api_key': api_key})
        except Exception as e:
            # Catch exceptions from API call (network issues, etc.)
            fail_count += 1
            error_msg = f"Keyword '{keyword}': API request failed - {e}"
            debug_messages.append(error_msg)
            logging.error(error_msg)
            serp_data = None

    # Handle API response if no exception:
    if serp_data is None:
        # If serp_data is None, treat as no results (could be due to exception or empty return)
        no_result_count += 1
        msg = f"Keyword '{keyword}': No results returned from API."
        debug_messages.append(msg)
        logging.warning(f"No results for keyword '{keyword}'.")
        # Append result entry with no domain/competitor found
        results.append({
            "keyword": keyword,
            "cluster": cluster,
            "volume": volume,
            "domain_rank": None,
            "competitor_rank": None
        })
    else:
        # If serp_data is not None, determine if it's already parsed or needs parsing
        serp_results = serp_data
        if isinstance(serp_data, dict):
            # Check for API errors in response
            if serp_data.get("error"):
                fail_count += 1
                err = serp_data.get("error")
                msg = f"Keyword '{keyword}': API error - {err}"
                debug_messages.append(msg)
                logging.error(f"API returned error for '{keyword}': {err}")
                serp_results = []  # treat as no results
            else:
                # Get organic results list from the JSON if available
                serp_results = serp_data.get("organic_results", serp_data.get("results", []))
        # If serp_results is still not a list (e.g., None or other), ensure it's a list
        if serp_results is None:
            serp_results = []

        # Analyze results for domain and competitor
        rank_domain = None
        rank_comp = None
        for res_index, res in enumerate(serp_results):
            # Each result might have 'link' or 'displayed_link' for URL
            url = ""
            if isinstance(res, str):
                url = res  # If api_manager returns a list of URLs (unlikely), handle as string
            elif isinstance(res, dict):
                url = res.get("link") or res.get("displayed_link") or ""
            else:
                continue
            if not url:
                continue
            # Extract netloc (domain) from URL
            netloc = urlparse(url).netloc.lower()
            if netloc.startswith("www."):
                netloc = netloc[4:]
            # Check if target domain matches
            if domain_filter and rank_domain is None:
                if netloc.endswith(domain_filter):
                    rank_domain = res.get("position", res_index + 1)
            # Check if competitor domain matches
            if competitor_domain and rank_comp is None:
                if netloc.endswith(competitor_domain):
                    rank_comp = res.get("position", res_index + 1)
            # If both found, no need to continue scanning this keyword's results
            if rank_domain is not None and rank_comp is not None:
                break

        # Log the outcome for this keyword
        if serp_results:
            success_count += 1
            status_msg = f"Keyword '{keyword}': {len(serp_results)} results retrieved."
            if domain_filter:
                status_msg += f" {'Target domain found at position ' + str(rank_domain) if rank_domain else 'Target domain not found'}."
            if competitor_domain:
                status_msg += f" {'Competitor found at position ' + str(rank_comp) if rank_comp else 'Competitor not found'}."
            debug_messages.append(status_msg)
            logging.info(f"Results for '{keyword}': Domain rank = {rank_domain}, Competitor rank = {rank_comp}")
        else:
            # If serp_results is empty list (no organic results)
            no_result_count += 1
            msg = f"Keyword '{keyword}': No organic results returned."
            debug_messages.append(msg)
            logging.warning(f"No organic results for keyword '{keyword}'.")

        # Append the results for calculation (including None ranks if not found)
        results.append({
            "keyword": keyword,
            "cluster": cluster,
            "volume": volume,
            "domain_rank": rank_domain,
            "competitor_rank": rank_comp
        })
    # Update progress bar
    progress = (idx + 1) / len(df)
    progress_bar.progress(min(progress, 1.0))

# Completed all queries - remove progress bar
progress_bar.empty()
logging.info(f"API querying complete. Success: {success_count}, No results: {no_result_count}, Failures: {fail_count}, Cache hits: {cache_hits}")

# -------------------------------------------
# Debugging Summary
# -------------------------------------------
# Display a summary of query outcomes
st.subheader("Query Execution Summary")
st.write(f"**Keywords processed:** {len(results)}")
st.write(f"**Successful API fetches:** {success_count}")
st.write(f"**Cache hits:** {cache_hits}")
st.write(f"**Keywords with no results:** {no_result_count}")
st.write(f"**API call failures:** {fail_count}")

if fail_count > 0:
    st.error(f"{fail_count} API queries failed. Please check your SerpAPI key or network connection.")
if success_count == 0 and cache_hits == 0:
    st.error("No successful results were obtained from the API. Unable to calculate visibility.")
    st.write("Please verify the CSV contents and API configuration, then try again.")
    st.expander("Debug Details").write("\n".join(debug_messages))
    st.stop()

# Optionally, provide detailed debug info per keyword in an expandable section
with st.expander("Detailed Debug Information per Keyword"):
    for msg in debug_messages:
        st.write(msg)

# -------------------------------------------
# Visibility Calculation
# -------------------------------------------
# Get improved CTR model
CTR_MAP = get_improved_ctr_map()

# Initialize aggregation structures
cluster_stats = {}  # to store aggregated stats per cluster
total_volume_all = 0.0
total_captured_volume_domain = 0.0
total_captured_volume_competitor = 0.0

# Aggregate results by cluster
for entry in results:
    cluster = entry["cluster"]
    vol = float(entry.get("volume", 0))
    rank_d = entry.get("domain_rank")
    rank_c = entry.get("competitor_rank")
    total_volume_all += vol

    if cluster not in cluster_stats:
        cluster_stats[cluster] = {
            "total_volume": 0.0,
            "keywords_count": 0,
            "domain_captured_volume": 0.0,
            "competitor_captured_volume": 0.0,
            "domain_keywords_ranked": 0,
            "competitor_keywords_ranked": 0
        }
    cluster_stats[cluster]["total_volume"] += vol
    cluster_stats[cluster]["keywords_count"] += 1

    # Calculate captured volume based on rank (if present in top results)
    if domain_filter and rank_d is not None:
        cluster_stats[cluster]["domain_captured_volume"] += vol * CTR_MAP.get(int(rank_d), 0)
        cluster_stats[cluster]["domain_keywords_ranked"] += 1
        total_captured_volume_domain += vol * CTR_MAP.get(int(rank_d), 0)
    if competitor_domain and rank_c is not None:
        cluster_stats[cluster]["competitor_captured_volume"] += vol * CTR_MAP.get(int(rank_c), 0)
        cluster_stats[cluster]["competitor_keywords_ranked"] += 1
        total_captured_volume_competitor += vol * CTR_MAP.get(int(rank_c), 0)

# Compute overall visibility percentages
overall_visibility_domain = None
overall_visibility_competitor = None
if domain_filter:
    overall_visibility_domain = (total_captured_volume_domain / total_volume_all * 100) if total_volume_all > 0 else 0.0
if competitor_domain:
    overall_visibility_competitor = (total_captured_volume_competitor / total_volume_all * 100) if total_volume_all > 0 else 0.0

# Calculate weighted visibility using improved method
weighted_visibility_domain = None
weighted_visibility_competitor = None
if domain_filter:
    weighted_visibility_domain = calculate_weighted_visibility([r for r in results if r.get("domain_rank") is not None])
if competitor_domain:
    weighted_visibility_competitor = calculate_weighted_visibility([r for r in results if r.get("competitor_rank") is not None])

# -------------------------------------------
# Results Output (Visibility Scores and Analysis)
# -------------------------------------------
st.subheader("SEO Visibility Results")

# Display overall visibility scores as metrics
if domain_filter and competitor_domain:
    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"Visibility Score ({domain_filter})", value=f"{weighted_visibility_domain:.1f}%", 
                delta=f"{(weighted_visibility_domain - overall_visibility_domain):.1f}%" 
                if overall_visibility_domain else None)
        st.write(f"Basic Visibility: {overall_visibility_domain:.1f}%")
    with col2:
        st.metric(f"Visibility Score ({competitor_domain})", value=f"{weighted_visibility_competitor:.1f}%", 
                delta=f"{(weighted_visibility_competitor - overall_visibility_competitor):.1f}%" 
                if overall_visibility_competitor else None)
        st.write(f"Basic Visibility: {overall_visibility_competitor:.1f}%")
elif domain_filter:
    st.metric(f"Visibility Score ({domain_filter})", value=f"{weighted_visibility_domain:.1f}%", 
             delta=f"{(weighted_visibility_domain - overall_visibility_domain):.1f}%" 
             if overall_visibility_domain else None)
    st.write(f"Basic Visibility: {overall_visibility_domain:.1f}%")
elif competitor_domain:
    st.metric(f"Visibility Score ({competitor_domain})", value=f"{weighted_visibility_competitor:.1f}%", 
             delta=f"{(weighted_visibility_competitor - overall_visibility_competitor):.1f}%" 
             if overall_visibility_competitor else None)
    st.write(f"Basic Visibility: {overall_visibility_competitor:.1f}%")

# Prepare a DataFrame for cluster-level details
output_rows = []
for cluster_name, stats in sorted(cluster_stats.items()):
    row = {
        "Cluster": cluster_name,
        "Keywords": stats["keywords_count"],
        "Total Search Volume": int(stats["total_volume"])
    }
    if domain_filter:
        # Visibility % per cluster for target domain
        vis_pct = (stats["domain_captured_volume"] / stats["total_volume"] * 100) if stats["total_volume"] > 0 else 0.0
        row[f"{domain_filter} Vis%"] = round(vis_pct, 1)
        row[f"{domain_filter} Keywords Ranked"] = stats["domain_keywords_ranked"]
    if competitor_domain:
        # Visibility % per cluster for competitor domain
        vis_pct_c = (stats["competitor_captured_volume"] / stats["total_volume"] * 100) if stats["total_volume"] > 0 else 0.0
        row[f"{competitor_domain} Vis%"] = round(vis_pct_c, 1)
        row[f"{competitor_domain} Keywords Ranked"] = stats["competitor_keywords_ranked"]
    output_rows.append(row)

if output_rows:
    cluster_df = pd.DataFrame(output_rows)
    # Order columns nicely: Cluster first, then Total, then domain and competitor metrics if present
    col_order = ["Cluster", "Keywords", "Total Search Volume"]
    if domain_filter:
        col_order += [f"{domain_filter} Vis%", f"{domain_filter} Keywords Ranked"]
    if competitor_domain:
        col_order += [f"{competitor_domain} Vis%", f"{competitor_domain} Keywords Ranked"]
    cluster_df = cluster_df[col_order]
    st.dataframe(cluster_df)
else:
    st.write("No cluster data to display.")

# Highlight opportunity areas if both domains provided
if domain_filter and competitor_domain:
    opp_clusters = []
    opp_keywords = []
    for entry in results:
        if entry.get("domain_rank") is None and entry.get("competitor_rank") is not None:
            opp_keywords.append(entry["keyword"])
            opp_clusters.append(entry["cluster"])
    opp_clusters = sorted(set(opp_clusters))
    if opp_keywords:
        st.subheader("Competitive Opportunity Analysis")
        st.write(f"**{competitor_domain}** is ranking for **{len(opp_keywords)}** of the keywords where **{domain_filter}** is not.")
        st.write(f"Clusters with opportunity (competitor present, target not): {', '.join(opp_clusters)}")
        logging.info(f"Competitor ranks in {len(opp_keywords)} keywords where target does not, across clusters: {opp_clusters}")
    else:
        st.write(f"{domain_filter} is ranking for all keywords that {competitor_domain} ranks for, within the selected set. No missed opportunities against this competitor.")

# -------------------------------------------
# Visualizations
# -------------------------------------------
def generate_visualizations(results, domain_filter, competitor_domain):
    """Generate visualizations for SEO visibility analysis."""
    # Create DataFrame of results for analysis
    results_df = pd.DataFrame(results)
    
    # 1. Ranking position distribution
    st.subheader("Ranking Position Distribution")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histograms for domain and competitor
    if domain_filter:
        domain_ranks = results_df[results_df['domain_rank'].notna()]['domain_rank']
        if not domain_ranks.empty:
            sns.histplot(domain_ranks, bins=list(range(1, 16)), alpha=0.5, label=domain_filter, color='blue', ax=ax)
    
    if competitor_domain:
        comp_ranks = results_df[results_df['competitor_rank'].notna()]['competitor_rank']
        if not comp_ranks.empty:
            sns.histplot(comp_ranks, bins=list(range(1, 16)), alpha=0.5, label=competitor_domain, color='red', ax=ax)
    
    ax.set_title('SERP Position Distribution')
    ax.set_xlabel('Position')
    ax.set_ylabel('Number of keywords')
    ax.legend()
    
    st.pyplot(fig)
    
    # 2. Visibility by cluster chart
    if domain_filter or competitor_domain:
        st.subheader("Visibility by Cluster")
        
        # Prepare data for chart
        cluster_vis_data = []
        
        for cluster, stats in cluster_stats.items():
            cluster_data = {'Cluster': cluster}
            
            if domain_filter:
                vis_pct = (stats["domain_captured_volume"] / stats["total_volume"] * 100) if stats["total_volume"] > 0 else 0.0
                cluster_data[domain_filter] = round(vis_pct, 1)
            
            if competitor_domain:
                vis_pct_c = (stats["competitor_captured_volume"] / stats["total_volume"] * 100) if stats["total_volume"] > 0 else 0.0
                cluster_data[competitor_domain] = round(vis_pct_c, 1)
            
            cluster_vis_data.append(cluster_data)
        
        # Create DataFrame for visualization
        cluster_vis_df = pd.DataFrame(cluster_vis_data)
        
        # Sort by visibility total
        if domain_filter and competitor_domain:
            cluster_vis_df['Total_Vis'] = cluster_vis_df[domain_filter] + cluster_vis_df[competitor_domain]
            cluster_vis_df = cluster_vis_df.sort_values('Total_Vis', ascending=False)
        elif domain_filter:
            cluster_vis_df = cluster_vis_df.sort_values(domain_filter, ascending=False)
        elif competitor_domain:
            cluster_vis_df = cluster_vis_df.sort_values(competitor_domain, ascending=False)
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Configure data for chart
        clusters = cluster_vis_df['Cluster']
        x = range(len(clusters))
        width = 0.35
        
        # Bars for domain and competitor
        if domain_filter and domain_filter in cluster_vis_df.columns:
            ax.bar(x, cluster_vis_df[domain_filter], width, label=domain_filter, color='blue')
        
        if competitor_domain and competitor_domain in cluster_vis_df.columns:
            if domain_filter:
                ax.bar([i + width for i in x], cluster_vis_df[competitor_domain], width, label=competitor_domain, color='red')
            else:
                ax.bar(x, cluster_vis_df[competitor_domain], width, label=competitor_domain, color='red')
        
        # Configure labels and legend
        ax.set_ylabel('Visibility (%)')
        ax.set_title('SEO Visibility by Cluster')
        ax.set_xticks([i + width/2 for i in x] if domain_filter and competitor_domain else x)
        ax.set_xticklabels(clusters, rotation=45, ha='right')
        ax.legend()
        
        # Adjust layout and display
        fig.tight_layout()
        st.pyplot(fig)

# Call visualization function if we have results
if success_count > 0 or cache_hits > 0:
    try:
        generate_visualizations(results, domain_filter, competitor_domain)
    except Exception as vis_error:
        st.error(f"Error generating visualizations: {vis_error}")
        logging.error(f"Visualization error: {vis_error}")

# -------------------------------------------
# Keyword Intent Analysis
# -------------------------------------------
def analyze_keyword_intents(results):
    """Analyze the intent of all keywords in the results."""
    # Counter for each intent type
    intent_counts = {
        "Informational": 0,
        "Transactional": 0,
        "Commercial": 0,
        "Navigational": 0,
        "Undetermined": 0
    }
    
    # Analysis by cluster
    cluster_intents = {}
    
    # Analyze each keyword
    for entry in results:
        keyword = entry.get("keyword", "")
        cluster = entry.get("cluster", "")
        
        # Get intent
        intent = analyze_keyword_intent(keyword)
        
        # Update general count
        intent_counts[intent] += 1
        
        # Update cluster analysis
        if cluster not in cluster_intents:
            cluster_intents[cluster] = {
                "Informational": 0,
                "Transactional": 0,
                "Commercial": 0,
                "Navigational": 0,
                "Undetermined": 0
            }
        cluster_intents[cluster][intent] += 1
    
    return intent_counts, cluster_intents

# Function to visualize intent distribution
def visualize_intents(intent_counts, cluster_intents):
    st.subheader("Search Intent Analysis")
    
    # General intent chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for pie chart
    labels = list(intent_counts.keys())
    sizes = list(intent_counts.values())
    
    # Filter zero-value labels
    non_zero_labels = [label for label, size in zip(labels, sizes) if size > 0]
    non_zero_sizes = [size for size in sizes if size > 0]
    
    if non_zero_sizes:
        # Create chart
        ax.pie(non_zero_sizes, labels=non_zero_labels, autopct='%1.1f%%', 
               shadow=False, startangle=90)
        ax.axis('equal')  # Circular aspect
        plt.title('Search Intent Distribution')
        
        st.pyplot(fig)
        
        # Show analysis by cluster
        st.write("### Intent by Cluster")
        
        # Create DataFrame for better visualization
        intent_data = []
        for cluster, intents in cluster_intents.items():
            row = {"Cluster": cluster}
            row.update(intents)
            intent_data.append(row)
        
        intent_df = pd.DataFrame(intent_data)
        
        # Calculate percentages to facilitate analysis
        for intent_type in ["Informational", "Transactional", "Commercial", "Navigational"]:
            if intent_type in intent_df.columns:
                total = intent_df[list(intent_counts.keys())].sum(axis=1)
                intent_df[f"%{intent_type}"] = (intent_df[intent_type] / total * 100).round(1)
        
        # Sort by cluster
        intent_df = intent_df.sort_values("Cluster")
        
        # Show table
        st.dataframe(intent_df)
        
        # Recommendations based on intent
        st.write("### Intent-Based Recommendations")
        
        # Determine predominant intent type
        predominant_intent = max(intent_counts.items(), key=lambda x: x[1])[0]
        
        if predominant_intent == "Informational":
            st.write("""
            **Recommendations for Informational Keywords:**
            - Create detailed educational content (guides, tutorials, FAQs)
            - Structure content to answer specific questions
            - Optimize for featured snippets
            - Include explanatory images, infographics and videos
            """)
        elif predominant_intent == "Transactional":
            st.write("""
            **Recommendations for Transactional Keywords:**
            - Optimize product and category pages
            - Include clear and visible CTAs
            - Display pricing, shipping and availability information
            - Implement testimonials and reviews
            - Optimize for local search if relevant
            """)
        elif predominant_intent == "Commercial":
            st.write("""
            **Recommendations for Commercial Keywords:**
            - Create comparative content (tables, pros and cons)
            - Develop detailed product/service reviews
            - Implement testimonial and case study sections
            - Create content like "best X for Y" or "alternatives to Z"
            """)
        elif predominant_intent == "Navigational":
            st.write("""
            **Recommendations for Navigational Keywords:**
            - Optimize brand pages and main landing pages
            - Implement schema markup to facilitate navigation in SERPs
            - Ensure presence on Google My Business if applicable
            - Optimize user experience on main pages
            """)

# Call intent analysis if we have results
if success_count > 0 or cache_hits > 0:
    try:
        # Analyze intents
        intent_counts, cluster_intents = analyze_keyword_intents(results)
        # Visualize results
        visualize_intents(intent_counts, cluster_intents)
    except Exception as intent_error:
        st.error(f"Error in intent analysis: {intent_error}")
        logging.error(f"Intent analysis error: {intent_error}")

# -------------------------------------------
# Cluster Correlation Analysis
# -------------------------------------------
def analyze_cluster_correlations(results):
    """
    Analyze correlations between clusters based on ranking patterns.
    No additional API required, uses already obtained data.
    """
    # Create structure for analysis
    clusters = sorted(set(entry.get("cluster", "") for entry in results))
    cluster_rankings = {cluster: [] for cluster in clusters}
    
    # Collect rankings by cluster
    for entry in results:
        cluster = entry.get("cluster", "")
        rank = entry.get("domain_rank")
        
        if cluster and rank is not None:
            cluster_rankings[cluster].append(rank)
    
    # Calculate average ranking by cluster
    cluster_avg_rank = {}
    for cluster, ranks in cluster_rankings.items():
        if ranks:  # Verify there's data
            cluster_avg_rank[cluster] = sum(ranks) / len(ranks)
    
    # Identify clusters with similar performance (possible correlation)
    correlations = []
    processed = set()
    
    for cluster1 in clusters:
        if cluster1 not in cluster_avg_rank:
            continue
            
        for cluster2 in clusters:
            if cluster2 not in cluster_avg_rank or cluster1 == cluster2:
                continue
                
            # Avoid duplicates (A-B vs B-A)
            pair = tuple(sorted([cluster1, cluster2]))
            if pair in processed:
                continue
                
            processed.add(pair)
            
            # Calculate ranking similarity (lower difference = higher correlation)
            rank_diff = abs(cluster_avg_rank[cluster1] - cluster_avg_rank[cluster2])
            
            # Only include clusters with small difference (possible correlation)
            if rank_diff < 3.0:  # Adjustable threshold
                correlations.append({
                    "cluster1": cluster1,
                    "cluster2": cluster2,
                    "rank_diff": round(rank_diff, 2),
                    "correlation": round(max(0, 1 - (rank_diff / 10)), 2)  # Conversion to 0-1 scale
                })
    
    # Sort by correlation strength
    correlations.sort(key=lambda x: x["correlation"], reverse=True)
    
    return correlations, cluster_avg_rank

# Visualize cluster correlations
def visualize_cluster_correlations(correlations, cluster_avg_rank):
    st.subheader("Cluster Correlation Analysis")
    
    if not correlations:
        st.write("No significant correlations found between clusters.")
        return
    
    # Show clusters with average ranking
    st.write("### Average Position by Cluster")
    
    # Convert to DataFrame for better visualization
    avg_rank_data = [{"Cluster": cluster, "Average Position": round(rank, 2)} 
                     for cluster, rank in cluster_avg_rank.items()]
    avg_rank_df = pd.DataFrame(avg_rank_data).sort_values("Average Position")
    
    st.dataframe(avg_rank_df)
    
    # Show found correlations
    st.write("### Clusters with similar behavior (possible correlation)")
    
    # Convert to DataFrame
    corr_df = pd.DataFrame(correlations)
    
    if not corr_df.empty:
        # Rename columns for better understanding
        corr_df = corr_df.rename(columns={
            "cluster1": "Cluster A", 
            "cluster2": "Cluster B",
            "rank_diff": "Ranking difference",
            "correlation": "Correlation index"
        })
        
        st.dataframe(corr_df)
        
        # Recommendations based on correlations
        st.write("### Recommendations based on correlations")
        st.write("""
        Clusters with high correlation tend to have similar behavior in search results,
        which may indicate:
        
        - **Internal linking opportunities**: Connect content from correlated clusters
        - **Complementary topics**: Develop content that combines aspects of both clusters
        - **Keyword strategy**: If one cluster performs well, apply similar strategies to the correlated cluster
        - **Content consolidation**: For highly correlated clusters, consider unifying or relating the content
        """)

# Call correlation analysis if we have results and multiple clusters
if (success_count > 0 or cache_hits > 0) and len(set(entry.get("cluster", "") for entry in results)) > 1:
    try:
        # Analyze correlations
        correlations, cluster_avg_rank = analyze_cluster_correlations(results)
        # Visualize results
        visualize_cluster_correlations(correlations, cluster_avg_rank)
    except Exception as corr_error:
        st.error(f"Error in correlation analysis: {corr_error}")
        logging.error(f"Correlation analysis error: {corr_error}")

# -------------------------------------------
# Optimization Suggestions
# -------------------------------------------
def generate_optimization_suggestions(results, domain_filter, competitor_domain):
    """
    Generate specific SEO optimization suggestions based on data analysis.
    No additional API required, uses already obtained data.
    """
    if not domain_filter or not competitor_domain:
        return []
    
    suggestions = []
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # 1. Identify keywords where competitor ranks better
    better_comp = results_df[
        (results_df['domain_rank'].notna()) & 
        (results_df['competitor_rank'].notna()) & 
        (results_df['competitor_rank'] < results_df['domain_rank'])
    ]
    
    # 2. Identify keywords where competitor ranks but you don't
    comp_only = results_df[
        (results_df['domain_rank'].isna()) & 
        (results_df['competitor_rank'].notna())
    ]
    
    # 3. Identify high-volume keywords with opportunity
    high_vol_threshold = results_df['volume'].quantile(0.75)  # Top 25% by volume
    high_vol_opps = pd.concat([
        better_comp[better_comp['volume'] >= high_vol_threshold],
        comp_only[comp_only['volume'] >= high_vol_threshold]
    ])
    
    # 4. Analyze clusters with greatest competitive gap
    cluster_gap = {}
    
    for cluster in results_df['cluster'].unique():
        cluster_data = results_df[results_df['cluster'] == cluster]
        
        # Calculate average position for domain and competitor
        domain_avg = cluster_data['domain_rank'].mean()
        comp_avg = cluster_data['competitor_rank'].mean()
        
        # If both have data, calculate gap
        if not pd.isna(domain_avg) and not pd.isna(comp_avg):
            gap = domain_avg - comp_avg
            cluster_gap[cluster] = gap
    
    # Sort clusters by gap (largest gap first)
    gap_clusters = sorted(cluster_gap.items(), key=lambda x: x[1], reverse=True)
    
    # Generate suggestions
    
    # 1. Suggestion for high-volume keywords
    if not high_vol_opps.empty:
        suggestions.append({
            "type": "high_volume",
            "title": "Optimize high-volume keywords",
            "description": f"There are {len(high_vol_opps)} high-volume keywords where your competitor has an advantage.",
            "keywords": high_vol_opps.sort_values('volume', ascending=False).head(5)['keyword'].tolist(),
            "action": "Prioritize optimizing these keywords to increase visibility in high-volume searches."
        })
    
    # 2. Suggestion for clusters with largest gap
    if gap_clusters:
        top_gap_clusters = [cluster for cluster, gap in gap_clusters[:3] if gap > 0]
        if top_gap_clusters:
            suggestions.append({
                "type": "cluster_gap",
                "title": "Close gap in key clusters",
                "description": f"There are clusters where your competitor has a significant advantage in positioning.",
                "clusters": top_gap_clusters,
                "action": "Analyze these clusters in depth and strengthen related content to close the competitive gap."
            })
    
    # 3. Suggestion for keywords where only competitor ranks
    if not comp_only.empty:
        suggestions.append({
            "type": "competitor_only",
            "title": "Opportunities in uncovered keywords",
            "description": f"Your competitor ranks for {len(comp_only)} keywords where your domain doesn't appear.",
            "keywords": comp_only.sort_values('volume', ascending=False).head(5)['keyword'].tolist(),
            "action": "Develop specific content for these keywords to close coverage gaps."
        })
    
    # 4. Suggestion for position improvement
    rank_improvements = better_comp[better_comp['domain_rank'] <= 20]  # Keywords where you're already in top 20
    if not rank_improvements.empty:
        suggestions.append({
            "type": "rank_improvement",
            "title": "Improve existing positions",
            "description": f"There are {len(rank_improvements)} keywords where you're close but competitor ranks better.",
            "keywords": rank_improvements.sort_values('domain_rank').head(5)['keyword'].tolist(),
            "action": "Optimize and update existing content to gain positions in these keywords where you already have presence."
        })
    
    return suggestions

# Display optimization suggestions
def display_optimization_suggestions(suggestions):
    st.subheader("SEO Optimization Suggestions")
    
    if not suggestions:
        st.write("There is not enough comparative data to generate specific suggestions.")
        return
    
    # Display each suggestion in an attractive visual format
    for i, suggestion in enumerate(suggestions):
        with st.expander(f"💡 {suggestion['title']}", expanded=True):
            st.write(f"**Description:** {suggestion['description']}")
            
            # Show keywords if they exist
            if 'keywords' in suggestion:
                st.write("**Key keywords:**")
                for kw in suggestion['keywords']:
                    st.write(f"- {kw}")
            
            # Show clusters if they exist
            if 'clusters' in suggestion:
                st.write("**Key clusters:**")
                for cluster in suggestion['clusters']:
                    st.write(f"- {cluster}")
            
            # Show recommended action
            st.write(f"**Recommended action:** {suggestion['action']}")
            
            # Add specific tips based on suggestion type
            if suggestion['type'] == 'high_volume':
                st.write("""
                **Additional tips:**
                - Create comprehensive content that covers all aspects of these keywords
                - Optimize titles, meta descriptions and H1-H6 structure
                - Improve user experience and time on page
                - Consider creating "skyscraper" content for these topics
                """)
            elif suggestion['type'] == 'cluster_gap':
                st.write("""
                **Additional tips:**
                - Analyze competitor content in these clusters
                - Identify missing information in your content
                - Improve semantic structure with related terms
                - Implement relevant schema markup for these clusters
                """)
            elif suggestion['type'] == 'competitor_only':
                st.write("""
                **Additional tips:**
                - Create new specific content for these keywords
                - Analyze how your competitor is positioning this content
                - Implement an internal linking structure that connects this new content with existing relevant pages
                """)
            elif suggestion['type'] == 'rank_improvement':
                st.write("""
                **Additional tips:**
                - Update and expand existing content
                - Improve technical factors (speed, mobile-friendly, etc.)
                - Get quality backlinks for these pages
                - Optimize CTR from search results (title and meta description)
                """)

# Call optimization suggestions if we have comparative results
if (success_count > 0 or cache_hits > 0) and domain_filter and competitor_domain:
    try:
        # Generate suggestions
        suggestions = generate_optimization_suggestions(results, domain_filter, competitor_domain)
        # Display suggestions
        display_optimization_suggestions(suggestions)
    except Exception as sugg_error:
        st.error(f"Error generating suggestions: {sugg_error}")
        logging.error(f"Suggestion generation error: {sugg_error}")

# -------------------------------------------
# Export Results Section
# -------------------------------------------
# Create a DataFrame with all detailed results
full_results_df = pd.DataFrame(results)

# Add download button for full results
if not full_results_df.empty:
    st.subheader("Export Results")
    
    # Convert DataFrame to CSV for download
    csv = full_results_df.to_csv(index=False)
    
    # Download button
    st.download_button(
        label="Download complete results (CSV)",
        data=csv,
        file_name=f"seo_visibility_results_{domain_filter or 'data'}.csv",
        mime="text/csv"
    )
    
    # Option to download cluster analysis
    if 'cluster_df' in locals() and not cluster_df.empty:
        csv_clusters = cluster_df.to_csv(index=False)
        st.download_button(
            label="Download cluster analysis (CSV)",
            data=csv_clusters,
            file_name=f"seo_cluster_analysis_{domain_filter or 'data'}.csv",
            mime="text/csv"
        )

# Add app footer with version info
st.markdown("""
---
### SEO Visibility Estimator v2.0
An improved tool for analyzing SEO visibility across keyword clusters.

**Features:**
- Improved CSV handling with flexible column detection
- API caching system for efficient SerpAPI usage
- Advanced visibility calculation with weighted model
- Search intent analysis
- Cluster correlation detection
- Specific SEO optimization suggestions
""")

# Log app completion
logging.info("App execution completed successfully.")
