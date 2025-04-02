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
import requests  # For fallback SERPAPI queries

# Configure logging for debugging
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)

# -------------------------------------------
# Import utility modules (or create dummy ones)
# -------------------------------------------
try:
    from utils import data_processing, seo_calculator, optimization, api_manager
except ImportError:
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
# AI Integration Functions
# -------------------------------------------
def generate_seo_suggestions(keyword):
    prompt = (
        f"Keyword: \"{keyword}\"\n"
        "1. **Intent**: Identify the search intent (Informational, Navigational, Transactional, or Commercial) and briefly explain why.\n"
        "2. **Content Type**: Recommend the best type of content (e.g., blog article, product page, landing page).\n"
        "3. **Title Tag**: Propose an SEO-optimized title tag (<= 60 characters) for this keyword.\n"
        "4. **Meta Description**: Draft a meta description (<= 155 characters) incorporating the keyword and enticing clicks.\n"
        "5. **CTA**: Provide a short call-to-action suitable for this page.\n"
        "6. **Content Outline**: Outline key sections for a comprehensive article or landing page based on the keyword's intent.\n"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
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

def analyze_serp_features(keyword, features):
    prompt = (
        f"Analyze the following SERP features for the keyword '{keyword}':\n{json.dumps(features)}\n"
        "Based on these features (such as organic results, local pack, knowledge graph, video results, ads, etc.), "
        "provide an analysis of the SERP result types and actionable SEO recommendations for improving ranking and click-through."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        logging.error(f"OpenAI API error for SERP feature analysis for keyword '{keyword}': {e}")
        return "AI SERP feature analysis unavailable."

# -------------------------------------------
# Domain Inputs: Target and Competitor Domains
# -------------------------------------------
st.title("SEO Visibility Estimator")
st.markdown("""
Upload a CSV of keywords (with **keyword**, **cluster_name**, **Avg. monthly searches** columns) and configure options to estimate SEO visibility.
This tool compares your target domains against competitor domains by querying Google SERPs via SERPAPI.
""")

# Leave domain fields empty by default
target_domains_input = st.text_input("Target Domains (comma separated)", "")
target_domains = [d.strip().lower() for d in target_domains_input.split(",") if d.strip()]

competitor_domains_input = st.text_input("Competitor Domains (comma separated)", "")
def clean_domain(domain: str) -> str:
    domain = domain.strip().replace("http://", "").replace("https://", "")
    domain = domain.split("/")[0]
    return domain[4:] if domain.startswith("www.") else domain.lower()
competitor_domains = [clean_domain(d) for d in competitor_domains_input.split(",") if clean_domain(d)]

# Combine into one list for analysis
all_domains = target_domains + competitor_domains

if not target_domains:
    st.warning("Please provide at least one target domain.")
    st.stop()

# -------------------------------------------
# CSV File Input with Error Handling
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
            st.warning("No volume data found. Using default value 10 for all keywords.")
            df['avg_monthly_searches'] = 10
    else:
        if 'avg. monthly searches' in df.columns and 'avg_monthly_searches' not in df.columns:
            df = df.rename(columns={'avg. monthly searches': 'avg_monthly_searches'})
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        st.error(f"CSV is missing required columns: {missing}. Found columns: {list(df.columns)}.")
        st.stop()
    df = df.dropna(subset=['keyword'])
    try:
        df['avg_monthly_searches'] = (df['avg_monthly_searches'].astype(str)
                                      .str.replace('K', '000')
                                      .str.replace(',', ''))
        df['avg_monthly_searches'] = pd.to_numeric(df['avg_monthly_searches'], errors='coerce').fillna(10)
    except Exception as vol_err:
        st.warning(f"Error converting volume data: {vol_err}. Using fallback values.")
        df['avg_monthly_searches'] = 10
    st.write(f"**Total keywords loaded:** {len(df)}")
    with st.expander("Preview of loaded data"):
        st.write(df.head())
except Exception as e:
    st.error(f"Error processing CSV file: {e}")
    st.stop()

# -------------------------------------------
# Additional Inputs: Localization, API Keys, Filtering, etc.
# -------------------------------------------
st.markdown("#### SERPAPI Localization Options")
country_code = st.text_input("Country Code (gl)", "us")
language_code = st.text_input("Language Code (hl)", "en")
location_input = st.text_input("City/Location (optional)", "")

api_key = st.text_input("SERPAPI API Key", type="password", help="Your SERPAPI key is required.")
if not api_key:
    st.warning("Please provide a SERPAPI API key to run the analysis.")
    st.stop()

openai_api_key = st.text_input("OpenAI API Key (optional)", type="password", help="Provide your OpenAI key for AI-driven suggestions.")
if openai_api_key:
    openai.api_key = openai_api_key
else:
    st.info("OpenAI API key not provided. AI-based suggestions will be skipped.")

cluster_names = sorted(df["cluster_name"].unique())
selected_clusters = st.multiselect("Filter by Clusters (optional)", cluster_names)
if selected_clusters:
    df = df[df["cluster_name"].isin(selected_clusters)]
    st.write(f"Filtered to {len(selected_clusters)} clusters. Keywords remaining: {len(df)}")
if df.empty:
    st.warning("No keywords to process after applying cluster filter.")
    st.stop()

use_cost_opt = st.checkbox("Use cost optimization (query only representative keywords per cluster)", value=False)
if use_cost_opt:
    col1, col2 = st.columns(2)
    samples_per_cluster = col1.slider("Keywords per cluster", min_value=1, max_value=10, value=3)
    min_volume_keywords = col2.slider("Min low-volume keywords", min_value=1, max_value=5, value=1)
    try:
        original_count = len(df)
        df = select_representative_keywords(df, samples_per_cluster, min_volume_keywords)
        st.write(f"Cost Optimization enabled: reduced {original_count} keywords to {len(df)} representative keywords.")
        st.write(df.groupby('cluster_name').size().reset_index(name='keyword_count'))
    except Exception as opt_e:
        st.error(f"Error during cost optimization: {opt_e}. Proceeding with all keywords.")

cache_control = st.checkbox("Use cache for API requests (reduces API usage)", value=True)
if cache_control:
    col1, col2 = st.columns(2)
    cache_days = col1.slider("Cache expiry (days)", min_value=1, max_value=30, value=7)
    clear_cache = col2.button("Clear existing cache")
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
    st.info("Click 'Run Analysis' to start processing the data.")
    st.stop()

# -------------------------------------------
# Define Visibility Scoring Functions
# -------------------------------------------
def simple_score(position):
    return 31 - position if (position is not None and position <= 30) else 0

def calculate_simple_visibility_for_domain(results, domain):
    score_sum = 0
    for entry in results:
        pos = entry["rankings"].get(domain)
        score_sum += simple_score(pos)
    return score_sum

def calculate_weighted_visibility(filtered_results):
    ctr_map = {
        1: 0.3042, 2: 0.1559, 3: 0.0916, 4: 0.0651, 5: 0.0478,
        6: 0.0367, 7: 0.0289, 8: 0.0241, 9: 0.0204, 10: 0.0185,
        11: 0.0156, 12: 0.0138, 13: 0.0122, 14: 0.0108, 15: 0.0096
    }
    total_volume = sum(float(entry.get("volume", 0)) for entry in filtered_results)
    captured_clicks = 0
    for entry in filtered_results:
        vol = float(entry.get("volume", 0))
        pos = entry.get("domain_rank")
        if pos is not None and int(pos) in ctr_map:
            captured_clicks += vol * ctr_map[int(pos)]
    max_clicks = total_volume * ctr_map[1] if total_volume > 0 else 0
    return round((captured_clicks / max_clicks * 100) if max_clicks > 0 else 0, 2)

def calculate_weighted_visibility_for_domain(results, domain):
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
    return calculate_weighted_visibility(filtered)

# -------------------------------------------
# SERPAPI Query Execution and Ranking Extraction
# -------------------------------------------
st.write("Starting SERPAPI data retrieval... (this may take a while)")
progress_bar = st.progress(0)
results = []
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
    if not using_cache:
        try:
            serp_data = api_manager.search_query(keyword, api_key=api_key, params=api_params)
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
            netloc = urlparse(url).netloc.lower()
            if netloc.startswith("www."):
                netloc = netloc[4:]
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
            "features": features
        })
    progress_bar.progress(min((idx + 1) / len(df), 1.0))
progress_bar.empty()
logging.info(f"Querying complete. Success: {success_count}, No results: {no_result_count}, Failures: {fail_count}, Cache hits: {cache_hits}")

st.subheader("Query Execution Summary")
st.write(f"Keywords processed: {len(results)}")
st.write(f"Successful API fetches: {success_count}")
st.write(f"Cache hits: {cache_hits}")
st.write(f"Keywords with no results: {no_result_count}")
st.write(f"API call failures: {fail_count}")
if fail_count > 0:
    st.error(f"{fail_count} API queries failed. Check your SERPAPI key or network.")
if success_count == 0 and cache_hits == 0:
    st.error("No successful results. Unable to calculate visibility.")
    st.expander("Debug Details").write("\n".join(debug_messages))
    st.stop()
with st.expander("Detailed Debug Information"):
    for msg in debug_messages:
        st.write(msg)

# -------------------------------------------
# Visibility Calculation & Aggregation
# -------------------------------------------
CTR_MAP = {
    1: 0.3042, 2: 0.1559, 3: 0.0916, 4: 0.0651, 5: 0.0478,
    6: 0.0367, 7: 0.0289, 8: 0.0241, 9: 0.0204, 10: 0.0185,
    11: 0.0156, 12: 0.0138, 13: 0.0122, 14: 0.0108, 15: 0.0096
}
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
        weighted = vol * CTR_MAP.get(int(pos), 0) if (pos is not None and int(pos) in CTR_MAP) else 0
        simple = simple_score(pos)
        cluster_stats[cluster]["weighted"][domain] += weighted
        cluster_stats[cluster]["simple"][domain] += simple
        overall_totals[domain]["weighted"] += weighted
        overall_totals[domain]["simple"] += simple

# -------------------------------------------
# Results Output: Keyword Ranking Table
# -------------------------------------------
st.subheader("Keyword Ranking Table")
ranking_data = []
for entry in results:
    row = {"Keyword": entry["keyword"], "Cluster": entry["cluster"], "Volume": entry["volume"]}
    for domain in all_domains:
        row[domain] = entry["rankings"].get(domain)
    ranking_data.append(row)
ranking_df = pd.DataFrame(ranking_data)
st.dataframe(ranking_df)

# -------------------------------------------
# Results Output: Visibility Summary Table
# -------------------------------------------
st.subheader("Visibility Summary")
summary_rows = []
total_volume_all = sum(float(e.get("volume", 0)) for e in results)
for domain in all_domains:
    summary_rows.append({
        "Domain": domain,
        "Simple Visibility": overall_totals[domain]["simple"],
        "Weighted Visibility (%)": round((overall_totals[domain]["weighted"] / (total_volume_all * CTR_MAP[1]) * 100) if total_volume_all > 0 else 0, 2)
    })
summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df)

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
st.subheader("Per-Cluster Visibility Summary")
st.dataframe(cluster_df)

# -------------------------------------------
# SERP Feature Analysis and SEO Recommendations per Keyword
# -------------------------------------------
if openai_api_key and st.button("Generate SERP Feature Analysis for Each Keyword"):
    st.info("Generating SERP feature analysis (this may take some time)...")
    serp_feature_analyses = []
    for entry in results:
        features = entry.get("features", {})
        if features:
            analysis = analyze_serp_features(entry["keyword"], features)
        else:
            analysis = "No additional SERP features detected."
        serp_feature_analyses.append({
            "Keyword": entry["keyword"],
            "Cluster": entry["cluster"],
            "SERP Features": features,
            "Analysis & Recommendations": analysis
        })
    serp_feature_df = pd.DataFrame(serp_feature_analyses)
    st.subheader("SERP Feature Analysis for Representative Keywords")
    st.dataframe(serp_feature_df)

# -------------------------------------------
# Keyword Intent Analysis and AI Content Strategy
# -------------------------------------------
if openai_api_key and st.button("Generate AI Suggestions for Each Keyword"):
    st.info("Generating AI suggestions (this may take some time)...")
    for entry in results:
        entry["ai_suggestion"] = generate_seo_suggestions(entry["keyword"])
    ai_df = pd.DataFrame([{"Keyword": e["keyword"], "Cluster": e["cluster"], "AI Suggestion": e.get("ai_suggestion", "")} for e in results])
    st.subheader("AI-Based Keyword Content Strategy")
    st.dataframe(ai_df)

if openai_api_key:
    st.subheader("AI-Driven Cluster-Level Content Strategy")
    for cluster, stats in cluster_stats.items():
        cluster_keywords = [entry["keyword"] for entry in results if entry["cluster"] == cluster]
        if cluster_keywords:
            strategy = generate_cluster_strategy(cluster, cluster_keywords)
            st.markdown(f"**Cluster: {cluster}**")
            st.write(strategy)

# -------------------------------------------
# Export Results Section
# -------------------------------------------
full_results_df = pd.DataFrame(results)
if not full_results_df.empty:
    st.subheader("Export Results")
    csv_full = full_results_df.to_csv(index=False)
    st.download_button("Download complete results (CSV)", data=csv_full, file_name="seo_visibility_results.csv", mime="text/csv")
    csv_cluster = cluster_df.to_csv(index=False)
    st.download_button("Download cluster analysis (CSV)", data=csv_cluster, file_name="seo_cluster_analysis.csv", mime="text/csv")

# -------------------------------------------
# Visualizations
# -------------------------------------------
def generate_visualizations(results, all_domains):
    results_df = pd.DataFrame(results)
    st.subheader("Ranking Position Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    for domain in all_domains:
        domain_data = results_df[results_df['rankings'].apply(lambda r: r.get(domain) is not None)]
        if not domain_data.empty:
            ranks = domain_data['rankings'].apply(lambda r: r.get(domain))
            sns.histplot(ranks, bins=list(range(1, 16)), alpha=0.5, label=domain, ax=ax)
    ax.set_title('SERP Position Distribution')
    ax.set_xlabel('Position')
    ax.set_ylabel('Number of Keywords')
    ax.legend()
    st.pyplot(fig)
    
    st.subheader("Cluster Visibility Comparison (Weighted)")
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
    st.pyplot(fig)

if success_count > 0 or cache_hits > 0:
    try:
        generate_visualizations(results, all_domains)
    except Exception as vis_error:
        st.error(f"Error generating visualizations: {vis_error}")

# -------------------------------------------
# App Footer
# -------------------------------------------
st.markdown("""
---
### SEO Visibility Estimator v2.0
An advanced tool for analyzing SEO visibility with AI-driven content strategy recommendations.
""")
logging.info("App execution completed successfully.")
