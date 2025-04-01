# -------------------------------------------
# Section: Imports and Configuration
# -------------------------------------------
import streamlit as st
import pandas as pd
import logging
from urllib.parse import urlparse

# Configure logging for debugging
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)

# Import utility modules from the project
from utils import data_processing, seo_calculator, optimization, api_manager

# (Optionally, if specific functions are known, import them directly)
# from utils.api_manager import search_query
# from utils.seo_calculator import calculate_visibility

# -------------------------------------------
# Section: Streamlit App Title and Description
# -------------------------------------------
st.title("SEO Visibility Estimator")
st.markdown("""
Upload a CSV of keywords (with **keyword**, **cluster_name**, **Avg. monthly searches** columns) and configure options to estimate SEO visibility.
This tool will calculate a visibility score for a given domain (and an optional competitor) across keyword clusters by querying Google SERPs via SerpAPI.
""")


# -------------------------------------------
# Section: CSV File Input
# -------------------------------------------
uploaded_file = st.file_uploader("Upload Keyword CSV", type=["csv"])
if not uploaded_file:
    st.info("Please upload a CSV file to continue.")
    st.stop()

# Read the CSV file into a DataFrame
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Error reading CSV file: {e}")
    logging.error(f"CSV read error: {e}")
    st.stop()

# Validate required columns in CSV
required_cols = {"keyword", "cluster_name", "Avg. monthly searches"}
if not required_cols.issubset(df.columns):
    st.error(f"CSV is missing required columns. Found columns: {list(df.columns)}. Expected at least: {required_cols}.")
    logging.error(f"CSV missing columns. Found: {list(df.columns)}; Expected: {required_cols}")
    st.stop()

# Drop any rows with missing values in required columns (defensive check)
df = df.dropna(subset=["keyword", "cluster_name", "Avg. monthly searches"])
if df.empty:
    st.error("No keyword data found in the CSV file after removing empty rows.")
    logging.warning("CSV contains no valid data rows.")
    st.stop()

# Show basic info about the loaded data
num_keywords = len(df)
st.write(f"**Total keywords loaded:** {num_keywords}")
logging.info(f"Loaded {num_keywords} keywords from CSV.")

# -------------------------------------------
# Section: User Inputs and Filters
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

# 4. (Optional) Cost optimization toggle if implemented
use_cost_opt = False
if hasattr(optimization, "select_representative_keywords"):
    use_cost_opt = st.checkbox("Use cost optimization (query only representative keywords per cluster)", value=False)
    if use_cost_opt:
        # If the optimization module provides a function to reduce keywords, use it
        try:
            original_count = len(df)
            df = optimization.select_representative_keywords(df)
            optimized_count = len(df)
            st.write(f"Cost Optimization enabled: reduced {original_count} keywords to {optimized_count} representative keywords for querying.")
            logging.info(f"Cost optimization applied. Keywords reduced from {original_count} to {optimized_count}.")
        except Exception as opt_e:
            st.error(f"Error during cost optimization: {opt_e}. Proceeding with all keywords.")
            logging.error(f"Cost optimization failed: {opt_e}. Using all keywords instead.")
            use_cost_opt = False
else:
    logging.info("Cost optimization module not available or not used. Proceeding with all keywords.")

# -------------------------------------------
# Section: SERP API Queries Execution
# -------------------------------------------
# Prepare for querying the API
st.write("Starting SERP data retrieval... This may take a while for large keyword sets.")
progress_bar = st.progress(0)
results = []  # to collect results for each keyword
debug_messages = []  # to collect debug info per keyword
success_count = 0
no_result_count = 0
fail_count = 0

# Loop through each keyword and perform API query
for idx, row in df.iterrows():
    keyword = str(row["keyword"]).strip()
    cluster = str(row["cluster_name"]).strip()
    try:
        volume = float(row["Avg. monthly searches"])
    except Exception:
        volume = 0.0

    # Defensive check: skip if keyword is empty after stripping
    if not keyword:
        logging.warning(f"Skipping empty keyword at index {idx}.")
        continue

    # Call SerpAPI (via utils.api_manager)
    serp_data = None
    try:
        # Assuming api_manager has a function `search_query` or similar to get SERP results.
        serp_data = api_manager.search_query(keyword, api_key=api_key)
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
    progress = (idx + 1) / num_keywords
    progress_bar.progress(min(progress, 1.0))

# Completed all queries - remove progress bar
progress_bar.empty()
logging.info(f"API querying complete. Success: {success_count}, No results: {no_result_count}, Failures: {fail_count}")

# -------------------------------------------
# Section: Debugging Summary
# -------------------------------------------
# Display a summary of query outcomes
st.subheader("Query Execution Summary")
st.write(f"**Keywords processed:** {len(results)}")
st.write(f"**Successful API fetches:** {success_count}")
st.write(f"**Keywords with no results:** {no_result_count}")
st.write(f"**API call failures:** {fail_count}")
if fail_count > 0:
    st.error(f"{fail_count} API queries failed. Please check your SerpAPI key or network connection.")
if success_count == 0:
    st.error("No successful results were obtained from the API. Unable to calculate visibility.")
    st.write("Please verify the CSV contents and API configuration, then try again.")
    st.expander("Debug Details").write("\n".join(debug_messages))
    st.stop()

# Optionally, provide detailed debug info per keyword in an expandable section
with st.expander("Detailed Debug Information per Keyword"):
    for msg in debug_messages:
        st.write(msg)

# -------------------------------------------
# Section: Visibility Calculation
# -------------------------------------------
# Define a CTR model for estimated click-through rates by rank (for visibility scoring)
CTR_MAP = {
    1: 0.30,   # Rank 1 ~30% CTR (estimated)
    2: 0.20,   # Rank 2 ~20%
    3: 0.10,   # Rank 3 ~10%
    4: 0.07,   # Rank 4 ~7%
    5: 0.05,   # Rank 5 ~5%
    6: 0.04,   # Rank 6 ~4%
    7: 0.03,   # Rank 7 ~3%
    8: 0.02,   # Rank 8 ~2%
    9: 0.01,   # Rank 9 ~1%
    10: 0.01   # Rank 10 ~1%
}
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

# -------------------------------------------
# Section: Results Output (Visibility Scores and Analysis)
# -------------------------------------------
st.subheader("SEO Visibility Results")

# Display overall visibility scores as metrics
if domain_filter and competitor_domain:
    colA, colB = st.columns(2)
    colA.metric(f"Overall Visibility ({domain_filter})", value=f"{overall_visibility_domain:.1f} %")
    colB.metric(f"Overall Visibility ({competitor_domain})", value=f"{overall_visibility_competitor:.1f} %")
elif domain_filter:
    st.metric(f"Overall Visibility ({domain_filter})", value=f"{overall_visibility_domain:.1f} %")
elif competitor_domain:
    st.metric(f"Overall Visibility ({competitor_domain})", value=f"{overall_visibility_competitor:.1f} %")

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

