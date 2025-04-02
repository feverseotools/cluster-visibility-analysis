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
# SERP API Queries Execution con manejo de errores mejorado
# -------------------------------------------
# Preparar la ejecución
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

    # Parámetros para la API: país, idioma y ubicación
    api_params = {
        'api_key': api_key,
        'country': search_country,
        'language': search_language
    }
    
    if search_location:
        api_params['location'] = search_location

    # Call SerpAPI (ahora usando nuestra implementación propia)
    serp_data = None
    
    # Check cache first if enabled
    using_cache = False
    if api_cache:
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
            # Call our implemented search function
            serp_data = search_query(keyword, 
                                     api_key=api_key, 
                                     country=search_country, 
                                     language=search_language, 
                                     location=search_location)
            
            # Save to cache if enabled
            if api_cache and serp_data:
                api_cache.set(keyword, serp_data, api_params)
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
        
        # Create an entry with null ranks for all domains
        result_entry = {
            "keyword": keyword,
            "cluster": cluster,
            "volume": volume
        }
        
        # Add ranks for all target domains (None as they weren't found)
        for domain in target_domains:
            result_entry[f"{domain}_rank"] = None
            
        # Add ranks for all competitor domains (None as they weren't found)
        for domain in competitor_domains:
            result_entry[f"{domain}_rank"] = None
            
        results.append(result_entry)
        
    else:
        # If serp_data is not None, determine if it's already parsed or needs parsing
        serp_results = []
        
        # Check for API errors in response
        if isinstance(serp_data, dict) and serp_data.get("error"):
            fail_count += 1
            err = serp_data.get("error")
            msg = f"Keyword '{keyword}': API error - {err}"
            debug_messages.append(msg)
            logging.error(f"API returned error for '{keyword}': {err}")
        else:
            # Try to extract organic results from different response formats
            if isinstance(serp_data, dict):
                # Formato SerpAPI típico
                serp_results = serp_data.get("organic_results", [])
                if not serp_results:
                    # Intentar formatos alternativos
                    serp_results = serp_data.get("results", [])
                    if not serp_results:
                        # Buscar en estructuras anidadas
                        for key, value in serp_data.items():
                            if isinstance(value, list) and key in ["results", "items", "listings"]:
                                serp_results = value
                                break
                
            elif isinstance(serp_data, list):
                # Si ya es una lista, usarla directamente
                serp_results = serp_data
                
        # Create a dictionary to store rankings for all domains
        domain_rankings = {}
        for domain in target_domains + competitor_domains:
            domain_rankings[domain] = None
            
        # Analyze results for all domains (targets and competitors)
        for res_index, res in enumerate(serp_results):
            # Extract URL from result based on structure
            url = ""
            if isinstance(res, str):
                url = res  # If API returns a list of URLs
            elif isinstance(res, dict):
                # Try various fields where the URL might be
                url = (res.get("link") or 
                       res.get("displayed_link") or 
                       res.get("url") or 
                       res.get("display_url") or 
                       res.get("domain") or "")
            else:
                continue
                
            if not url:
                continue
                
            # Extract netloc (domain) from URL
            try:
                netloc = urlparse(url).netloc.lower()
                if netloc.startswith("www."):
                    netloc = netloc[4:]
            except Exception:
                # En caso de error al parsear la URL
                netloc = url.lower()
                
            # Check if the result matches any of our tracked domains
            for domain in domain_rankings:
                if domain_rankings[domain] is None and (netloc.endswith(domain) or netloc == domain):
                    # Get position directly from result or use index+1
                    position = res.get("position", res.get("rank", res_index + 1))
                    domain_rankings[domain] = position

        # Create result entry with keyword info and all domain rankings
        result_entry = {
            "keyword": keyword,
            "cluster": cluster,
            "volume": volume
        }
        
        # Add rankings for all domains
        for domain, rank in domain_rankings.items():
            result_entry[f"{domain}_rank"] = rank
            
        # Add the result to our collection
        results.append(result_entry)
        
        # Log the outcome for this keyword
        if serp_results:
            success_count += 1
            # Create status message with rankings for all domains
            status_parts = [f"Keyword '{keyword}': {len(serp_results)} results retrieved."]
            
            for domain, rank in domain_rankings.items():
                domain_status = f"{domain}: " + (f"position {rank}" if rank else "not found")
                status_parts.append(domain_status)
                
            status_msg = " ".join(status_parts)
            debug_messages.append(status_msg)
            
            # Log the rankings
            log_domains = [f"{domain}={rank}" for domain, rank in domain_rankings.items()]
            logging.info(f"Results for '{keyword}': {', '.join(log_domains)}")
        else:
            # If serp_results is empty list (no organic results)
            no_result_count += 1
            msg = f"Keyword '{keyword}': No organic results returned."
            debug_messages.append(msg)
            logging.warning(f"No organic results for keyword '{keyword}'.")

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
# Modified: Visibility Calculation for Multiple Domains
# -------------------------------------------
# Get improved CTR model
CTR_MAP = get_improved_ctr_map()

# Initialize aggregation structures
cluster_stats = {}  # to store aggregated stats per cluster
total_volume_all = 0.0
total_captured_volume = {domain: 0.0 for domain in target_domains + competitor_domains}

# Aggregate results by cluster
for entry in results:
    cluster = entry["cluster"]
    vol = float(entry.get("volume", 0))
    total_volume_all += vol

    if cluster not in cluster_stats:
        cluster_stats[cluster] = {
            "total_volume": 0.0,
            "keywords_count": 0
        }
        # Initialize domain-specific stats for all domains
        for domain in target_domains + competitor_domains:
            cluster_stats[cluster][f"{domain}_captured_volume"] = 0.0
            cluster_stats[cluster][f"{domain}_keywords_ranked"] = 0
            
    cluster_stats[cluster]["total_volume"] += vol
    cluster_stats[cluster]["keywords_count"] += 1

    # Calculate captured volume for each domain based on its rank
    for domain in target_domains + competitor_domains:
        rank_key = f"{domain}_rank"
        rank = entry.get(rank_key)
        
        if rank is not None:
            # Calculate and add captured volume
            captured = vol * CTR_MAP.get(int(rank), 0)
            cluster_stats[cluster][f"{domain}_captured_volume"] += captured
            cluster_stats[cluster][f"{domain}_keywords_ranked"] += 1
            total_captured_volume[domain] += captured

# Compute overall visibility percentages for each domain
overall_visibility = {}
weighted_visibility = {}

for domain in target_domains + competitor_domains:
    # Basic visibility (percentage of total potential volume)
    if total_volume_all > 0:
        overall_visibility[domain] = (total_captured_volume[domain] / total_volume_all * 100)
    else:
        overall_visibility[domain] = 0.0
        
    # Calculate weighted visibility using improved method
    domain_results = []
    for r in results:
        # Create a compatible entry for the visibility calculation
        if r.get(f"{domain}_rank") is not None:
            entry = {
                "keyword": r["keyword"],
                "cluster": r["cluster"],
                "volume": r["volume"],
                "domain_rank": r[f"{domain}_rank"]
            }
            domain_results.append(entry)
    
    weighted_visibility[domain] = calculate_weighted_visibility(domain_results)

# -------------------------------------------
# Modified: Results Output for Multiple Domains
# -------------------------------------------
st.subheader("SEO Visibility Results")

# Display overall visibility scores as metrics in columns for better organization
st.write("### Visibility Scores")

# Function to create color gradient for comparing domains
def get_color(value, max_value):
    """Returns a color from red to green based on value relative to max."""
    if max_value == 0:
        return "lightgrey"  # Default if no scores
    
    # Normalize the value
    norm_value = value / max_value
    
    # Return color (green for highest, yellow for middle, red for lowest)
    if norm_value > 0.8:
        return "#28a745"  # Green for high values
    elif norm_value > 0.5:
        return "#ffc107"  # Yellow for medium values
    else:
        return "#dc3545"  # Red for low values

# Find max visibility to normalize colors
max_visibility = max(weighted_visibility.values()) if weighted_visibility else 1.0

# Display target domains first, then competitors
# Use columns for better space utilization with many domains
n_cols = min(3, len(target_domains) + len(competitor_domains))  # Max 3 columns to ensure readability
cols = st.columns(n_cols)

# Create container with tabs - one for targets and one for competitors
vis_tabs = st.tabs(["Target Domains", "Competitor Domains"])

# Target domains tab
with vis_tabs[0]:
    if target_domains:
        # Create columns within the tab
        t_cols = st.columns(min(3, len(target_domains)))
        
        for i, domain in enumerate(target_domains):
            col_idx = i % len(t_cols)
            with t_cols[col_idx]:
                st.metric(
                    label=f"{domain}",
                    value=f"{weighted_visibility.get(domain, 0):.1f}%",
                    delta=f"{(weighted_visibility.get(domain, 0) - overall_visibility.get(domain, 0)):.1f}%",
                    delta_color="normal"
                )
                st.write(f"Basic Visibility: {overall_visibility.get(domain, 0):.1f}%")
                
                # Add color indicator
                vis_color = get_color(weighted_visibility.get(domain, 0), max_visibility)
                st.markdown(f"<div style='background-color: {vis_color}; height: 5px; border-radius: 2px;'></div>", unsafe_allow_html=True)
    else:
        st.write("No target domains specified.")

# Competitor domains tab
with vis_tabs[1]:
    if competitor_domains:
        # Create columns within the tab
        c_cols = st.columns(min(3, len(competitor_domains)))
        
        for i, domain in enumerate(competitor_domains):
            col_idx = i % len(c_cols)
            with c_cols[col_idx]:
                st.metric(
                    label=f"{domain}",
                    value=f"{weighted_visibility.get(domain, 0):.1f}%",
                    delta=f"{(weighted_visibility.get(domain, 0) - overall_visibility.get(domain, 0)):.1f}%",
                    delta_color="normal"
                )
                st.write(f"Basic Visibility: {overall_visibility.get(domain, 0):.1f}%")
                
                # Add color indicator
                vis_color = get_color(weighted_visibility.get(domain, 0), max_visibility)
                st.markdown(f"<div style='background-color: {vis_color}; height: 5px; border-radius: 2px;'></div>", unsafe_allow_html=True)
    else:
        st.write("No competitor domains specified.")

# Prepare a DataFrame for cluster-level details
output_rows = []
for cluster_name, stats in sorted(cluster_stats.items()):
    row = {
        "Cluster": cluster_name,
        "Keywords": stats["keywords_count"],
        "Total Search Volume": int(stats["total_volume"])
    }
    
    # Add metrics for all target domains
    for domain in target_domains:
        # Visibility % per cluster for each target domain
        if stats["total_volume"] > 0:
            vis_pct = (stats[f"{domain}_captured_volume"] / stats["total_volume"] * 100)
        else:
            vis_pct = 0.0
        row[f"{domain} Vis%"] = round(vis_pct, 1)
        row[f"{domain} Keywords Ranked"] = stats[f"{domain}_keywords_ranked"]
    
    # Add metrics for all competitor domains
    for domain in competitor_domains:
        # Visibility % per cluster for each competitor domain
        if stats["total_volume"] > 0:
            vis_pct = (stats[f"{domain}_captured_volume"] / stats["total_volume"] * 100)
        else:
            vis_pct = 0.0
        row[f"{domain} Vis%"] = round(vis_pct, 1)
        row[f"{domain} Keywords Ranked"] = stats[f"{domain}_keywords_ranked"]
    
    output_rows.append(row)

# Display the cluster-level details table
if output_rows:
    st.subheader("Cluster Analysis")
    cluster_df = pd.DataFrame(output_rows)
    
    # Order columns logically: Cluster first, then Total, then domain metrics
    col_order = ["Cluster", "Keywords", "Total Search Volume"]
    
    # Add target domain columns
    for domain in target_domains:
        col_order += [f"{domain} Vis%", f"{domain} Keywords Ranked"]
    
    # Add competitor domain columns
    for domain in competitor_domains:
        col_order += [f"{domain} Vis%", f"{domain} Keywords Ranked"]
    
    # Only include columns that actually exist in the DataFrame
    col_order = [col for col in col_order if col in cluster_df.columns]
    
    cluster_df = cluster_df[col_order]
    st.dataframe(cluster_df)
else:
    st.write("No cluster data to display.")

# -------------------------------------------
# NUEVO: GPT Insights Integration
# -------------------------------------------
if use_gpt and gpt_api_key and target_domains and competitor_domains:
    st.subheader("GPT Insights")
    
    # Seleccionar un dominio target y un competidor para el análisis
    target_for_insights = target_domains[0] if target_domains else None
    competitor_for_insights = competitor_domains[0] if competitor_domains else None
    
    # Si hay múltiples dominios, permitir elegir
    if len(target_domains) > 1 or len(competitor_domains) > 1:
        col1, col2 = st.columns(2)
        with col1:
            target_for_insights = st.selectbox("Seleccionar dominio para análisis", 
                                              options=target_domains,
                                              index=0)
        with col2:
            competitor_for_insights = st.selectbox("Seleccionar competidor para análisis", 
                                                  options=competitor_domains,
                                                  index=0)
    
    # Mostrar un botón para generar los insights
    if st.button("Generar Insights con GPT"):
        with st.spinner("Generando insights con GPT..."):
            insights = generate_insights_with_gpt(
                target_for_insights, 
                competitor_for_insights, 
                results, 
                gpt_api_key
            )
            
            # Mostrar resultados en un contenedor especial
            st.info("Análisis generado por GPT")
            st.markdown(insights)

# -------------------------------------------
# Modified: Cross-Domain Opportunity Analysis
# -------------------------------------------
# Only perform if we have both target and competitor domains
if target_domains and competitor_domains:
    st.subheader("Cross-Domain Opportunity Analysis")
    
    # For each target domain
    for target in target_domains:
        # Analyze against all competitors
        target_opportunities = {}
        
        for competitor in competitor_domains:
            # Find keywords where competitor ranks but target doesn't
            opp_keywords = []
            opp_clusters = set()
            
            for entry in results:
                if entry.get(f"{target}_rank") is None and entry.get(f"{competitor}_rank") is not None:
                    opp_keywords.append(entry["keyword"])
                    opp_clusters.add(entry["cluster"])
            
            # Store the results
            if opp_keywords:
                target_opportunities[competitor] = {
                    "keywords": opp_keywords,
                    "clusters": sorted(opp_clusters),
                    "count": len(opp_keywords)
                }
        
        # Display opportunities for this target
        if target_opportunities:
            st.write(f"### Opportunities for {target}")
            
            # Create expandable sections for each competitor
            for competitor, opps in target_opportunities.items():
                with st.expander(f"vs {competitor} - {opps['count']} keywords"):
                    st.write(f"**{competitor}** is ranking for **{opps['count']}** keywords where **{target}** is not.")
                    st.write(f"**Clusters with opportunity:** {', '.join(opps['clusters'])}")
                    
                    # Show sample keywords
                    if len(opps['keywords']) > 10:
                        st.write(f"**Sample keywords:** {', '.join(opps['keywords'][:10])}...")
                    else:
                        st.write(f"**Keywords:** {', '.join(opps['keywords'])}")
                    
                    logging.info(f"{competitor} ranks in {opps['count']} keywords where {target} does not, across clusters: {opps['clusters']}")
        else:
            st.write(f"No opportunities found for {target} against the specified competitors.")

# -------------------------------------------
# Modified: Visualizations for Multiple Domains
# -------------------------------------------
def generate_multi_domain_visualizations(results, target_domains, competitor_domains):
    """Generate visualizations for SEO visibility analysis with multiple domains."""
    # Create DataFrame of results for analysis
    results_df = pd.DataFrame(results)
    
    # Only proceed if we have results to visualize
    if len(results_df) == 0:
        st.write("No data available for visualization.")
        return
    
    # 1. Ranking position distribution
    st.subheader("Ranking Position Distribution")
    
    # First tab for target domains, second for competitors
    dist_tabs = st.tabs(["Target Domains Distribution", "Competitor Domains Distribution"])
    
    # Target domains distribution
    with dist_tabs[0]:
        if target_domains:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create histograms for each target domain with different colors
            colors = ['blue', 'green', 'purple', 'cyan', 'magenta', 'yellow', 'black', 'orange']
            
            for i, domain in enumerate(target_domains):
                rank_col = f"{domain}_rank"
                domain_ranks = results_df[results_df[rank_col].notna()][rank_col]
                
                if not domain_ranks.empty:
                    color = colors[i % len(colors)]  # Cycle through colors
                    sns.histplot(domain_ranks, bins=list(range(1, 16)), alpha=0.6, 
                                label=domain, color=color, ax=ax)
            
            ax.set_title('Target Domains: SERP Position Distribution')
            ax.set_xlabel('Position')
            ax.set_ylabel('Number of keywords')
            ax.legend()
            
            st.pyplot(fig)
        else:
            st.write("No target domains to visualize.")
    
    # Competitor domains distribution
    with dist_tabs[1]:
        if competitor_domains:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create histograms for each competitor domain with different colors
            colors = ['red', 'orange', 'brown', 'pink', 'gray', 'olive', 'teal', 'navy']
            
            for i, domain in enumerate(competitor_domains):
                rank_col = f"{domain}_rank"
                domain_ranks = results_df[results_df[rank_col].notna()][rank_col]
                
                if not domain_ranks.empty:
                    color = colors[i % len(colors)]  # Cycle through colors
                    sns.histplot(domain_ranks, bins=list(range(1, 16)), alpha=0.6, 
                                label=domain, color=color, ax=ax)
            
            ax.set_title('Competitor Domains: SERP Position Distribution')
            ax.set_xlabel('Position')
            ax.set_ylabel('Number of keywords')
            ax.legend()
            
            st.pyplot(fig)
        else:
            st.write("No competitor domains to visualize.")
    
    # 2. Visibility by cluster chart
    if target_domains or competitor_domains:
        st.subheader("Visibility by Cluster")
        
        # Create tabs for different visualizations
        vis_tabs = st.tabs(["Target Domains", "Competitor Domains", "All Domains"])
        
        # Prepare data for charts
        cluster_vis_data = []
        
        for cluster, stats in cluster_stats.items():
            cluster_data = {'Cluster': cluster}
            
            # Add target domains visibility
            for domain in target_domains:
                if stats["total_volume"] > 0:
                    vis_pct = (stats[f"{domain}_captured_volume"] / stats["total_volume"] * 100)
                else:
                    vis_pct = 0.0
                cluster_data[domain] = round(vis_pct, 1)
            
            # Add competitor domains visibility
            for domain in competitor_domains:
                if stats["total_volume"] > 0:
                    vis_pct = (stats[f"{domain}_captured_volume"] / stats["total_volume"] * 100)
                else:
                    vis_pct = 0.0
                cluster_data[domain] = round(vis_pct, 1)
            
            cluster_vis_data.append(cluster_data)
        
        # Create DataFrame for visualization
        cluster_vis_df = pd.DataFrame(cluster_vis_data)
        
        # Sort by total visibility
        all_domains = target_domains + competitor_domains
        if all_domains:
            cluster_vis_df['Total_Vis'] = cluster_vis_df[all_domains].sum(axis=1)
            cluster_vis_df = cluster_vis_df.sort_values('Total_Vis', ascending=False)
        
        # Tab 1: Target Domains Visualization
        with vis_tabs[0]:
            if target_domains:
                # Create bar chart for target domains
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Configure data for chart
                clusters = cluster_vis_df['Cluster']
                x = range(len(clusters))
                width = 0.8 / len(target_domains) if len(target_domains) > 0 else 0.35
                
                # Bars for each target domain
                for i, domain in enumerate(target_domains):
                    if domain in cluster_vis_df.columns:
                        offset = width * i - (width * (len(target_domains) - 1) / 2)
                        ax.bar([pos + offset for pos in x], cluster_vis_df[domain], width, 
                              label=domain)
                
                # Configure labels and legend
                ax.set_ylabel('Visibility (%)')
                ax.set_title('Target Domains: SEO Visibility by Cluster')
                ax.set_xticks(x)
                ax.set_xticklabels(clusters, rotation=45, ha='right')
                ax.legend()
                
                # Adjust layout and display
                fig.tight_layout()
                st.pyplot(fig)
            else:
                st.write("No target domains to visualize.")
        
        # Tab 2: Competitor Domains Visualization
        with vis_tabs[1]:
            if competitor_domains:
                # Create bar chart for competitor domains
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Configure data for chart
                clusters = cluster_vis_df['Cluster']
                x = range(len(clusters))
                width = 0.8 / len(competitor_domains) if len(competitor_domains) > 0 else 0.35
                
                # Bars for each competitor domain
                for i, domain in enumerate(competitor_domains):
                    if domain in cluster_vis_df.columns:
                        offset = width * i - (width * (len(competitor_domains) - 1) / 2)
                        ax.bar([pos + offset for pos in x], cluster_vis_df[domain], width, 
                              label=domain)
                
                # Configure labels and legend
                ax.set_ylabel('Visibility (%)')
                ax.set_title('Competitor Domains: SEO Visibility by Cluster')
                ax.set_xticks(x)
                ax.set_xticklabels(clusters, rotation=45, ha='right')
                ax.legend()
                
                # Adjust layout and display
                fig.tight_layout()
                st.pyplot(fig)
            else:
                st.write("No competitor domains to visualize.")
        
        # Tab 3: All Domains Visualization
        with vis_tabs[2]:
            if target_domains or competitor_domains:
                # Create a comparative visualization all domains
                # Use a heatmap for better visualization when there are many domains
                
                # Prepare data for heatmap
                heatmap_data = cluster_vis_df.set_index('Cluster')[all_domains]
                
                # Create heatmap
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".1f", 
                           linewidths=.5, ax=ax)
                
                ax.set_title('All Domains: SEO Visibility Heatmap by Cluster')
                fig.tight_layout()
                st.pyplot(fig)
                
                # Also show a domain comparison bar chart
                st.write("### Overall Domain Visibility Comparison")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Extract overall visibility for all domains
                domains = []
                scores = []
                colors = []
                
                for domain in target_domains:
                    domains.append(domain)
                    scores.append(weighted_visibility.get(domain, 0))
                    colors.append('blue')  # Blue for target domains
                
                for domain in competitor_domains:
                    domains.append(domain)
                    scores.append(weighted_visibility.get(domain, 0))
                    colors.append('red')  # Red for competitors
                
                # Create bar chart
                bars = ax.bar(domains, scores, color=colors)
                
                # Add a line showing average visibility
                if scores:
                    avg_vis = sum(scores) / len(scores)
                    ax.axhline(y=avg_vis, color='gray', linestyle='--', 
                              label=f'Average: {avg_vis:.1f}%')
                
                # Add labels above bars
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.1f}%',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),  # 3 points vertical offset
                               textcoords="offset points",
                               ha='center', va='bottom')
                
                ax.set_title('Overall Domain Visibility Comparison')
                ax.set_ylabel('Weighted Visibility %')
                ax.set_ylim(0, max(scores) * 1.1 if scores else 10)  # Add some headroom
                
                # Add legend to differentiate target vs competitor
                if target_domains and competitor_domains:
                    from matplotlib.patches import Patch
                    legend_elements = [
                        Patch(facecolor='blue', label='Target Domains'),
                        Patch(facecolor='red', label='Competitor Domains')
                    ]
                    ax.legend(handles=legend_elements)
                
                fig.tight_layout()
                st.pyplot(fig)
            else:
                st.write("No domains to visualize.")

# Call visualization function if we have results
if success_count > 0 or cache_hits > 0:
    try:
        generate_multi_domain_visualizations(results, target_domains, competitor_domains)
    except Exception as vis_error:
        st.error(f"Error generating visualizations: {vis_error}")
        logging.error(f"Visualization error: {vis_error}")

# -------------------------------------------
# Keyword Intent Analysis - No major changes needed
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
# Modified: Cluster Correlation Analysis with Domain Selection
# -------------------------------------------
def analyze_cluster_correlations_multi(results, domains):
    """
    Analyze correlations between clusters based on ranking patterns for multiple domains.
    """
    # Let user select a domain to analyze correlations for
    if not domains:
        return {}
    
    # If we have domains to analyze
    domain_correlations = {}
    
    for domain in domains:
        # Create structure for analysis
        clusters = sorted(set(entry.get("cluster", "") for entry in results))
        cluster_rankings = {cluster: [] for cluster in clusters}
        
        # Collect rankings by cluster for this domain
        for entry in results:
            cluster = entry.get("cluster", "")
            rank = entry.get(f"{domain}_rank")
            
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
        
        # Store for this domain
        domain_correlations[domain] = {
            "correlations": correlations,
            "cluster_avg_rank": cluster_avg_rank
        }
    
    return domain_correlations

# Visualize cluster correlations for multiple domains
def visualize_cluster_correlations_multi(domain_correlations):
    st.subheader("Cluster Correlation Analysis")
    
    if not domain_correlations:
        st.write("No domains available for correlation analysis.")
        return
    
    # Create domain tabs for correlation analysis
    domain_tabs = st.tabs(list(domain_correlations.keys()))
    
    for i, domain in enumerate(domain_correlations.keys()):
        with domain_tabs[i]:
            correlations = domain_correlations[domain]["correlations"]
            cluster_avg_rank = domain_correlations[domain]["cluster_avg_rank"]
            
            if not correlations:
                st.write(f"No significant correlations found between clusters for {domain}.")
                continue
            
            # Show clusters with average ranking
            st.write(f"### Average Position by Cluster for {domain}")
            
            # Convert to DataFrame for better visualization
            avg_rank_data = [{"Cluster": cluster, "Average Position": round(rank, 2)} 
                             for cluster, rank in cluster_avg_rank.items()]
            avg_rank_df = pd.DataFrame(avg_rank_data).sort_values("Average Position")
            
            st.dataframe(avg_rank_df)
            
            # Show found correlations
            st.write(f"### Clusters with similar behavior for {domain}")
            
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
        # Combine target and competitor domains for analysis
        analysis_domains = target_domains + competitor_domains
        
        # Allow user to select which domains to analyze for correlation if there are many
        if len(analysis_domains) > 5:
            selected_domains = st.multiselect(
                "Select domains for correlation analysis (max 5 recommended)", 
                options=analysis_domains,
                default=analysis_domains[:2] if len(analysis_domains) >= 2 else analysis_domains
            )
        else:
            selected_domains = analysis_domains
        
        if selected_domains:
            # Analyze correlations
            domain_correlations = analyze_cluster_correlations_multi(results, selected_domains)
            # Visualize results
            visualize_cluster_correlations_multi(domain_correlations)
        else:
            st.write("Please select at least one domain for correlation analysis.")
    except Exception as corr_error:
        st.error(f"Error in correlation analysis: {corr_error}")
        logging.error(f"Correlation analysis error: {corr_error}")

# -------------------------------------------
# Modified: Optimization Suggestions for Multiple Domains
# -------------------------------------------
def generate_optimization_suggestions_multi(results, target_domains, competitor_domains):
    """
    Generate specific SEO optimization suggestions based on data analysis.
    Adapted for multiple domains.
    """
    if not target_domains or not competitor_domains:
        return {}
    
    # Dictionary to store suggestions for each target domain
    domain_suggestions = {}
    
    # Create DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # For each target domain
    for target in target_domains:
        suggestions = []
        
        # 1. Cross-competitor analysis - find keywords where all competitors rank but target doesn't
        universal_comp_keywords = []
        for entry in results:
            # Check if target doesn't rank
            if entry.get(f"{target}_rank") is None:
                # Check if all competitors rank
                all_comp_rank = True
                for comp in competitor_domains:
                    if entry.get(f"{comp}_rank") is None:
                        all_comp_rank = False
                        break
                
                if all_comp_rank:
                    universal_comp_keywords.append({
                        "keyword": entry["keyword"],
                        "cluster": entry["cluster"],
                        "volume": entry["volume"]
                    })
        
        if universal_comp_keywords:
            # Sort by volume
            universal_comp_keywords.sort(key=lambda x: float(x["volume"]), reverse=True)
            suggestions.append({
                "type": "universal_opportunity",
                "title": "Universal competitor advantage",
                "description": f"There are {len(universal_comp_keywords)} keywords where ALL competitors rank but {target} doesn't.",
                "keywords": [item["keyword"] for item in universal_comp_keywords[:10]],
                "clusters": list(set(item["cluster"] for item in universal_comp_keywords)),
                "action": "These represent critical content gaps that should be addressed with high priority."
            })
        
        # 2. For each competitor, find where they outrank target
        for competitor in competitor_domains:
            # Find keywords where competitor ranks better
            better_comp = []
            for entry in results:
                # If both domains rank, but competitor ranks higher
                target_rank = entry.get(f"{target}_rank")
                comp_rank = entry.get(f"{competitor}_rank")
                
                if target_rank is not None and comp_rank is not None and comp_rank < target_rank:
                    better_comp.append({
                        "keyword": entry["keyword"],
                        "cluster": entry["cluster"],
                        "volume": entry["volume"],
                        "target_rank": target_rank,
                        "comp_rank": comp_rank,
                        "rank_diff": target_rank - comp_rank
                    })
            
            if better_comp:
                # Sort by rank difference (largest first)
                better_comp.sort(key=lambda x: x["rank_diff"], reverse=True)
                suggestions.append({
                    "type": "ranking_gap",
                    "title": f"Ranking gap vs {competitor}",
                    "description": f"{competitor} ranks better than {target} for {len(better_comp)} keywords.",
                    "keywords": [f"{item['keyword']} (them: {item['comp_rank']}, you: {item['target_rank']})" for item in better_comp[:5]],
                    "action": "Analyze competitor content and improve your pages to close the ranking gap."
                })
            
            # Find keywords where competitor ranks but target doesn't
            comp_only = []
            for entry in results:
                # If competitor ranks but target doesn't
                if entry.get(f"{target}_rank") is None and entry.get(f"{competitor}_rank") is not None:
                    comp_only.append({
                        "keyword": entry["keyword"],
                        "cluster": entry["cluster"],
                        "volume": entry["volume"],
                        "comp_rank": entry.get(f"{competitor}_rank")
                    })
            
            if comp_only:
                # Sort by volume
                comp_only.sort(key=lambda x: float(x["volume"]), reverse=True)
                suggestions.append({
                    "type": "coverage_gap",
                    "title": f"Coverage gap vs {competitor}",
                    "description": f"{competitor} ranks for {len(comp_only)} keywords where {target} doesn't appear.",
                    "keywords": [f"{item['keyword']} (them: {item['comp_rank']})" for item in comp_only[:5]],
                    "clusters": list(set(item["cluster"] for item in comp_only)),
                    "action": "Create new content targeting these keywords to close coverage gaps."
                })
        
        # 3. Cluster-level analysis - identify clusters where target has low visibility
        if cluster_stats:
            low_vis_clusters = []
            for cluster, stats in cluster_stats.items():
                if stats["total_volume"] > 0:
                    target_vis = stats.get(f"{target}_captured_volume", 0) / stats["total_volume"] * 100
                    
                    # Calculate average competitor visibility for this cluster
                    comp_vis_values = []
                    for comp in competitor_domains:
                        if f"{comp}_captured_volume" in stats:
                            comp_vis = stats[f"{comp}_captured_volume"] / stats["total_volume"] * 100
                            comp_vis_values.append(comp_vis)
                    
                    avg_comp_vis = sum(comp_vis_values) / len(comp_vis_values) if comp_vis_values else 0
                    
                    # If target visibility is significantly lower than average competitor
                    if target_vis < avg_comp_vis * 0.7:  # Target is below 70% of average competitor
                        low_vis_clusters.append({
                            "cluster": cluster,
                            "target_vis": target_vis,
                            "avg_comp_vis": avg_comp_vis,
                            "vis_gap": avg_comp_vis - target_vis,
                            "volume": stats["total_volume"]
                        })
            
            if low_vis_clusters:
                # Sort by visibility gap
                low_vis_clusters.sort(key=lambda x: x["vis_gap"], reverse=True)
                suggestions.append({
                    "type": "cluster_weakness",
                    "title": "Underperforming clusters",
                    "description": f"There are {len(low_vis_clusters)} clusters where {target} significantly underperforms compared to competitors.",
                    "clusters": [f"{item['cluster']} (you: {item['target_vis']:.1f}%, competitors avg: {item['avg_comp_vis']:.1f}%)" for item in low_vis_clusters[:5]],
                    "action": "Focus optimization efforts on these clusters to improve overall visibility."
                })
        
        # Store suggestions for this target domain
        domain_suggestions[target] = suggestions
    
    return domain_suggestions

# Display optimization suggestions for multiple domains
def display_optimization_suggestions_multi(domain_suggestions):
    st.subheader("SEO Optimization Suggestions")
    
    if not domain_suggestions:
        st.write("There is not enough comparative data to generate specific suggestions.")
        return
    
    # Create tabs for each target domain
    domain_tabs = st.tabs(list(domain_suggestions.keys()))
    
    for i, domain in enumerate(domain_suggestions.keys()):
        with domain_tabs[i]:
            suggestions = domain_suggestions[domain]
            
            if not suggestions:
                st.write(f"No specific suggestions available for {domain}.")
                continue
            
            st.write(f"### Optimization suggestions for {domain}")
            
            # Display each suggestion in an expandable section
            for j, suggestion in enumerate(suggestions):
                with st.expander(f"💡 {suggestion['title']}", expanded=j == 0):  # Expand first suggestion by default
                    st.write(f"**Description:** {suggestion['description']}")
                    
                    # Show keywords if they exist
                    if 'keywords' in suggestion and suggestion['keywords']:
                        st.write("**Key keywords:**")
                        for kw in suggestion['keywords']:
                            st.write(f"- {kw}")
                    
                    # Show clusters if they exist
                    if 'clusters' in suggestion and suggestion['clusters']:
                        st.write("**Key clusters:**")
                        for cluster in suggestion['clusters']:
                            st.write(f"- {cluster}")
                    
                    # Show recommended action
                    st.write(f"**Recommended action:** {suggestion['action']}")
                    
                    # Add specific tips based on suggestion type
                    if suggestion['type'] == 'universal_opportunity':
                        st.write("""
                        **Additional tips:**
                        - These keywords represent a critical competitive gap
                        - Analyze ALL competitor content ranking for these terms
                        - Create comprehensive content that covers all aspects
                        - Consider creating interactive tools or unique resources
                        """)
                    elif suggestion['type'] == 'ranking_gap':
                        st.write("""
                        **Additional tips:**
                        - Compare content depth and quality with competitor
                        - Check technical factors (site speed, mobile experience)
                        - Analyze backlink profile differences
                        - Improve internal linking to these pages
                        """)
                    elif suggestion['type'] == 'coverage_gap':
                        st.write("""
                        **Additional tips:**
                        - Research top-ranking pages thoroughly
                        - Identify content formats that perform well
                        - Create content that addresses user intent more comprehensively
                        - Support new content with internal links from relevant pages
                        """)
                    elif suggestion['type'] == 'cluster_weakness':
                        st.write("""
                        **Additional tips:**
                        - Perform a topical gap analysis for these clusters
                        - Consider publishing pillar content and supporting articles
                        - Develop a cluster-specific internal linking strategy
                        - Evaluate keyword cannibalization issues within these clusters
                        """)

# Call optimization suggestions if we have comparative results
if (success_count > 0 or cache_hits > 0) and target_domains and competitor_domains:
    try:
        # Generate suggestions
        domain_suggestions = generate_optimization_suggestions_multi(results, target_domains, competitor_domains)
        # Display suggestions
        display_optimization_suggestions_multi(domain_suggestions)
    except Exception as sugg_error:
        st.error(f"Error generating suggestions: {sugg_error}")
        logging.error(f"Suggestion generation error: {sugg_error}")

# -------------------------------------------
# Modified: Export Results Section for Multiple Domains
# -------------------------------------------
# Create a DataFrame with all detailed results
full_results_df = pd.DataFrame(results)

# Add download button for full results
if not full_results_df.empty:
    st.subheader("Export Results")
    
    # Convert DataFrame to CSV for download
    csv = full_results_df.to_csv(index=False)
    
    # Download button for complete results
    st.download_button(
        label="Download complete results (CSV)",
        data=csv,
        file_name=f"seo_visibility_results_multi_domain.csv",
        mime="text/csv"
    )
    
    # Option to download cluster analysis
    if 'cluster_df' in locals() and not cluster_df.empty:
        csv_clusters = cluster_df.to_csv(index=False)
        st.download_button(
            label="Download cluster analysis (CSV)",
            data=csv_clusters,
            file_name=f"seo_cluster_analysis_multi_domain.csv",
            mime="text/csv"
        )
    
    # Export domain comparison summary
    if weighted_visibility:
        # Create a domain comparison summary
        domain_summary = []
        for domain in sorted(target_domains + competitor_domains):
            domain_rank_col = f"{domain}_rank"
            
            # Count keywords where this domain ranks
            keywords_ranked = sum(1 for entry in results if entry.get(domain_rank_col) is not None)
            
            # Calculate average position for the domain
            positions = [entry.get(domain_rank_col) for entry in results if entry.get(domain_rank_col) is not None]
            avg_position = sum(positions) / len(positions) if positions else None
            
            # Count keywords in top positions
            top3 = sum(1 for entry in results if entry.get(domain_rank_col) is not None and entry.get(domain_rank_col) <= 3)
            top10 = sum(1 for entry in results if entry.get(domain_rank_col) is not None and entry.get(domain_rank_col) <= 10)
            
            # Domain type
            domain_type = "Target" if domain in target_domains else "Competitor"
            
            domain_summary.append({
                "Domain": domain,
                "Type": domain_type,
                "Visibility Score": weighted_visibility.get(domain, 0),
                "Keywords Ranked": keywords_ranked,
                "Average Position": round(avg_position, 2) if avg_position else None,
                "Top 3 Rankings": top3,
                "Top 10 Rankings": top10,
                "Keywords Analyzed": len(results)
            })
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(domain_summary)
        
        # Export summary
        csv_summary = summary_df.to_csv(index=False)
        st.download_button(
            label="Download domain comparison summary (CSV)",
            data=csv_summary,
            file_name=f"seo_domain_comparison_summary.csv",
            mime="text/csv"
        )

# Add app footer with version info
st.markdown("""
---
### SEO Visibility Estimator v3.5 (Multiple Domains)
An improved tool for analyzing SEO visibility across multiple domains and keyword clusters.

**Features:**
- Support for multiple target and competitor domains
- Enhanced visualizations with domain comparison
- Advanced cross-domain opportunity analysis
- Domain-specific optimization suggestions
- Search location, country and language configuration
- GPT integration for advanced insights
- Cluster correlation analysis by domain
- Search intent analysis
- Flexible CSV handling with intelligent column detection
- API caching system for efficient SerpAPI usage
""")

# Log app completion
logging.info("Multi-domain SEO app execution completed successfully.")# -------------------------------------------
# SEO Visibility Estimator - Multiple Domains Support
# Con solución de errores y configuraciones extendidas
# -------------------------------------------
import streamlit as st
import pandas as pd
import logging
import os
import json
import hashlib
import requests
from datetime import datetime, timedelta
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging for debugging
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)

# -------------------------------------------
# Implementación propia de la función de búsqueda
# -------------------------------------------
def search_query(keyword, api_key, country='us', language='en', location=None):
    """
    Implementación directa de la llamada a SerpAPI para evitar 
    dependencia del módulo api_manager ausente.
    """
    try:
        base_url = "https://serpapi.com/search"
        
        # Parámetros base para la consulta
        params = {
            "q": keyword,
            "api_key": api_key,
            "engine": "google",
            "gl": country,          # Código de país
            "hl": language,         # Código de idioma
        }
        
        # Añadir ubicación si se especifica
        if location:
            params["location"] = location
        
        # Realizar la solicitud a la API
        response = requests.get(base_url, params=params)
        
        # Comprobar si la solicitud fue exitosa
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Error de API: {response.status_code} - {response.text}"}
            
    except Exception as e:
        logging.error(f"Error en consulta SerpAPI: {e}")
        return {"error": str(e)}

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
# GPT Integration para Insights
# -------------------------------------------
def generate_insights_with_gpt(domain, competitor_domain, results, api_key):
    """
    Genera insights utilizando GPT basados en los resultados del análisis SEO.
    
    Args:
        domain: Dominio principal
        competitor_domain: Dominio competidor
        results: Resultados del análisis
        api_key: API key para GPT
    
    Returns:
        Texto con insights generados por GPT
    """
    if not api_key:
        return "Se requiere API key de OpenAI para generar insights."
    
    try:
        # Crear un resumen del análisis para enviarlo a GPT
        summary = {
            "target_domain": domain,
            "competitor_domain": competitor_domain,
            "total_keywords": len(results),
            "ranked_keywords": sum(1 for r in results if r.get(f"{domain}_rank") is not None),
            "competitor_ranked": sum(1 for r in results if r.get(f"{competitor_domain}_rank") is not None),
            "better_positions": sum(1 for r in results if r.get(f"{domain}_rank") is not None and 
                                r.get(f"{competitor_domain}_rank") is not None and 
                                r.get(f"{domain}_rank") < r.get(f"{competitor_domain}_rank")),
            "worse_positions": sum(1 for r in results if r.get(f"{domain}_rank") is not None and 
                                r.get(f"{competitor_domain}_rank") is not None and 
                                r.get(f"{domain}_rank") > r.get(f"{competitor_domain}_rank"))
        }
        
        # Seleccionar algunas palabras clave de ejemplo donde el competidor es mejor
        sample_keywords = []
        for r in results:
            if r.get(f"{domain}_rank") is not None and r.get(f"{competitor_domain}_rank") is not None and r.get(f"{domain}_rank") > r.get(f"{competitor_domain}_rank"):
                sample_keywords.append({
                    "keyword": r["keyword"],
                    "target_position": r.get(f"{domain}_rank"),
                    "competitor_position": r.get(f"{competitor_domain}_rank"),
                    "cluster": r["cluster"]
                })
                if len(sample_keywords) >= 5:
                    break
        
        # Preparar el mensaje para GPT
        prompt = f"""
        Analiza los siguientes datos de posicionamiento SEO de "{domain}" frente a su competidor "{competitor_domain}" y proporciona insights estratégicos:
        
        RESUMEN:
        - Total de keywords analizadas: {summary['total_keywords']}
        - Keywords donde {domain} aparece: {summary['ranked_keywords']}
        - Keywords donde {competitor_domain} aparece: {summary['competitor_ranked']}
        - Keywords donde {domain} tiene mejor posición: {summary['better_positions']}
        - Keywords donde {domain} tiene peor posición: {summary['worse_positions']}
        
        EJEMPLOS DE KEYWORDS DONDE EL COMPETIDOR ES MEJOR:
        {json.dumps(sample_keywords, indent=2)}
        
        Por favor proporciona:
        1. Un análisis de la situación competitiva general
        2. Las posibles causas de las diferencias de rendimiento
        3. Tres recomendaciones estratégicas específicas para mejorar
        4. Enfoque en qué oportunidades aprovechar y qué amenazas abordar
        
        Limita tu respuesta a 4-5 párrafos concisos e informativos.
        """
        
        # Realizar la llamada a la API de OpenAI
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "model": "gpt-4-turbo",
            "messages": [
                {"role": "system", "content": "Eres un experto en SEO que proporciona análisis competitivo y recomendaciones estratégicas."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            insights = result["choices"][0]["message"]["content"]
            return insights
        else:
            logging.error(f"Error en llamada a GPT: {response.status_code} - {response.text}")
            return f"Error al generar insights: {response.status_code} - {response.text}"
            
    except Exception as e:
        logging.error(f"Error al generar insights con GPT: {e}")
        return f"Error al generar insights: {str(e)}"

# -------------------------------------------
# Streamlit App Title and Description
# -------------------------------------------
st.title("SEO Visibility Estimator - Multiple Domains")
st.markdown("""
Upload a CSV of keywords (with **keyword**, **cluster_name**, **Avg. monthly searches** columns) and configure options to estimate SEO visibility.
This tool will calculate a visibility score for multiple target domains and competitors across keyword clusters by querying Google SERPs via SerpAPI.
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

# -------------------------------------------
# MODIFICADO: Configuraciones de búsqueda SerpAPI
# -------------------------------------------
st.subheader("Configuración de búsqueda SerpAPI")

# Selección de país para la búsqueda
countries = {
    'us': 'Estados Unidos',
    'es': 'España',
    'uk': 'Reino Unido',
    'fr': 'Francia',
    'de': 'Alemania',
    'it': 'Italia',
    'mx': 'México',
    'ar': 'Argentina',
    'co': 'Colombia',
    'cl': 'Chile',
    'pe': 'Perú',
    'br': 'Brasil',
    'ca': 'Canadá',
    'au': 'Australia',
    'in': 'India',
    'jp': 'Japón'
}

search_country = st.selectbox(
    "País de búsqueda", 
    options=list(countries.keys()),
    format_func=lambda x: f"{countries[x]} ({x})",
    index=0
)

# Selección de idioma para la búsqueda
languages = {
    'en': 'Inglés',
    'es': 'Español',
    'fr': 'Francés',
    'de': 'Alemán',
    'it': 'Italiano',
    'pt': 'Portugués',
    'ja': 'Japonés',
    'ru': 'Ruso',
    'ar': 'Árabe',
    'zh': 'Chino'
}

search_language = st.selectbox(
    "Idioma de búsqueda", 
    options=list(languages.keys()),
    format_func=lambda x: f"{languages[x]} ({x})",
    index=0
)

# Ubicación específica (opcional)
search_location = st.text_input(
    "Ubicación específica (opcional)",
    help="Especifica una ciudad o ubicación (ej: 'New York, New York, United States')"
)

# -------------------------------------------
# MODIFICADO: Input Multiple Target and Competitor Domains
# -------------------------------------------
st.subheader("Domain Configuration")

# Target domains input
target_domains_input = st.text_area(
    "Target Domains (one per line)",
    help="Enter one or more target domains, each on a new line (e.g. example.com)"
)

# Competitor domains input
competitor_domains_input = st.text_area(
    "Competitor Domains (one per line)",
    help="Enter one or more competitor domains, each on a new line (e.g. competitor.com)"
)

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

# Process multiple domains
def process_multi_domains(domains_text):
    """Process multiple domains from a text area input."""
    if not domains_text:
        return []
    
    # Split by newline and process each domain
    domains = []
    for domain in domains_text.split('\n'):
        cleaned = clean_domain(domain)
        if cleaned:
            domains.append(cleaned)
    
    return domains

# Process domains
target_domains = process_multi_domains(target_domains_input)
competitor_domains = process_multi_domains(competitor_domains_input)

# Display processed domains for confirmation
if target_domains:
    st.write(f"**Target domains processed:** {', '.join(target_domains)}")
else:
    st.warning("No valid target domains provided.")
    
if competitor_domains:
    st.write(f"**Competitor domains processed:** {', '.join(competitor_domains)}")
else:
    st.warning("No valid competitor domains provided.")

# Ensure at least one domain is provided for visibility analysis
if not target_domains and not competitor_domains:
    st.warning("No domains provided for analysis. Please enter at least one target domain or competitor domain.")
    logging.info("No domain or competitor provided. Stopping execution as there's nothing to analyze.")
    st.stop()

# 3. SerpAPI API key input
api_key = st.text_input("SerpAPI API Key", type="password", help="Your SerpAPI key is required to fetch Google results.")
if not api_key:
    st.warning("Please provide a SerpAPI API key to run the analysis.")
    logging.warning("No SerpAPI API key provided by user.")
    st.stop()

# -------------------------------------------
# NUEVO: Integración con GPT para insights
# -------------------------------------------
st.subheader("Integración con GPT (opcional)")

use_gpt = st.checkbox("Utilizar GPT para generar insights", value=False)

if use_gpt:
    gpt_api_key = st.text_input("OpenAI API Key", type="password", help="Tu API key de OpenAI para generar insights")
    if not gpt_api_key:
        st.warning("Se requiere una API key de OpenAI para utilizar GPT. Los insights no estarán disponibles.")

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
        
        # Show breakdown
