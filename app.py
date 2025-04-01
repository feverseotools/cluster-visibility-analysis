import streamlit as st
import pandas as pd
import plotly.express as px
from serpapi import GoogleSearch
import time
import os
import json
from datetime import datetime
import concurrent.futures
from io import BytesIO
import re
from jinja2 import Template
import math

# Page configuration
st.set_page_config(page_title='SEO Visibility Estimator Pro', layout='wide')

# Caching functions
@st.cache_data(ttl=86400)  # Cache for 24 hours
def fetch_serp_results(keyword, params):
    """Cache search results to avoid duplicate API calls"""
    search = GoogleSearch(params)
    return search.get_dict()

@st.cache_data(ttl=3600)  # Cache for 1 hour
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
    """Return estimated CTR by position"""
    ctr_model = {
        1: 0.316,  # 31.6% CTR for position 1
        2: 0.158,  # 15.8% CTR for position 2
        3: 0.096,  # 9.6% CTR for position 3
        4: 0.072,  # 7.2% CTR for position 4
        5: 0.0596, # 5.96% CTR for position 5
        6: 0.0454, # 4.54% CTR for position 6
        7: 0.0379, # 3.79% CTR for position 7
        8: 0.0312, # 3.12% CTR for position 8
        9: 0.0278, # 2.78% CTR for position 9
        10: 0.0236, # 2.36% CTR for position 10
    }
    return ctr_model.get(position, 0.01)  # Default 1% for positions > 10

# Cost calculation functions
def calculate_api_cost(num_keywords, api_plan="basic"):
    """Calculate estimated SerpAPI cost based on number of queries"""
    # SerpAPI pricing tiers (as of March 2025)
    pricing = {
        "basic": {"monthly_cost": 50, "searches": 5000},
        "business": {"monthly_cost": 250, "searches": 30000},
        "enterprise": {"monthly_cost": 500, "searches": 70000}
    }
    
    # Select pricing tier
    plan = pricing.get(api_plan.lower(), pricing["basic"])
    
    # Calculate cost per search
    cost_per_search = plan["monthly_cost"] / plan["searches"]
    
    # Calculate estimated cost
    estimated_cost = num_keywords * cost_per_search
    
    # Calculate percentage of monthly quota
    quota_percentage = (num_keywords / plan["searches"]) * 100
    
    return {
        "num_queries": num_keywords,
        "estimated_cost": round(estimated_cost, 2),
        "quota_percentage": round(quota_percentage, 2),
        "plan_details": f"{plan['monthly_cost']}$ for {plan['searches']} searches"
    }

def cluster_representative_keywords(keywords_df, max_representatives=100):
    """
    Select representative keywords from each cluster to reduce API costs
    This returns a subset of keywords that represent each cluster
    """
    if keywords_df.empty or 'cluster_name' not in keywords_df.columns:
        return keywords_df
    
    # Group by cluster
    grouped = keywords_df.groupby('cluster_name')
    
    # Calculate how many representatives to take from each cluster
    # Use proportional distribution based on cluster size and search volume
    cluster_sizes = grouped.size()
    cluster_volumes = grouped['Avg. monthly searches'].sum()
    
    # Create a score combining size and volume
    cluster_importance = (cluster_sizes / cluster_sizes.sum() + 
                         cluster_volumes / cluster_volumes.sum()) / 2
    
    # Calculate representatives per cluster (minimum 1)
    reps_per_cluster = (cluster_importance * max_representatives).apply(lambda x: max(1, round(x)))
    
    # Ensure we don't exceed max_representatives
    while reps_per_cluster.sum() > max_representatives:
        # Find cluster with most representatives
        max_cluster = reps_per_cluster.idxmax()
        # Reduce by 1
        reps_per_cluster[max_cluster] -= 1
    
    # Select representatives
    selected_keywords = []
    
    for cluster, count in reps_per_cluster.items():
        # Get cluster data
        cluster_data = keywords_df[keywords_df['cluster_name'] == cluster]
        
        # Sort by search volume (descending)
        sorted_data = cluster_data.sort_values('Avg. monthly searches', ascending=False)
        
        # Take top keywords as representatives
        selected_keywords.append(sorted_data.head(int(count)))
    
    # Combine all selected keywords
    representative_df = pd.concat(selected_keywords)
    
    return representative_df

def display_cost_calculator(keywords_df):
    """Display interactive cost calculator in Streamlit"""
    st.subheader("API Cost Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # API plan selection
        api_plan = st.selectbox(
            "SerpAPI Plan",
            options=["Basic", "Business", "Enterprise"],
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
        
        if use_representatives:
            max_keywords = st.slider(
                "Maximum number of representative keywords", 
                min_value=10, 
                max_value=min(500, len(keywords_df)), 
                value=min(100, len(keywords_df) // 2)
            )
        else:
            max_keywords = len(keywords_df)
    
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
        sample_df = cluster_representative_keywords(keywords_df, max_keywords)
        num_queries = len(sample_df)
        sampling_ratio = num_queries / len(keywords_df)
    else:
        num_queries = len(keywords_df)
        sampling_ratio = 1.0
    
    # Calculate total cost
    total_queries = num_queries * multiplier
    cost_data = calculate_api_cost(total_queries, api_plan.lower())
    
    # Display results
    st.subheader("Cost Estimate")
    
    cost_col1, cost_col2 = st.columns(2)
    
    with cost_col1:
        st.metric("Total Keywords", f"{len(keywords_df):,}")
        st.metric("Keywords Analyzed", f"{num_queries:,}")
        st.metric("Sampling Rate", f"{sampling_ratio:.1%}")
    
    with cost_col2:
        st.metric("Total API Queries", f"{total_queries:,}")
        st.metric("Estimated Cost", f"${cost_data['estimated_cost']:.2f}")
        st.metric("Monthly Quota Used", f"{cost_data['quota_percentage']:.1f}%")
    
    # Return the representative keywords if checkbox is selected
    if use_representatives:
        return sample_df, cost_data
    else:
        return keywords_df, cost_data

# Data analysis functions
def process_keywords_in_batches(keywords_df, domains, params_template, batch_size=5):
    """Process keywords in batches to improve performance and respect API limits"""
    all_results = []
    progress_bar = st.progress(0)
    
    # Process in batches to respect API limits
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
                except Exception as e:
                    st.error(f"Error processing {row['keyword']}: {str(e)}")
        
        # Update progress
        progress_bar.progress((i + batch_size) / len(keywords_df))
        
        # Pause between batches to respect API limits
        if i + batch_size < len(keywords_df):
            time.sleep(2)
    
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

def analyze_competitors(results_df, keywords_df, domains, params_template):
    """Analyze competitors in the same search terms"""
    if keywords_df.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    # Extract competitors from SERP results
    all_competitors = {}
    keyword_count = {}
    progress_bar = st.progress(0)
    
    # Use a subset of keywords for competitive analysis
    analysis_sample = keywords_df.sort_values('Avg. monthly searches', ascending=False).head(50)
    
    for i, row in enumerate(analysis_sample.iterrows()):
        _, row_data = row
        params = params_template.copy()
        params["q"] = row_data['keyword']
        params["num"] = 20  # Get more results for better competitive analysis
        
        try:
            results = fetch_serp_results(row_data['keyword'], params)
            organic_results = results.get("organic_results", [])
            
            # Record all domains in the top 20 results
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
    
    # Keywords not ranking for the analyzed domains
    non_ranking_keywords = keywords_df[~keywords_df['keyword'].isin(ranking_keywords)]
    
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
    """Save results for historical tracking"""
    if results_df.empty:
        return None
        
    # Create directory if it doesn't exist
    if not os.path.exists('historical_data'):
        os.makedirs('historical_data')
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"historical_data/seo_data_{timestamp}.json"
    
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
    
    # Save as JSON
    with open(filename, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return filename

def load_historical_data():
    """Load all available historical data"""
    if not os.path.exists('historical_data'):
        return []
    
    historical_data = []
    for filename in os.listdir('historical_data'):
        if filename.startswith('seo_data_') and filename.endswith('.json'):
            with open(os.path.join('historical_data', filename), 'r') as f:
                try:
                    data = json.load(f)
                    # Extract date from filename
                    date_str = filename.replace('seo_data_', '').replace('.json', '')
                    date = datetime.strptime(date_str.split('_')[0], "%Y%m%d")
                    
                    # Add date and file to data
                    data['_meta'] = {
                        'date': date.strftime("%Y-%m-%d"),
                        'filename': filename
                    }
                    
                    historical_data.append(data)
                except:
                    continue
    
    # Sort by date
    historical_data.sort(key=lambda x: x['_meta']['date'])
    return historical_data

# ChatGPT integration
class ChatGPTAnalyzer:
    """Class to handle ChatGPT API interactions for SEO analysis"""
    
    def __init__(self, api_key):
        """Initialize with OpenAI API key"""
        self.api_key = api_key
        openai.api_key = api_key
    
    def analyze_serp_competitors(self, keyword, competitors):
        """
        Analyze SERP competitors for a specific keyword
        Returns insights on content strategy based on top-ranking pages
        """
        if not self.api_key or not competitors:
            return "API key or competitor data missing"
        
        # Format competitor data
        competitor_text = "\n".join([
            f"{i+1}. {comp['Domain']} (Position: {comp['Rank']})"
            for i, comp in enumerate(competitors[:5])  # Top 5 competitors
        ])
        
        # Prompt for GPT
        prompt = f"""
        Analyze these top 5 competitors for the keyword "{keyword}":
        
        {competitor_text}
        
        Based on these ranking pages, provide insights on:
        1. Key content elements to include
        2. Approximate content length
        3. Content structure recommendations
        4. Unique angle to differentiate
        
        Format your response as bullet points for each section.
        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an SEO content strategy expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate_cluster_content_brief(self, cluster_name, keywords, competition_level="medium"):
        """
        Generate a content brief for a keyword cluster
        Provides recommendations for content that could rank for multiple keywords
        """
        if not self.api_key:
            return "API key missing"
        
        # Format keywords with volumes
        keywords_text = "\n".join([
            f"- {kw['Keyword']} ({kw['Search_Volume']} searches/month)"
            for kw in keywords[:10]  # Top 10 keywords by volume
        ])
        
        # Prompt for GPT
        prompt = f"""
        Create a content brief for the topic cluster: "{cluster_name}"
        
        Top keywords in this cluster:
        {keywords_text}
        
        Competition level: {competition_level}
        
        Provide recommendations for:
        1. Suggested title (include the main keyword)
        2. Meta description
        3. Content outline (H2s and H3s)
        4. Key points to cover
        5. Types of media to include
        6. Word count range
        
        Focus on creating comprehensive content that could rank for multiple keywords in this cluster.
        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an SEO content strategist who creates detailed content briefs."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def analyze_opportunity_gaps(self, domain, competitors, cluster_data):
        """
        Analyze gaps and opportunities compared to competitors
        Provides strategic recommendations for improvements
        """
        if not self.api_key:
            return "API key missing"
        
        # Format competitor data
        competitor_text = "\n".join([
            f"- {comp['Domain']} (Visibility: {comp['Total_Visibility']}, Keywords: {comp['Keyword_Count']})"
            for comp in competitors[:3]  # Top 3 competitors
        ])
        
        # Format cluster performance
        cluster_text = "\n".join([
            f"- {cluster}: Keywords: {data['Keywords_Count']}, Avg Position: {data['Average_Position']:.1f}"
            for cluster, data in cluster_data.items()
        ])
        
        # Prompt for GPT
        prompt = f"""
        Analyze SEO opportunities for domain: {domain}
        
        Top competitors:
        {competitor_text}
        
        Current performance by cluster:
        {cluster_text}
        
        Provide strategic recommendations:
        1. Key opportunity areas based on competitor analysis
        2. Specific clusters to prioritize
        3. Content improvement suggestions
        4. Technical SEO considerations
        5. Quick wins vs long-term strategies
        
        Focus on actionable insights.
        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4",  # Using GPT-4 for more strategic analysis
                messages=[
                    {"role": "system", "content": "You are an SEO strategy consultant with expertise in competitive analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=1200
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

# Integration with Streamlit UI
def display_chatgpt_features(openai_api_key, results_df, competitor_df, domain_metrics, opportunity_df):
    """Display ChatGPT-powered analysis features in Streamlit"""
    st.header("AI-Powered SEO Insights")
    
    if not openai_api_key:
        st.warning("Enter your OpenAI API key in the sidebar to unlock AI-powered insights")
        return
    
    # Initialize ChatGPT analyzer
    analyzer = ChatGPTAnalyzer(openai_api_key)
    
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

# Main function
def main():
    # Application title
    st.title('🔍 SEO Visibility Estimator Pro')
    st.markdown("*Advanced visibility analysis by semantic clusters*")
    
    # Sidebar with configuration options
    st.sidebar.header('Configuration')
    
    # File and domain inputs
    uploaded_file = st.sidebar.file_uploader('Upload Keywords CSV', type=['csv'])
    domains_input = st.sidebar.text_area('Domains (comma-separated)', 'example.com, example2.com')
    
    # API credentials
    serp_api_key = st.sidebar.text_input('SerpAPI Key', type='password')
    openai_api_key = st.sidebar.text_input('OpenAI API Key (optional)', type='password',
                                         help="Required for AI-powered SEO insights")
    
    # Location configuration
    country_code = st.sidebar.selectbox('Country', 
                options=['us', 'es', 'mx', 'ar', 'co', 'pe', 'cl', 'uk', 'ca', 'fr', 'de', 'it'], 
                index=0)
    language = st.sidebar.selectbox('Language', 
                options=['en', 'es', 'fr', 'de', 'it'], 
                index=0)
    city = st.sidebar.text_input('City (optional)')
    
    # Advanced filters
    with st.sidebar.expander("Advanced filters"):
        min_search_volume = st.number_input('Minimum volume', min_value=0, value=100)
        max_keywords = st.number_input('Maximum keywords', min_value=0, value=100, 
                help="Limit the number of keywords analyzed (0 = no limit)")
        
        cluster_filter = []
        if uploaded_file:
            try:
                df = process_csv(uploaded_file)
                if 'cluster_name' in df.columns:
                    cluster_options = df['cluster_name'].unique()
                    cluster_filter = st.multiselect('Filter by clusters', options=cluster_options)
            except:
                st.sidebar.error("Error processing CSV")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "🔍 Detailed Analysis", "🏆 Competition", "📈 Historical", "🧠 AI Insights"])
    
    # Usage guide
    with st.expander("📖 How to use this tool"):
        st.markdown('''
        ### Step-by-step Guide:

        1. **Upload** your CSV with keywords, clusters and search volumes.
        2. Enter the **domains** you want to analyze (comma-separated).
        3. Enter your **SerpAPI Key** (required for searches).
        4. Select your target **country**, **language** and optionally a **city**.
        5. Use **advanced filters** to limit the analysis if needed.
        6. Review results in the different dashboard tabs.
        7. Export your results in various formats.

        *Note: SerpAPI usage limits apply. Consider API rate limits.*
        ''')
    
    # Process data when all required parameters are provided
    if uploaded_file and domains_input and serp_api_key:
        try:
            # Load and process CSV
            keywords_df = process_csv(uploaded_file)
            
            # Check required columns
            required_columns = ['keyword', 'cluster_name', 'Avg. monthly searches']
            if not all(col in keywords_df.columns for col in required_columns):
                st.error("CSV must contain these columns: 'keyword', 'cluster_name', 'Avg. monthly searches'")
                return
                
            # Filter data according to configuration
            keywords_df = keywords_df[required_columns].dropna()
            
            if min_search_volume > 0:
                keywords_df = keywords_df[keywords_df['Avg. monthly searches'] >= min_search_volume]
            
            if cluster_filter:
                keywords_df = keywords_df[keywords_df['cluster_name'].isin(cluster_filter)]
            
            # Show cost calculator and get optimized keywords
            with tab1:
                st.header("Cost Optimization")
                optimized_df, cost_data = display_cost_calculator(keywords_df)
                
                proceed = st.button("Proceed with Analysis", type="primary")
                
                if not proceed:
                    st.info("Review the cost estimate above and click 'Proceed with Analysis' when ready.")
                    return
            
            # Extract domains
            domains = [d.strip() for d in domains_input.split(',')]
            
            # Prepare base parameters for SerpAPI
            params_template = {
                "engine": "google",
                "google_domain": f"google.{country_code}",
                "location": city or None,
                "hl": language,
                "api_key": serp_api_key
            }
            
            # Process keywords and get results
            with st.spinner('Analyzing keywords... this may take several minutes'):
                results_df = process_keywords_in_batches(optimized_df, domains, params_template)
            
            # If there are results, proceed with analysis
            if not results_df.empty:
                # Calculate advanced metrics
                domain_metrics = calculate_advanced_metrics(results_df)
                
                # Tab 1: General dashboard
                with tab1:
                    # General metrics
                    st.subheader('General Metrics')
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            label="Keywords analyzed",
                            value=f"{len(results_df['Keyword'].unique()):,}",
                            delta=None
                        )
                    
                    with col2:
                        st.metric(
                            label="Total visibility",
                            value=f"{int(results_df['Visibility Score'].sum()):,}",
                            delta=None  # Updated with historical data
                        )
                    
                    with col3:
                        st.metric(
                            label="Average position",
                            value=f"{results_df['Rank'].mean():.1f}",
                            delta=None  # Updated with historical data
                        )
                    
                    with col4:
                        st.metric(
                            label="Estimated traffic",
                            value=f"{int(results_df['Estimated_Traffic'].sum()):,}",
                            delta=None  # Updated with historical data
                        )
                    
                    # Cluster summary
                    st.subheader('Visibility by cluster')
                    
                    cluster_summary = results_df.groupby('Cluster').agg({
                        'Keyword': pd.Series.nunique,
                        'Search Volume': 'sum',
                        'Visibility Score': 'sum',
                        'Estimated_Traffic': 'sum'
                    }).reset_index()
                    
                    cluster_summary.columns = ['Cluster', 'Keyword Count', 'Search Volume', 'Visibility Score', 'Estimated Traffic']
                    
                    # Bar chart to compare clusters
                    fig_clusters = px.bar(
                        cluster_summary,
                        x='Cluster',
                        y=['Visibility Score', 'Search Volume', 'Estimated Traffic'],
                        barmode='group',
                        title='Visibility and potential by cluster',
                        labels={'value': 'Value', 'variable': 'Metric'}
                    )
                    
                    st.plotly_chart(fig_clusters, use_container_width=True)
                    
                    # Position heatmap by domain and cluster
                    st.subheader('Position heatmap')
                    
                    pivot_positions = results_df.pivot_table(
                        values='Rank',
                        index='Cluster',
                        columns='Domain',
                        aggfunc='mean'
                    ).fillna(0)
                    
                    fig_heatmap = px.imshow(
                        pivot_positions,
                        title='Average positions by cluster and domain',
                        labels=dict(x="Domain", y="Cluster", color="Average position"),
                        color_continuous_scale='Viridis_r',  # Reversed scale (low values = better)
                        aspect="auto"
                    )
                    
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                # Tab 2: Detailed analysis
                with tab2:
                    st.subheader('Detailed results')
                    st.dataframe(results_df, use_container_width=True)
                    
                    st.subheader('Advanced metrics by domain and cluster')
                    st.dataframe(domain_metrics, use_container_width=True)
                    
                    # SOV visualization
                    st.subheader('Share of Voice (SOV) by domain')
                    
                    domain_sov = domain_metrics.groupby('Domain').agg({
                        'Total_Visibility_Score': 'sum',
                        'Total_Search_Volume': 'sum',
                        'Keywords_Count': 'sum'
                    }).reset_index()
                    
                    total_visibility = domain_sov['Total_Visibility_Score'].sum()
                    domain_sov['SOV_Percentage'] = (domain_sov['Total_Visibility_Score'] / total_visibility * 100).round(2)
                    
                    fig_sov = px.pie(
                        domain_sov,
                        values='SOV_Percentage',
                        names='Domain',
                        title='Share of Voice (%)',
                        hover_data=['Total_Visibility_Score', 'Keywords_Count']
                    )
                    
                    st.plotly_chart(fig_sov, use_container_width=True)
                    
                    # Bubble chart to compare visibility and positions
                    st.subheader('Visibility vs Average Position')
                    
                    fig_bubble = px.scatter(
                        domain_metrics,
                        x='Average_Position',
                        y='SOV_Percentage',
                        size='Total_Search_Volume',
                        color='Domain',
                        hover_name='Cluster',
                        hover_data=['Keywords_Count', 'Best_Position', 'Worst_Position', 'Total_Estimated_Traffic'],
                        title='Relationship between position and visibility (size = search volume)'
                    )
                    
                    # Reverse X axis so better positions appear on the right
                    fig_bubble.update_xaxes(autorange="reversed")
                    
                    st.plotly_chart(fig_bubble, use_container_width=True)
                
                # Tab 3: Competitive analysis
                with tab3:
                    st.subheader('Competitor Analysis')
                    
                    with st.spinner('Analyzing competitors... this may take several minutes'):
                        competitor_df, opportunity_df = analyze_competitors(results_df, keywords_df, domains, params_template)
                    
                    if not competitor_df.empty:
                        # Competitor visualization
                        st.dataframe(competitor_df, use_container_width=True)
                        
                        fig_competitors = px.bar(
                            competitor_df.head(10), 
                            x='Domain', 
                            y='Total_Visibility',
                            color='Avg_Position',
                            color_continuous_scale='Viridis_r',
                            hover_data=['Appearances', 'SERP_Coverage', 'Keyword_Count', 'Cluster_Count'],
                            title='Top 10 competitors by visibility'
                        )
                        st.plotly_chart(fig_competitors, use_container_width=True)
                    else:
                        st.info("No competitor data found. This could be due to API limitations or analysis errors.")
                    
                    # Opportunities
                    st.subheader('Keyword Opportunities')
                    
                    if not opportunity_df.empty:
                        st.dataframe(opportunity_df.head(20), use_container_width=True)
                        
                        # Opportunity visualization
                        fig_opportunities = px.scatter(
                            opportunity_df.head(50),
                            x='Difficulty',
                            y='Search_Volume',
                            size='Opportunity_Score',
                            color='Cluster',
                            hover_name='Keyword',
                            hover_data=['Competitor_Count'],
                            title='Keyword opportunity map'
                        )
                        st.plotly_chart(fig_opportunities, use_container_width=True)
                    else:
                        st.info("No keyword opportunities identified. This could be because all terms are already ranking or due to analysis errors.")
                
                # Tab 4: Historical analysis
                with tab4:
                    st.subheader('Historical Tracking')
                    
                    # Save current data for historical tracking
                    save_file = save_historical_data(results_df)
                    if save_file:
                        st.success(f"Data saved for historical tracking")
                    
                    # Load historical data
                    historical_data = load_historical_data()
                    
                    if historical_data:
                        # Convert historical data to format for charts
                        trend_data = []
                        
                        for entry in historical_data:
                            date = entry['_meta']['date']
                            
                            for domain, data in entry.items():
                                if domain == '_meta':
                                    continue
                                    
                                if domain in domains:
                                    trend_data.append({
                                        'Date': date,
                                        'Domain': domain,
                                        'Visibility': data.get('total_visibility', 0),
                                        'Keywords': data.get('total_keywords', 0),
                                        'Position': data.get('avg_position', 0)
                                    })
                        
                        # If there's data to show
                        if trend_data:
                            trend_df = pd.DataFrame(trend_data)
                            
                            # Visibility trend chart
                            fig_trend = px.line(
                                trend_df,
                                x='Date',
                                y='Visibility',
                                color='Domain',
                                title='Visibility evolution by domain',
                                markers=True
                            )
                            st.plotly_chart(fig_trend, use_container_width=True)
                            
                            # Average positions chart
                            fig_positions = px.line(
                                trend_df,
                                x='Date',
                                y='Position',
                                color='Domain',
                                title='Average position evolution',
                                markers=True
                            )
                            # Reverse Y axis so better positions appear at the top
                            fig_positions.update_yaxes(autorange="reversed")
                            st.plotly_chart(fig_positions, use_container_width=True)
                    else:
                        st.info("No historical data available. Current results will be saved for future analysis.")
                    
                    # Historical data management
                    with st.expander("Manage historical data"):
                        if st.button("Delete all historical data"):
                            import shutil
                            if os.path.exists('historical_data'):
                                shutil.rmtree('historical_data')
                                os.makedirs('historical_data')
                            st.success("Historical data successfully deleted")
                
                # Tab 5: AI Insights
                with tab5:
                    display_chatgpt_features(openai_api_key, results_df, competitor_df, domain_metrics, opportunity_df)
                
                # Export options
                st.subheader('Export Results')
                
                export_format = st.radio(
                    "Export format",
                    ("CSV", "Excel"),
                    horizontal=True
                )
                
                if export_format == "CSV":
                    # Simple CSV
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button('Download complete results (CSV)', csv, 'seo_visibility_results.csv', 'text/csv')
                    
                    # Advanced metrics
                    metrics_csv = domain_metrics.to_csv(index=False).encode('utf-8')
                    st.download_button('Download advanced metrics (CSV)', metrics_csv, 'seo_advanced_metrics.csv', 'text/csv')
                    
                    # Opportunities
                    if not opportunity_df.empty:
                        opp_csv = opportunity_df.to_csv(index=False).encode('utf-8')
                        st.download_button('Download opportunities (CSV)', opp_csv, 'seo_opportunities.csv', 'text/csv')
                else:
                    # Create Excel file with multiple sheets
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        results_df.to_excel(writer, sheet_name='Results', index=False)
                        domain_metrics.to_excel(writer, sheet_name='Metrics', index=False)
                        
                        if not opportunity_df.empty:
                            opportunity_df.to_excel(writer, sheet_name='Opportunities', index=False)
                        
                        if not competitor_df.empty:
                            competitor_df.to_excel(writer, sheet_name='Competitors', index=False)
                        
                        # Add sheet with executive summary
                        pd.DataFrame({
                            'Metric': ['Total Keywords', 'Domains analyzed', 'Total volume', 'Total visibility'],
                            'Value': [
                                len(results_df['Keyword'].unique()),
                                len(results_df['Domain'].unique()),
                                results_df['Search Volume'].sum(),
                                results_df['Visibility Score'].sum()
                            ]
                        }).to_excel(writer, sheet_name='Summary', index=False)
                        
                    excel_data = output.getvalue()
                    st.download_button('Download complete report (Excel)', excel_data, 'seo_visibility_report.xlsx')
                    
            else:
                st.warning('No results found for the analyzed keywords in the specified domains.')
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)
    else:
        st.info('Complete all fields in the sidebar to start the analysis.')

if __name__ == "__main__":
    main()
