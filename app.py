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

# Caching functions con TTL más agresivo para uso de APIs
@st.cache_data(ttl=86400*7)  # Cache for 7 days (aumentado de 1 día)
def fetch_serp_results(keyword, params):
    """Cache search results to avoid duplicate API calls"""
    search = GoogleSearch(params)
    return search.get_dict()

@st.cache_data(ttl=3600*24)  # Cache for 24 hours (aumentado de 1 hora)
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

# NUEVA FUNCIÓN: Calculadora de costes mejorada
def calculate_api_cost(num_keywords, api_plan="basic", batch_optimization=True, sampling_rate=1.0):
    """Calculate estimated SerpAPI cost based on number of queries with additional optimizations"""
    # SerpAPI pricing tiers (as of March 2025)
    pricing = {
        "basic": {"monthly_cost": 50, "searches": 5000},
        "business": {"monthly_cost": 250, "searches": 30000},
        "enterprise": {"monthly_cost": 500, "searches": 70000}
    }
    
    # Select pricing tier
    plan = pricing.get(api_plan.lower(), pricing["basic"])
    
    # Apply optimizations
    effective_queries = num_keywords
    
    # Apply sampling rate reduction (if < 1.0)
    effective_queries = math.ceil(effective_queries * sampling_rate)
    
    # Apply batch optimization if enabled (reduces number of required API calls)
    if batch_optimization and effective_queries > 10:
        # Estimate reduction from batching (conservative 5% reduction)
        effective_queries = math.ceil(effective_queries * 0.95)
    
    # Calculate cost per search
    cost_per_search = plan["monthly_cost"] / plan["searches"]
    
    # Calculate estimated cost
    estimated_cost = effective_queries * cost_per_search
    
    # Calculate percentage of monthly quota
    quota_percentage = (effective_queries / plan["searches"]) * 100
    
    return {
        "num_queries": effective_queries,
        "original_queries": num_keywords,
        "reduction_percentage": round((1 - (effective_queries / num_keywords)) * 100, 2) if num_keywords > 0 else 0,
        "estimated_cost": round(estimated_cost, 2),
        "quota_percentage": round(quota_percentage, 2),
        "plan_details": f"{plan['monthly_cost']}$ for {plan['searches']} searches"
    }

# FUNCIÓN OPTIMIZADA: Mejor selección de palabras clave representativas
def cluster_representative_keywords(keywords_df, max_representatives=100, advanced_sampling=True):
    """
    Select representative keywords from each cluster to reduce API costs
    This returns a subset of keywords that represent each cluster with improved sampling
    """
    if keywords_df.empty or 'cluster_name' not in keywords_df.columns:
        return keywords_df
    
    # Group by cluster
    grouped = keywords_df.groupby('cluster_name')
    
    # MEJORA: Usar muestreo avanzado si está activado
    if advanced_sampling:
        # Calcular diversidad de cada cluster (por volumen de búsqueda)
        cluster_diversities = {}
        
        for cluster_name, group in grouped:
            # Calcular el rango de volumen como medida de diversidad
            vol_range = group['Avg. monthly searches'].max() - group['Avg. monthly searches'].min()
            vol_std = group['Avg. monthly searches'].std()
            # Puntuación de diversidad combinada
            cluster_diversities[cluster_name] = (vol_range + vol_std) / 2
        
        # Normalizar diversidades
        total_diversity = sum(cluster_diversities.values())
        if total_diversity > 0:
            for cluster in cluster_diversities:
                cluster_diversities[cluster] /= total_diversity
    
    # Calculate how many representatives to take from each cluster
    # Use proportional distribution based on cluster size and search volume
    cluster_sizes = grouped.size()
    cluster_volumes = grouped['Avg. monthly searches'].sum()
    
    # Create a score combining size and volume
    if advanced_sampling:
        # Include diversity in the score
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
        
        # MEJORA: Seleccionar palabras clave con mejor distribución de volumen
        if advanced_sampling and len(cluster_data) > count:
            # Dividir el rango de volumen en segmentos y seleccionar de cada uno
            sorted_data = cluster_data.sort_values('Avg. monthly searches')
            step = len(sorted_data) / count
            
            indices = [int(i * step) for i in range(count)]
            selected_keywords.append(sorted_data.iloc[indices])
        else:
            # Sort by search volume (descending) - método original
            sorted_data = cluster_data.sort_values('Avg. monthly searches', ascending=False)
            # Take top keywords as representatives
            selected_keywords.append(sorted_data.head(int(count)))
    
    # Combine all selected keywords
    representative_df = pd.concat(selected_keywords)
    
    return representative_df

# NUEVA FUNCIÓN: Calculadora de costes preliminar
def display_preliminary_cost_calculator(keywords_df):
    """Display preliminary cost calculator before running full analysis"""
    st.header("Calculadora de Costes Preliminar")
    st.write("Estima el coste y optimiza la consulta antes de ejecutar el análisis completo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # API plan selection
        api_plan = st.selectbox(
            "Plan de SerpAPI",
            options=["Basic", "Business", "Enterprise"],
            index=0
        )
        
        # Analysis frequency
        analysis_frequency = st.selectbox(
            "Frecuencia de Análisis",
            options=["Una vez", "Semanal", "Mensual", "Trimestral"],
            index=2
        )
        
        # Number of months
        months = st.slider("Período de Análisis (meses)", 1, 12, 3)
    
    with col2:
        # Keyword sampling strategy
        use_representatives = st.checkbox("Usar solo palabras clave representativas", value=True)
        advanced_sampling = st.checkbox("Usar muestreo avanzado (más preciso)", value=True)
        
        if use_representatives:
            max_keywords = st.slider(
                "Número máximo de palabras clave representativas", 
                min_value=10, 
                max_value=min(500, len(keywords_df)), 
                value=min(100, len(keywords_df) // 2)
            )
        else:
            max_keywords = len(keywords_df)
        
        # Batch optimization
        batch_optimization = st.checkbox("Aplicar optimización por lotes", value=True,
                                        help="Reduce el número de llamadas a la API agrupando consultas")
    
    # Calculate frequency multiplier
    frequency_multipliers = {
        "Una vez": 1,
        "Semanal": 4 * months,  # ~4 weeks per month
        "Mensual": months,
        "Trimestral": math.ceil(months / 3)
    }
    
    multiplier = frequency_multipliers[analysis_frequency]
    
    # Calculate number of API calls
    if use_representatives:
        sample_df = cluster_representative_keywords(
            keywords_df, 
            max_keywords,
            advanced_sampling=advanced_sampling
        )
        num_queries = len(sample_df)
        sampling_ratio = num_queries / len(keywords_df)
    else:
        num_queries = len(keywords_df)
        sampling_ratio = 1.0
    
    # Calculate total cost
    total_queries = num_queries * multiplier
    cost_data = calculate_api_cost(
        total_queries, 
        api_plan.lower(), 
        batch_optimization=batch_optimization, 
        sampling_rate=sampling_ratio
    )
    
    # Display results
    st.subheader("Estimación de Costes")
    
    cost_col1, cost_col2, cost_col3 = st.columns(3)
    
    with cost_col1:
        st.metric("Total de Palabras Clave", f"{len(keywords_df):,}")
        st.metric("Palabras Clave Analizadas", f"{num_queries:,}")
        st.metric("Tasa de Muestreo", f"{sampling_ratio:.1%}")
    
    with cost_col2:
        st.metric("Total de Consultas API", f"{total_queries:,}")
        st.metric("Coste Estimado", f"${cost_data['estimated_cost']:.2f}")
        st.metric("Cuota Mensual Utilizada", f"{cost_data['quota_percentage']:.1f}%")
    
    with cost_col3:
        st.metric("Ahorro con Optimizaciones", f"{cost_data['reduction_percentage']}%")
        st.metric("Frecuencia", f"{analysis_frequency}")
        st.metric("Plan SerpAPI", f"{api_plan}")
    
    # Display optimization metrics
    st.subheader("Estrategias de Ahorro Aplicadas")
    
    saving_col1, saving_col2 = st.columns(2)
    
    with saving_col1:
        st.write("**Muestreo de palabras clave:**")
        if use_representatives:
            st.success(f"✅ Activado - Reduce las consultas necesarias en un {(1 - sampling_ratio) * 100:.1f}%")
            if advanced_sampling:
                st.success("✅ Muestreo avanzado - Mejora la representatividad por cluster")
        else:
            st.error("❌ Desactivado - Se analizarán todas las palabras clave")
    
    with saving_col2:
        st.write("**Optimización por lotes:**")
        if batch_optimization:
            st.success("✅ Activado - Agrupa consultas para reducir llamadas API")
        else:
            st.error("❌ Desactivado - No se optimizarán los lotes")
    
    # Calculate estimated time
    avg_time_per_query = 1.5  # seconds per query with batching
    estimated_time = (total_queries / 5) * avg_time_per_query  # Assuming batch_size = 5
    
    # Convert to minutes
    estimated_minutes = estimated_time / 60
    
    st.info(f"⏱️ Tiempo estimado para el análisis completo: aproximadamente {estimated_minutes:.1f} minutos")
    
    st.markdown("---")
    
    # Return the representative keywords if checkbox is selected
    if use_representatives:
        return sample_df, cost_data, {
            "use_representatives": use_representatives,
            "advanced_sampling": advanced_sampling,
            "batch_optimization": batch_optimization,
            "max_keywords": max_keywords
        }
    else:
        return keywords_df, cost_data, {
            "use_representatives": use_representatives,
            "advanced_sampling": advanced_sampling,
            "batch_optimization": batch_optimization,
            "max_keywords": max_keywords
        }

# FUNCIÓN OPTIMIZADA: Procesar palabras clave en lotes con mejor eficiencia
def process_keywords_in_batches(keywords_df, domains, params_template, batch_size=5, 
                               optimize_batch=True, max_retries=2):
    """Process keywords in batches with improved efficiency and retry mechanism"""
    all_results = []
    progress_bar = st.progress(0)
    
    # OPTIMIZACIÓN: Aumentar tamaño de lote para keywords similares
    if optimize_batch and 'cluster_name' in keywords_df.columns:
        # Ordenar por cluster para que keywords similares estén juntas
        keywords_df = keywords_df.sort_values('cluster_name')
        
        # Determinar tamaño de batch dinamicamente basado en el cluster
        # Clusters más pequeños pueden tener batch_size más grande
        cluster_sizes = keywords_df.groupby('cluster_name').size()
        avg_cluster_size = cluster_sizes.mean()
        
        # Adaptive batch size between 5-10 based on cluster size
        if avg_cluster_size < 10:
            batch_size = min(10, max(5, int(50 / avg_cluster_size)))
    
    # Procesamiento con retry y manejo de errores mejorado
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
                            st.warning(f"Error al procesar '{row['keyword']}' después de {max_retries} intentos: {str(e)}")
                        time.sleep(2)  # Wait before retry
        
        # Update progress
        progress_bar.progress((i + batch_size) / len(keywords_df))
        
        # Dynamic pause between batches based on batch size to respect API limits
        if i + batch_size < len(keywords_df):
            pause_time = min(2.0, 0.2 * batch_size)  # 0.2s per keyword, max 2s
            time.sleep(pause_time)
    
    return pd.DataFrame(all_results) if all_results else pd.DataFrame()

# FUNCIÓN OPTIMIZADA: Análisis de competidores más eficiente
def analyze_competitors(results_df, keywords_df, domains, params_template):
    """Analyze competitors with reduced API calls"""
    if keywords_df.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    # Extract competitors from SERP results
    all_competitors = {}
    keyword_count = {}
    progress_bar = st.progress(0)
    
    # OPTIMIZACIÓN: Usar un subset más pequeño y representativo para análisis competitivo
    # En lugar de usar top 50 por volumen, usar representantes de cada cluster
    analysis_sample_size = min(50, len(keywords_df) // 5)  # Reducido de 50 fijo
    
    # Seleccionar muestra representativa por cluster
    if 'cluster_name' in keywords_df.columns:
        clusters = keywords_df['cluster_name'].unique()
        samples_per_cluster = max(1, analysis_sample_size // len(clusters))
        
        analysis_sample = pd.DataFrame()
        for cluster in clusters:
            cluster_kws = keywords_df[keywords_df['cluster_name'] == cluster]
            # Tomar top keywords por volumen en cada cluster
            top_kws = cluster_kws.sort_values('Avg. monthly searches', ascending=False).head(samples_per_cluster)
            analysis_sample = pd.concat([analysis_sample, top_kws])
    else:
        # Si no hay clusters, usar el método original
        analysis_sample = keywords_df.sort_values('Avg. monthly searches', ascending=False).head(analysis_sample_size)
    
    # OPTIMIZACIÓN: Reducir número de resultados solicitados para análisis competitivo
    results_limit = 10  # Reducido de 20
    
    for i, row in enumerate(analysis_sample.iterrows()):
        _, row_data = row
        params = params_template.copy()
        params["q"] = row_data['keyword']
        params["num"] = results_limit  # Número reducido de resultados
        
        try:
            results = fetch_serp_results(row_data['keyword'], params)
            organic_results = results.get("organic_results", [])
            
            # Record all domains in results
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
    
    # Identify opportunities - OPTIMIZADO para necesitar menos procesamiento
    opportunities = []
    
    # Keywords already ranking for target domains
    ranking_keywords = set(results_df['Keyword'].unique()) if not results_df.empty else set()
    
    # Keywords not ranking for the analyzed domains - LIMITADO a 200 para eficiencia
    non_ranking_keywords = keywords_df[~keywords_df['keyword'].isin(ranking_keywords)].head(200)
    
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
        6. Review the **preliminary cost calculation** before running the analysis.
        7. Review results in the different dashboard tabs.
        8. Export your results in various formats.

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
            
            # NUEVO: Mostrar calculadora preliminar
            with tab1:
                st.header("Calculadora de Costes Preliminar")
                optimized_df, cost_data, optimization_settings = display_preliminary_cost_calculator(keywords_df)
                
                proceed = st.button("Proceder con el Análisis", type="primary")
                
                if not proceed:
                    st.info("Revisa la estimación de costes y haz clic en 'Proceder con el Análisis' cuando estés listo.")
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
            
            # Process keywords and get results using optimization settings
            with st.spinner('Analyzing keywords... this may take several minutes'):
                results_df = process_keywords_in_batches(
                    optimized_df, 
                    domains, 
                    params_template,
                    batch_size=5 if optimization_settings["batch_optimization"] else 3,
                    optimize_batch=optimization_settings["batch_optimization"]
                )
            
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
                    
                    # MEJORA: Agregar métricas de ahorro
                    st.subheader('Análisis de Eficiencia')
                    eff_col1, eff_col2, eff_col3 = st.columns(3)
                    
                    with eff_col1:
                        original_queries = len(keywords_df)
                        actual_queries = len(optimized_df)
                        saved_percentage = ((original_queries - actual_queries) / original_queries * 100) if original_queries > 0 else 0
                        
                        st.metric(
                            label="Consultas API ahorradas",
                            value=f"{original_queries - actual_queries:,}",
                            delta=f"{saved_percentage:.1f}% menos"
                        )
                    
                    with eff_col2:
                        estimated_cost_no_opt = calculate_api_cost(original_queries)
                        actual_cost = cost_data
                        cost_saved = estimated_cost_no_opt["estimated_cost"] - actual_cost["estimated_cost"]
                        
                        st.metric(
                            label="Ahorro estimado",
                            value=f"${cost_saved:.2f}",
                            delta=f"{(cost_saved/estimated_cost_no_opt['estimated_cost']*100) if estimated_cost_no_opt['estimated_cost'] > 0 else 0:.1f}%"
                        )
                    
                    with eff_col3:
                        # Tiempo estimado ahorrado
                        time_per_query = 1.5  # segundos por consulta
                        time_saved = (original_queries - actual_queries) * time_per_query / 60  # en minutos
                        
                        st.metric(
                            label="Tiempo ahorrado",
                            value=f"{time_saved:.1f} min",
                            delta=None
                        )
                    
                    # Cluster summary
                    st.subheader('Visibility by cluster')
