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

# Importar módulos personalizados
from utils.seo_calculator import (
    display_preliminary_calculator,
    detailed_cost_calculator,
    display_cost_breakdown
)
from utils.optimization import (
    cluster_representative_keywords,
    calculate_api_cost, 
    group_similar_keywords,
    estimate_processing_time,
    optimize_batch_sizes  # Se importa para optimizar el tamaño de los batches
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

# Cargar configuración desde config.yaml
def load_config():
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        st.warning("Archivo de configuración no encontrado. Se usarán valores por defecto.")
        return {}
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except Exception as e:
            st.error(f"Error cargando la configuración: {str(e)}")
            return {}

config = load_config()

# Configuración de la página
st.set_page_config(page_title='SEO Visibility Estimator Pro', layout='wide')

# Funciones con caching
@st.cache_data(ttl=86400*7)  # Cache de 7 días
def fetch_serp_results(keyword, params):
    return fetch_serp_results_optimized(
        keyword, 
        params, 
        use_cache=True, 
        cache_ttl=config.get('api', {}).get('serpapi', {}).get('cache_ttl', 86400)
    )

@st.cache_data(ttl=3600*24)  # Cache de 24 horas
def process_csv(uploaded_file):
    return pd.read_csv(uploaded_file)

# Funciones utilitarias
def extract_domain(url):
    pattern = r'(?:https?:\/\/)?(?:www\.)?([^\/\n]+)'
    match = re.search(pattern, url)
    return match.group(1) if match else url

def get_ctr_by_position(position):
    ctr_model = config.get('ctr_model', {})
    default_ctr = ctr_model.get('default', 0.01)
    position_str = str(position)
    if position_str in ctr_model:
        return ctr_model[position_str]
    elif position in ctr_model:
        return ctr_model[position]
    else:
        return default_ctr

# Funciones de análisis de datos
def process_keywords_in_batches(keywords_df, domains, params_template, optimization_settings=None):
    if optimization_settings is None:
        optimization_settings = {"batch_optimization": True, "max_retries": 3}
    batch_optimization = optimization_settings.get("batch_optimization", True)
    max_retries = optimization_settings.get("max_retries", 3)
    default_batch_size = config.get('optimization', {}).get('batching', {}).get('min_batch_size', 5)
    
    if batch_optimization and 'cluster_name' in keywords_df.columns:
        batch_sizes = optimize_batch_sizes(keywords_df)
    else:
        batch_sizes = {'default': default_batch_size}
    
    all_results = []
    progress_bar = st.progress(0)
    total_processed = 0
    
    if 'cluster_name' in keywords_df.columns and batch_optimization:
        clusters = keywords_df['cluster_name'].unique()
        for i, cluster in enumerate(clusters):
            cluster_data = keywords_df[keywords_df['cluster_name'] == cluster]
            batch_size = batch_sizes.get(cluster, batch_sizes.get('default', default_batch_size))
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
                                    st.warning(f"Error procesando '{row['keyword']}' tras {max_retries} intentos: {str(e)}")
                                time.sleep(2)
                total_processed += len(batch)
                progress_bar.progress(min(1.0, total_processed / len(keywords_df)))
                if j + batch_size < len(cluster_data):
                    pause_time = config.get('api', {}).get('serpapi', {}).get('batch_pause', 0.2)
                    pause_time = min(2.0, pause_time * batch_size)
                    time.sleep(pause_time)
            if len(clusters) > 5:
                st.info(f"Cluster procesado: '{cluster}' ({i+1}/{len(clusters)})")
    else:
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
                                st.warning(f"Error procesando '{row['keyword']}' tras {max_retries} intentos: {str(e)}")
                            time.sleep(2)
            progress_bar.progress(min(1.0, (i + default_batch_size) / len(keywords_df)))
            if i + default_batch_size < len(keywords_df):
                pause_time = config.get('api', {}).get('serpapi', {}).get('batch_pause', 0.2)
                pause_time = min(2.0, pause_time * default_batch_size)
                time.sleep(pause_time)
    
    return pd.DataFrame(all_results) if all_results else pd.DataFrame()

def calculate_advanced_metrics(results_df):
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

def analyze_competitors(results_df, keywords_df, domains, params_template, optimization_settings=None):
    if keywords_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    if optimization_settings is None:
        optimization_settings = {"limit_analysis": True, "sample_size": 50, "serp_depth": 10}
    limit_analysis = optimization_settings.get("limit_analysis", True)
    sample_size = optimization_settings.get("sample_size", 50)
    serp_depth = optimization_settings.get("serp_depth", 10)
    
    all_competitors = {}
    keyword_count = {}
    progress_bar = st.progress(0)
    
    if limit_analysis:
        config_sample_size = config.get('optimization', {}).get('competitor_analysis', {}).get('max_sample_size')
        if config_sample_size:
            sample_size = min(config_sample_size, len(keywords_df))
        if 'cluster_name' in keywords_df.columns:
            clusters = keywords_df['cluster_name'].unique()
            samples_per_cluster = max(1, sample_size // len(clusters))
            analysis_sample = pd.DataFrame()
            for cluster in clusters:
                cluster_kws = keywords_df[keywords_df['cluster_name'] == cluster]
                top_kws = cluster_kws.sort_values('Avg. monthly searches', ascending=False).head(samples_per_cluster)
                analysis_sample = pd.concat([analysis_sample, top_kws])
            if len(analysis_sample) > sample_size:
                analysis_sample = analysis_sample.sort_values('Avg. monthly searches', ascending=False).head(sample_size)
        else:
            analysis_sample = keywords_df.sort_values('Avg. monthly searches', ascending=False).head(sample_size)
    else:
        analysis_sample = keywords_df

    config_serp_depth = config.get('optimization', {}).get('competitor_analysis', {}).get('default_serp_depth')
    if config_serp_depth:
        serp_depth = config_serp_depth
    
    for i, row in enumerate(analysis_sample.iterrows()):
        _, row_data = row
        params = params_template.copy()
        params["q"] = row_data['keyword']
        params["num"] = serp_depth
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
            st.error(f"Error analizando competidores para '{row_data['keyword']}': {str(e)}")
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
    non_ranking_limit = 200
    non_ranking_keywords = keywords_df[~keywords_df['keyword'].isin(ranking_keywords)]
    if len(non_ranking_keywords) > non_ranking_limit:
        non_ranking_keywords = non_ranking_keywords.sort_values('Avg. monthly searches', ascending=False).head(non_ranking_limit)
    
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

def save_historical_data(results_df):
    if results_df.empty:
        return None
    historical_dir = config.get('performance', {}).get('historical_data_dir', 'historical_data')
    if not os.path.exists(historical_dir):
        os.makedirs(historical_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{historical_dir}/seo_data_{timestamp}.json"
    
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
        for cluster in domain_data['Cluster'].unique():
            cluster_data = domain_data[domain_data['Cluster'] == cluster]
            summary[domain]['clusters'][cluster] = {
                'keywords': len(cluster_data['Keyword'].unique()),
                'volume': int(cluster_data['Search Volume'].sum()),
                'visibility': int(cluster_data['Visibility Score'].sum()),
                'avg_position': float(cluster_data['Rank'].mean())
            }
    summary['_meta'] = {
        'timestamp': datetime.now().isoformat(),
        'version': '1.0',
        'total_domains': len(results_df['Domain'].unique()),
        'total_keywords': len(results_df['Keyword'].unique()),
        'total_clusters': len(results_df['Cluster'].unique()),
    }
    with open(filename, 'w') as f:
        json.dump(summary, f, indent=2)
    return filename

def load_historical_data():
    historical_dir = config.get('performance', {}).get('historical_data_dir', 'historical_data')
    if not os.path.exists(historical_dir):
        return []
    historical_data = []
    for filename in os.listdir(historical_dir):
        if filename.startswith('seo_data_') and filename.endswith('.json'):
            with open(os.path.join(historical_dir, filename), 'r') as f:
                try:
                    data = json.load(f)
                    date_str = filename.replace('seo_data_', '').replace('.json', '')
                    date = datetime.strptime(date_str.split('_')[0], "%Y%m%d")
                    if '_meta' not in data:
                        data['_meta'] = {}
                    data['_meta']['date'] = date.strftime("%Y-%m-%d")
                    data['_meta']['filename'] = filename
                    historical_data.append(data)
                except Exception as e:
                    st.error(f"Error cargando el archivo histórico {filename}: {str(e)}")
                    continue
    historical_data.sort(key=lambda x: x['_meta'].get('date', ''))
    return historical_data

# Integración con ChatGPT
class ChatGPTAnalyzer:
    def __init__(self, api_key, use_gpt35=True, limit_analysis=True):
        self.api_key = api_key
        openai.api_key = api_key
        self.use_gpt35 = use_gpt35
        self.limit_analysis = limit_analysis
        self.analysis_count = 0
        self.max_analyses = config.get('api', {}).get('openai', {}).get('max_analyses_per_session', 5)
    
    def analyze_serp_competitors(self, keyword, competitors):
        if self.limit_analysis:
            if self.analysis_count >= self.max_analyses:
                return "Límite de análisis alcanzado. Para más análisis, deshabilita la limitación en la configuración."
            self.analysis_count += 1
        if not self.api_key or not competitors:
            return "Falta la API key o los datos de competidores"
        competitors_to_analyze = 3 if self.limit_analysis else 5
        competitor_text = "\n".join([
            f"{i+1}. {comp['Domain']} (Position: {comp['Rank']})"
            for i, comp in enumerate(competitors[:competitors_to_analyze])
        ])
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
            model = "gpt-3.5-turbo" if self.use_gpt35 else "gpt-4"
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an SEO content strategy expert. Be concise."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=400 if self.limit_analysis else 500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate_cluster_content_brief(self, cluster_name, keywords, competition_level="medium"):
        if self.limit_analysis:
            if self.analysis_count >= self.max_analyses:
                return "Límite de análisis alcanzado. Para más análisis, deshabilita la limitación en la configuración."
            self.analysis_count += 1
        if not self.api_key:
            return "Falta la API key"
        keyword_limit = 5 if self.limit_analysis else 10
        keywords_text = "\n".join([
            f"- {kw['Keyword']} ({kw['Search_Volume']} searches/month)"
            for kw in keywords[:keyword_limit]
        ])
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
        if self.limit_analysis:
            if self.analysis_count >= self.max_analyses:
                return "Límite de análisis alcanzado. Para más análisis, deshabilita la limitación en la configuración."
            self.analysis_count += 1
        if not self.api_key:
            return "Falta la API key"
        competitor_limit = 3
        competitor_text = "\n".join([
            f"- {comp['Domain']} (Visibility: {comp['Total_Visibility']}, Keywords: {comp['Keyword_Count']})"
            for comp in competitors[:competitor_limit]
        ])
        cluster_limit = 10 if self.limit_analysis else 20
        sorted_clusters = sorted(
            cluster_data.items(), 
            key=lambda x: x[1].get('Total_Visibility_Score', 0), 
            reverse=True
        )[:cluster_limit]
        cluster_text = "\n".join([
            f"- {cluster}: Keywords: {data['Keywords_Count']}, Avg Position: {data['Average_Position']:.1f}"
            for cluster, data in sorted_clusters
        ])
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

def display_chatgpt_features(openai_api_key, results_df, competitor_df, domain_metrics, opportunity_df, 
                           use_gpt35=True, limit_analysis=True):
    st.header("AI-Powered SEO Insights")
    if not openai_api_key:
        st.warning("Introduce tu API Key de OpenAI en la barra lateral para desbloquear los insights con IA")
        return
    if use_gpt35:
        st.success("✅ Usando GPT-3.5 Turbo para análisis (más económico)")
    else:
        st.info("ℹ️ Usando GPT-4 para análisis (mayor calidad, mayor costo)")
    if limit_analysis:
        st.success("✅ Limitando el número de análisis para reducir costos")
    
    analyzer = ChatGPTAnalyzer(openai_api_key, use_gpt35=use_gpt35, limit_analysis=limit_analysis)
    tab1, tab2, tab3 = st.tabs(["Content Strategy", "Cluster Briefs", "Strategic Opportunities"])
    
    with tab1:
        st.subheader("Competitor Content Analysis")
        st.write("Analiza a los competidores que posicionan para palabras clave específicas y obtén insights para tu estrategia de contenidos")
        if not results_df.empty:
            selected_keyword = st.selectbox(
                "Selecciona una palabra clave para analizar:",
                options=results_df['Keyword'].unique(),
                index=0
            )
            keyword_data = results_df[results_df['Keyword'] == selected_keyword]
            competitors = [{"Domain": row['Domain'], "Rank": row['Rank']} for _, row in keyword_data.iterrows()]
            if st.button("Generar estrategia de contenido", key="content_strategy"):
                with st.spinner("Analizando contenido de competidores..."):
                    analysis = analyzer.analyze_serp_competitors(selected_keyword, competitors)
                    st.markdown(analysis)
        else:
            st.info("Ejecuta el análisis primero para ver las palabras clave")
    
    with tab2:
        st.subheader("Cluster Content Briefs")
        st.write("Genera briefs de contenido integrales para clusters de palabras clave")
        if not results_df.empty:
            selected_cluster = st.selectbox(
                "Selecciona un cluster:",
                options=results_df['Cluster'].unique(),
                index=0
            )
            cluster_keywords = results_df[results_df['Cluster'] == selected_cluster]
            keywords_for_brief = [{"Keyword": row['Keyword'], "Search_Volume": row['Search Volume']} for _, row in cluster_keywords.iterrows()]
            competition = st.select_slider("Nivel de competencia estimado:", options=["low", "medium", "high"], value="medium")
            if st.button("Generar brief de contenido", key="cluster_brief"):
                with st.spinner("Creando brief de contenido..."):
                    brief = analyzer.generate_cluster_content_brief(selected_cluster, keywords_for_brief, competition)
                    st.markdown(brief)
        else:
            st.info("Ejecuta el análisis primero para ver los clusters")
    
    with tab3:
        st.subheader("Análisis de Oportunidades Estratégicas")
        st.write("Obtén recomendaciones estratégicas basadas en el análisis competitivo")
        if not results_df.empty and not competitor_df.empty and not domain_metrics.empty:
            selected_domain = st.selectbox(
                "Selecciona tu dominio:",
                options=results_df['Domain'].unique(),
                index=0
            )
            cluster_performance = {}
            for _, row in domain_metrics[domain_metrics['Domain'] == selected_domain].iterrows():
                cluster_performance[row['Cluster']] = {
                    "Keywords_Count": row['Keywords_Count'],
                    "Average_Position": row['Average_Position'],
                    "Total_Visibility_Score": row['Total_Visibility_Score']
                }
            if st.button("Generar análisis estratégico", key="strategy"):
                with st.spinner("Analizando oportunidades estratégicas..."):
                    strategy = analyzer.analyze_opportunity_gaps(selected_domain, competitor_df.to_dict('records'), cluster_performance)
                    st.markdown(strategy)
        else:
            st.info("Ejecuta un análisis completo primero para acceder a insights estratégicos")

# Función principal que integra la interfaz de usuario
def main():
    st.title("Cluster Visibility Analysis Tool")
    st.sidebar.header("Configuración")
    uploaded_file = st.sidebar.file_uploader("Sube el archivo CSV con los datos de palabras clave", type=["csv"])
    domains_input = st.sidebar.text_input("Ingresa los dominios a analizar (separados por coma)", "example.com")
    openai_api_key = st.sidebar.text_input("API Key de OpenAI", type="password")
    
    if uploaded_file is not None:
        keywords_df = process_csv(uploaded_file)
        st.subheader("Datos cargados")
        st.dataframe(keywords_df)
        domains = [d.strip() for d in domains_input.split(",")]
        params_template = {
            "engine": "google",
            "hl": "es",
            "gl": "es"
        }
        st.subheader("Procesando análisis de visibilidad...")
        results_df = process_keywords_in_batches(keywords_df, domains, params_template)
        if not results_df.empty:
            st.success("¡Análisis completado!")
            st.dataframe(results_df)
            domain_metrics = calculate_advanced_metrics(results_df)
            st.subheader("Métricas avanzadas")
            st.dataframe(domain_metrics)
            competitor_df, opportunity_df = analyze_competitors(results_df, keywords_df, domains, params_template)
            st.subheader("Competidores")
            st.dataframe(competitor_df)
            st.subheader("Oportunidades")
            st.dataframe(opportunity_df)
            if st.button("Guardar datos históricos"):
                filename = save_historical_data(results_df)
                if filename:
                    st.success(f"Datos guardados en {filename}")
            display_chatgpt_features(openai_api_key, results_df, competitor_df, domain_metrics, opportunity_df)
        else:
            st.info("No se obtuvieron resultados. Revisa el archivo y la configuración.")
    else:
        st.info("Por favor, sube un archivo CSV para comenzar el análisis.")

if __name__ == "__main__":
    main()
