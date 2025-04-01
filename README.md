# SEO Visibility Estimator Pro

An advanced SEO analysis tool that estimates keyword visibility, analyzes competitive landscapes, and optimizes API usage costs.

## Features

- **Keyword Visibility Analysis**: Analyze how your domains rank for specific keywords
- **Cluster-Based Analysis**: Insights organized by semantic keyword clusters
- **Competitive Analysis**: Identify top competitors and find opportunities
- **Cost Optimization**: Intelligent API usage optimization to reduce costs
- **Historical Tracking**: Monitor SEO performance over time
- **AI-Powered Insights**: Get content and strategy recommendations

## Cost Optimization Features

The application includes several features to minimize API costs:

- **Preliminary Cost Calculator**: Estimate API costs before running full analysis
- **Intelligent Keyword Sampling**: Select representative keywords from each cluster
- **Batch Processing Optimization**: Group queries efficiently to reduce API calls
- **Caching System**: Prevent duplicate API calls through intelligent caching
- **Strategic SERP Depth**: Configure how many SERP results to analyze

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/seo-visibility-estimator.git
cd seo-visibility-estimator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your `config.yaml` file (see Configuration section)

4. Run the application:
```bash
streamlit run app.py
```

## Requirements

- Python 3.7+
- Streamlit
- pandas
- plotly
- SerpAPI account (for SERP data)
- OpenAI API key (optional, for AI insights)

See `requirements.txt` for a complete list of dependencies.

## Configuration

The application uses a central `config.yaml` file for customizing behavior:

```yaml
# Example configuration
api:
  serpapi:
    rate_limit: 1.0
    plans:
      basic:
        monthly_cost: 50
        searches: 5000

optimization:
  sampling:
    enabled: true
    advanced: true
  batching:
    enabled: true
    min_batch_size: 3
```

See the `config.yaml` file for all available options.

## Input CSV Format

Your CSV file should contain the following columns:
- `keyword`: The search term
- `cluster_name`: Semantic cluster or category
- `Avg. monthly searches`: Monthly search volume

Example:
```csv
keyword,cluster_name,Avg. monthly searches
seo tools,tools,5400
keyword research,research,2900
backlink checker,tools,1800
```

## Usage

1. **Upload your CSV** with keywords, clusters, and search volumes
2. **Enter domains** you want to analyze (comma-separated)
3. **Enter your SerpAPI Key** (required for searches)
4. **Configure location** settings (country, language)
5. **Review the cost estimate** before proceeding
6. **Analyze results** in the dashboard tabs

## Project Structure

```
seo-visibility-estimator/
├── app.py                      # Main application
├── requirements.txt            # Dependencies
├── config.yaml                 # Configuration
├── utils/                      # Utility modules
│   ├── __init__.py             # Package initialization
│   ├── seo_calculator.py       # Cost calculators
│   ├── optimization.py         # Optimization functions
│   ├── api_manager.py          # API handling
│   └── data_processing.py      # Data processing utilities
├── cache/                      # Cache directory
│   └── .gitignore              # Git ignore for cache
├── historical_data/            # Historical data
│   └── .gitignore              # Git ignore for historical data
└── README.md                   # This file
```

## Extending the Application

### Adding New Optimization Strategies

1. Implement your strategy in the appropriate module in `utils/`
2. Add configuration options to `config.yaml`
3. Integrate with the main application in `app.py`

### Supporting New Data Sources

1. Create a new data connector in the `utils/` directory
2. Implement the interface to match the expected format
3. Add configuration options in `config.yaml`

## License

[MIT License](LICENSE)

## Acknowledgments

- [SerpAPI](https://serpapi.com/) for SERP data
- [Streamlit](https://streamlit.io/) for the web interface
- [Plotly](https://plotly.com/) for visualizations
