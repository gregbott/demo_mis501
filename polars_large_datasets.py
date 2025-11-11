import marimo

__generated_with = "0.16.3"
app = marimo.App()


@app.cell
def _():
    import polars as pl
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import time
    from datetime import datetime
    import plotly.express as px
    import plotly.graph_objects as go
    import marimo as mo

    # Set up plotting style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)

    # Parquet file path (constant throughout notebook)
    parquet_path = "./data/chicago_crime_2001_2025.parquet"
    return parquet_path, pd, pl, px, time


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(parquet_path, pl):
    # Method 1: Eager Loading (loads entire file into memory)
    df_eager = pl.read_parquet(parquet_path)

    print(f"Eager Load - Shape: {df_eager.shape}")
    print(f"Eager Load - Columns: {df_eager.columns}")
    print(f"\nFirst few rows:")
    df_eager.head()
    return (df_eager,)


@app.cell
def _(df_eager):
    # Get basic statistics
    print("DataFrame Info:")
    print(f"Shape: {df_eager.shape}")
    print(f"Memory Usage: {df_eager.estimated_size() / 1024 / 1024:.2f} MB")
    print(f"\nData Types:")
    df_eager.schema
    return


@app.cell
def _():
    return


@app.cell
def _(parquet_path, pl):
    # Method 2: Lazy Loading (doesn't load data yet)
    df_lazy = pl.scan_parquet(parquet_path)

    print("Lazy Load - No data loaded yet!")
    print(f"Schema: {df_lazy.schema}")
    print(f"Lazy query plan:")
    print(df_lazy)
    return


@app.cell
def _():
    return


@app.cell
def _(parquet_path, pl):
    # Build a complex query with lazy evaluation
    lazy_query = (
        pl.scan_parquet(parquet_path)
        .filter(pl.col("Year") >= 2010)  # Filter first (efficient!)
        .select([
            "Year",
            "Primary Type",
            "Arrest",
            "Domestic",
            "Latitude",
            "Longitude",
            "Date"
        ])
        .with_columns([
            (pl.col("Arrest").cast(pl.Boolean)).alias("Was_Arrest"),
            (pl.col("Domestic").cast(pl.Boolean)).alias("Domestic_Crime")
        ])
    )

    print("Lazy Query Plan (optimized):")
    print(lazy_query)

    print(lazy_query.explain)

    # Collect (execute) the query
    print("\nCollecting results...")
    df_filtered = lazy_query.collect()

    print(f"\nFiltered DataFrame Shape: {df_filtered.shape}")
    print(f"Memory Usage: {df_filtered.estimated_size() / 1024 / 1024:.2f} MB")
    print(f"\nFirst few rows:")
    df_filtered.head()
    return


@app.cell
def _():
    return


@app.cell
def _(parquet_path, pd, pl, time):
    # Polars: Lazy loading with filter
    timings = {}

    start_polars = time.time()
    df_polars = (
        pl.scan_parquet(parquet_path)
        .filter(pl.col("Year") >= 2010)
        .collect()
    )
    polars_time = time.time() - start_polars
    timings['Polars (Lazy + Filter)'] = polars_time

    print(f"Polars (Lazy + Filter): {polars_time:.4f} seconds")

    # Pandas: Standard read
    start_pandas = time.time()
    df_pandas_initial = pd.read_parquet(parquet_path)
    df_pandas_initial = df_pandas_initial[df_pandas_initial['Year'] >= 2010]
    pandas_time = time.time() - start_pandas
    timings['Pandas (Read + Filter)'] = pandas_time

    print(f"Pandas (Read + Filter): {pandas_time:.4f} seconds")

    # Calculate speedup
    speedup = pandas_time / polars_time
    print(f"\nSpeedup: Polars is {speedup:.2f}x faster!")

    # Store for visualization
    timing_comparison = timings
    return


@app.cell
def _():
    return


@app.cell
def _(parquet_path, pl):
    # Filter for theft crimes using string operations
    theft_crimes = (
        pl.scan_parquet(parquet_path)
        .filter(pl.col("Primary Type").str.contains("THEFT"))
        .select(["Year", "Primary Type", "Arrest"])
        .collect()
    )

    print(f"Theft Crimes Found: {theft_crimes.shape[0]}")
    print(f"\nCrime Types:")
    theft_crimes.select("Primary Type").unique()
    return


@app.cell
def _():
    return


@app.cell
def _(parquet_path, pl):
    # Aggregation: crimes by year
    crimes_by_year = (
        pl.scan_parquet(parquet_path)
        .group_by("Year")
        .agg([
            pl.len().alias("Total_Crimes"),
            pl.col("Arrest").sum().alias("Arrests"),
            pl.col("Domestic").sum().alias("Domestic_Crimes")
        ])
        .sort("Year")
        .collect()
    )

    print("Crimes by Year:")
    crimes_by_year
    return (crimes_by_year,)


@app.cell
def _(crimes_by_year, px):
    # Interactive line chart
    fig_crimes_year = px.line(
        crimes_by_year.to_pandas(),
        x="Year",
        y="Total_Crimes",
        title="Total Crimes by Year (2001-2025)",
        labels={"Total_Crimes": "Number of Crimes", "Year": "Year"},
        markers=True
    )
    fig_crimes_year.update_layout(hovermode="x unified", height=500)
    fig_crimes_year
    return


@app.cell
def _():
    return


@app.cell
def _(parquet_path, pd, pl, time):
    benchmark_results = []

    # Test 1: Filtering
    start_filter = time.time()
    _ = (
        pl.scan_parquet(parquet_path)
        .filter(pl.col("Year") >= 2015)
        .collect()
    )
    polars_filter = time.time() - start_filter

    start_filter_pd = time.time()
    df_pd_bench = pd.read_parquet(parquet_path)
    _ = df_pd_bench[df_pd_bench['Year'] >= 2015]
    pandas_filter = time.time() - start_filter_pd

    benchmark_results.append({
        'Operation': 'Filter (Year >= 2015)',
        'Polars': polars_filter,
        'Pandas': pandas_filter,
        'Speedup': pandas_filter / polars_filter
    })

    # Test 2: Aggregation
    start_agg = time.time()
    _ = (
        pl.scan_parquet(parquet_path)
        .group_by("Year")
        .agg(pl.len().alias("count"))
        .collect()
    )
    polars_agg = time.time() - start_agg

    start_agg_pd = time.time()
    _ = df_pd_bench.groupby('Year').size()
    pandas_agg = time.time() - start_agg_pd

    benchmark_results.append({
        'Operation': 'Group By (Count per Year)',
        'Polars': polars_agg,
        'Pandas': pandas_agg,
        'Speedup': pandas_agg / polars_agg
    })

    # Test 3: Selection
    start_sel = time.time()
    _ = (
        pl.scan_parquet(parquet_path)
        .select(['Year', 'Primary Type', 'Arrest'])
        .collect()
    )
    polars_select = time.time() - start_sel

    start_sel_pd = time.time()
    _ = df_pd_bench[['Year', 'Primary Type', 'Arrest']]
    pandas_select = time.time() - start_sel_pd

    benchmark_results.append({
        'Operation': 'Select Columns',
        'Polars': polars_select,
        'Pandas': pandas_select,
        'Speedup': pandas_select / polars_select
    })

    benchmark_df = pl.DataFrame(benchmark_results)
    print("Benchmark Results:")
    print(benchmark_df)
    return (benchmark_df,)


@app.cell
def _(benchmark_df, px):
    # Benchmark visualization
    bench_pd_bar = benchmark_df.to_pandas()

    fig_bench_bar = px.bar(
        bench_pd_bar,
        x='Operation',
        y=['Polars', 'Pandas'],
        barmode='group',
        title='Performance Comparison: Polars vs Pandas',
        labels={'value': 'Time (seconds)', 'Operation': 'Operation'},
        height=500
    )
    fig_bench_bar.update_layout(hovermode='x unified')
    fig_bench_bar
    return


@app.cell
def _(benchmark_df, px):
    # Speedup histogram
    bench_pd_speedup = benchmark_df.to_pandas()

    fig_speedup = px.bar(
        bench_pd_speedup,
        x='Operation',
        y='Speedup',
        title='Polars Speedup over Pandas',
        labels={'Speedup': 'Speedup Factor (higher is better)', 'Operation': 'Operation'},
        color='Speedup',
        color_continuous_scale='Viridis',
        height=500
    )

    # Add a horizontal line at 1x (equal performance)
    fig_speedup.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="1x (Equal)")

    fig_speedup.update_layout(hovermode='x unified', showlegend=False)
    fig_speedup
    return


@app.cell
def _():
    return


@app.cell
def _(parquet_path, pd, pl):
    # Polars eager
    df_polars_eager = pl.read_parquet(parquet_path)
    polars_memory = df_polars_eager.estimated_size() / 1024 / 1024

    # Pandas
    df_pandas_memory = pd.read_parquet(parquet_path)
    pandas_memory = df_pandas_memory.memory_usage(deep=True).sum() / 1024 / 1024

    memory_comparison = pl.DataFrame({
        'Library': ['Polars', 'Pandas'],
        'Memory (MB)': [polars_memory, pandas_memory],
        'Efficiency': [f"{polars_memory:.2f} MB", f"{pandas_memory:.2f} MB"]
    })

    print("Memory Usage Comparison:")
    print(memory_comparison)

    memory_comparison
    return (memory_comparison,)


@app.cell
def _(memory_comparison, px):
    mem_pd = memory_comparison.to_pandas()

    fig_memory = px.bar(
        mem_pd,
        x='Library',
        y='Memory (MB)',
        title='Memory Usage: Polars vs Pandas',
        color='Library',
        text='Memory (MB)',
        height=500
    )
    fig_memory.update_traces(texttemplate='%{text:.2f} MB', textposition='outside')
    fig_memory.update_layout(showlegend=False)
    fig_memory
    return


@app.cell
def _():
    return


@app.cell
def _(parquet_path, pl):
    # Crime statistics by primary type
    crime_stats = (
        pl.scan_parquet(parquet_path)
        .group_by("Primary Type")
        .agg([
            pl.len().alias("Total_Crimes"),
            (pl.col("Arrest").sum()).alias("Arrests"),
            (pl.col("Arrest").sum() / pl.len() * 100).alias("Arrest_Rate_%")
        ])
        .sort("Total_Crimes", descending=True)
        .limit(10)
        .collect()
    )

    print("Top 10 Crime Types by Frequency:")
    crime_stats
    return (crime_stats,)


@app.cell
def _(crime_stats, px):
    crime_pd_bar = crime_stats.to_pandas()

    fig_crime_bar = px.bar(
        crime_pd_bar,
        x='Primary Type',
        y='Total_Crimes',
        title='Top 10 Crime Types in Chicago (2001-2025)',
        labels={'Total_Crimes': 'Number of Crimes', 'Primary Type': 'Crime Type'},
        height=500
    )
    fig_crime_bar.update_xaxes(tickangle=-45)
    fig_crime_bar.update_layout(hovermode='x unified')
    fig_crime_bar
    return


@app.cell
def _(crime_stats, px):
    crime_pd_scatter = crime_stats.to_pandas()

    fig_crime_scatter = px.scatter(
        crime_pd_scatter,
        x='Total_Crimes',
        y='Arrest_Rate_%',
        size='Total_Crimes',
        text='Primary Type',
        title='Crime Volume vs Arrest Rate',
        labels={'Total_Crimes': 'Total Crimes', 'Arrest_Rate_%': 'Arrest Rate (%)'},
        height=500
    )
    fig_crime_scatter.update_traces(textposition='top center', textfont=dict(size=10))
    fig_crime_scatter.update_layout(hovermode='closest')
    fig_crime_scatter
    return


@app.cell
def _():
    return


@app.cell
def _(parquet_path, pl):
    # Geographic distribution
    geo_data = (
        pl.scan_parquet(parquet_path)
        .filter(
            (pl.col("Latitude").is_not_null()) &
            (pl.col("Longitude").is_not_null())
        )
        .select(["Latitude", "Longitude", "Primary Type", "Year"])
        .filter(pl.col("Year") == 2024)  # Latest year
        .collect()
    )

    print(f"Crime incidents with coordinates in 2024: {geo_data.shape[0]}")
    print(f"\nSample locations:")
    geo_data.head(10)
    return (geo_data,)


@app.cell
def _(geo_data, px):
    # Scatter map of crimes
    geo_pd = geo_data.to_pandas()

    fig_geo_map = px.scatter_mapbox(
        geo_pd,
        lat='Latitude',
        lon='Longitude',
        title='Chicago Crime Locations (2024)',
        zoom=10,
        height=600
    )
    fig_geo_map.update_layout(mapbox_style="open-street-map")
    fig_geo_map
    return


@app.cell
def _():
    return


@app.cell
def _(parquet_path, pl):
    # Show query plan explanation
    query = (
        pl.scan_parquet(parquet_path)
        .filter(pl.col("Year") >= 2020)
        .group_by("Primary Type")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )

    print("Optimized Query Plan:")
    print(query.explain())
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
