import marimo

__generated_with = "0.16.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import pandas as pd
    import plotly.express as px
    import time
    return pd, pl, px, time


@app.cell
def _():
    # Parquet file path (constant throughout notebook)
    parquet_path = "./data/chicago_crime_2001_2025.parquet"
    return (parquet_path,)


@app.cell
def _(parquet_path, pl):
    df_eager = pl.read_parquet(parquet_path)
    df_eager.head()
    return (df_eager,)


@app.cell
def _(df_eager):
    df_eager.shape
    return


@app.cell
def _(df_eager):
    df_eager.estimated_size()
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

    # print(lazy_query.explain)
    df_filtered = lazy_query.collect()
    print(df_filtered.shape)
    df_filtered.head()
    return


@app.cell
def _(parquet_path, pd, pl, time):
    timings = {}

    start_polars = time.time()
    df_polars = (
        pl.scan_parquet(parquet_path)
        .filter(pl.col("Year") >= 2010)
        .collect()
    )
    polars_time = time.time() - start_polars
    timings['Polars'] = polars_time

    print(f'Polars: {polars_time:.4f} seconds')

    start_pandas = time.time()
    df_pandas_initial = pd.read_parquet(parquet_path)
    df_pandas_initial = df_pandas_initial[df_pandas_initial['Year'] >= 2010]
    pandas_time = time.time() - start_pandas
    timings['Pandas'] = pandas_time
    print(f'Pandas: {pandas_time:.4f} seconds')

    speedup = pandas_time / polars_time
    print(f" Polars is {speedup:.2f}x faster!")
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


if __name__ == "__main__":
    app.run()
