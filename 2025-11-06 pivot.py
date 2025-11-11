import marimo

__generated_with = "0.16.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    from datetime import datetime, timedelta
    import random
    return datetime, mo, np, pl, random, timedelta


@app.cell
def _(datetime, np, pl, random, timedelta):
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Define parameters
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2023, 12, 31)
    num_days = (end_date - start_date).days

    regions = ["North America", "Europe", "Asia", "South America"]
    products = ["Laptop", "Smartphone", "Tablet", "Headphones", "Smartwatch"]
    sales_reps = ["Alice Johnson", "Bob Smith", "Carol White", "David Brown",
                  "Emma Davis", "Frank Wilson", "Grace Lee", "Henry Taylor"]

    # Base prices for products (in USD)
    base_prices = {
        "Laptop": 1200,
        "Smartphone": 800,
        "Tablet": 500,
        "Headphones": 150,
        "Smartwatch": 350
    }

    # Generate approximately 10,000 transactions over 5 years
    num_transactions = 10000

    dates = [start_date + timedelta(days=random.randint(0, num_days))
             for _ in range(num_transactions)]

    data = {
        "date": dates,
        "region": [random.choice(regions) for _ in range(num_transactions)],
        "product": [random.choice(products) for _ in range(num_transactions)],
        "sales_rep": [random.choice(sales_reps) for _ in range(num_transactions)],
        "quantity": np.random.randint(1, 20, num_transactions),
    }

    # Create DataFrame
    sales_df = pl.DataFrame(data)

    # Add unit price with some random variation
    sales_df = sales_df.with_columns([
        pl.col("product").map_elements(
            lambda x: base_prices[x] * random.uniform(0.9, 1.1),
            return_dtype=pl.Float64
        ).alias("unit_price")
    ])

    # Calculate total sales
    sales_df = sales_df.with_columns([
        (pl.col("quantity") * pl.col("unit_price")).alias("total_sales")
    ])

    # Add year, quarter, and month columns for easier analysis
    sales_df = sales_df.with_columns([
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.quarter().alias("quarter"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.strftime("%B").alias("month_name")
    ])

    # Sort by date
    sales_df = sales_df.sort("date")

    sales_df.head(10)
    return (sales_df,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Basic Pivot Table Syntax

    The basic syntax for creating a pivot table in Polars is:

    ```python
    df.pivot(
        values="column_to_aggregate",
        index="row_dimension",
        columns="column_dimension",
        aggregate_function="sum"  # or "mean", "count", "max", "min", etc.
    )
    ```

    **Parameters:**
    - `values`: The column(s) containing values to aggregate
    - `index`: Column(s) to use as row labels
    - `columns`: Column(s) to use as column labels
    - `aggregate_function`: How to combine multiple values (sum, mean, count, etc.)
    """
    )
    return


@app.cell
def _(pl, sales_df):
    pivot_region_year = sales_df.pivot(
        values="total_sales",
        index="region",
        columns="year",
        aggregate_function="sum"
    ).sort("region")

    # Format numbers for better readability
    pivot_region_year_display = pivot_region_year.with_columns([
        pl.col(str(year)).round(2) for year in range(2019, 2024)
    ])

    pivot_region_year_display
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Chicago data
    - Homicides
    - By district
    - Last three years (2022-2024)
    """
    )
    return


if __name__ == "__main__":
    app.run()
