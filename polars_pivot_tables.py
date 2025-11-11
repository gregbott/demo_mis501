import marimo

__generated_with = "0.16.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
    # Pivot Tables with Polars

    This notebook demonstrates how to create and work with pivot tables using Polars,
    a lightning-fast DataFrame library for Python. We'll use 5 years of fictional
    electronics sales data to explore various pivot table operations.

    ## What is a Pivot Table?

    A pivot table is a data summarization tool that allows you to:
    - Aggregate data by grouping it along different dimensions
    - Reshape data from long to wide format
    - Calculate summary statistics for different categories
    - Analyze relationships between categorical variables

    In Polars, pivot tables are created using the `.pivot()` method, which is both
    powerful and efficient.
    """
    )
    return


@app.cell
def _():
    import polars as pl
    import numpy as np
    from datetime import datetime, timedelta
    import random
    return datetime, np, pl, random, timedelta


@app.cell
def _(mo):
    mo.md(
        """
    ## Generating Sample Data

    Let's create 5 years of fictional electronics sales data with:
    - Multiple sales regions (North America, Europe, Asia, South America)
    - Various product categories (Laptops, Smartphones, Tablets, Headphones, Smartwatches)
    - Sales representatives
    - Transaction details (quantity, unit price, total sales)
    """
    )
    return


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

    # Create DataFrame with all transformations chained together
    sales_df = (
        pl.DataFrame(data)
        # Add unit price with some random variation
        .with_columns([
            pl.col("product").map_elements(
                lambda x: base_prices[x] * random.uniform(0.9, 1.1),
                return_dtype=pl.Float64
            ).alias("unit_price")
        ])
        # Calculate total sales
        .with_columns([
            (pl.col("quantity") * pl.col("unit_price")).alias("total_sales")
        ])
        # Add year, quarter, and month columns for easier analysis
        .with_columns([
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.quarter().alias("quarter"),
            pl.col("date").dt.month().alias("month"),
            pl.col("date").dt.strftime("%B").alias("month_name")
        ])
        # Sort by date
        .sort("date")
    )

    sales_df.head(10)
    return (sales_df,)


@app.cell
def _(mo, sales_df):
    mo.md(
        f"""
    ### Dataset Overview

    Our dataset contains **{len(sales_df):,}** transactions spanning from
    {sales_df['date'].min()} to {sales_df['date'].max()}.

    **Column Descriptions:**
    - `date`: Transaction date
    - `region`: Sales region
    - `product`: Product type
    - `sales_rep`: Sales representative
    - `quantity`: Number of units sold
    - `unit_price`: Price per unit (with slight variations)
    - `total_sales`: Total revenue (quantity × unit_price)
    - `year`, `quarter`, `month`, `month_name`: Temporal dimensions
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
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
def _(mo):
    mo.md(
        """
    ## Example 1: Sales by Region and Year

    Let's create a pivot table showing total sales by region for each year.
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
        """
    **Interpretation:** This table shows how sales evolved across different regions
    over the 5-year period. Each cell represents the total sales (in dollars) for
    a specific region in a specific year.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Example 2: Product Sales by Region

    Now let's see which products perform best in each region.
    """
    )
    return


@app.cell
def _(pl, sales_df):
    pivot_product_region = sales_df.pivot(
        values="total_sales",
        index="product",
        columns="region",
        aggregate_function="sum"
    ).sort("product")

    # Format for display
    pivot_product_region_display = pivot_product_region.with_columns([
        pl.col(col).round(2)
        for col in ["Asia", "Europe", "North America", "South America"]
    ])

    pivot_product_region_display
    return


@app.cell
def _(mo):
    mo.md(
        """
    **Insight:** This pivot table helps identify regional product preferences and
    market opportunities.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Example 3: Average Sales by Product and Quarter

    Let's analyze seasonal trends by looking at average transaction values.
    """
    )
    return


@app.cell
def _(pl, sales_df):
    pivot_product_quarter = sales_df.pivot(
        values="total_sales",
        index="product",
        columns="quarter",
        aggregate_function="mean"
    ).sort("product")

    # Rename columns for clarity and round values
    pivot_product_quarter_display = pivot_product_quarter.rename({
        "1": "Q1_avg",
        "2": "Q2_avg",
        "3": "Q3_avg",
        "4": "Q4_avg"
    }).with_columns([
        pl.col(f"Q{q}_avg").round(2) for q in range(1, 5)
    ])

    pivot_product_quarter_display
    return


@app.cell
def _(mo):
    mo.md(
        """
    **Note:** Using `aggregate_function="mean"` gives us the average transaction
    value instead of the total, which is useful for understanding typical order sizes.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Example 4: Transaction Count by Sales Rep and Year

    Let's see how active each sales representative was over time.
    """
    )
    return


@app.cell
def _(sales_df):
    pivot_rep_year = sales_df.pivot(
        values="total_sales",  # We're counting, but need to specify a value column
        index="sales_rep",
        columns="year",
        aggregate_function="count"
    ).sort("sales_rep")

    pivot_rep_year
    return


@app.cell
def _(mo):
    mo.md(
        """
    **Usage:** The `count` aggregate function tells us how many transactions each
    sales rep completed, regardless of the sales amount.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Example 5: Multiple Aggregations

    We can also create summary statistics by combining pivot tables with groupby operations.
    Let's calculate total quantity sold by product and region.
    """
    )
    return


@app.cell
def _(pl, sales_df):
    pivot_quantity = sales_df.pivot(
        values="quantity",
        index="product",
        columns="region",
        aggregate_function="sum"
    ).sort("product")

    pivot_quantity_display = pivot_quantity.with_columns([
        pl.col(col).cast(pl.Int64)
        for col in ["Asia", "Europe", "North America", "South America"]
    ])

    pivot_quantity_display
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Example 6: Monthly Sales Trend for a Specific Product

    Let's zoom in on a single product to see monthly patterns.
    """
    )
    return


@app.cell
def _(pl, sales_df):
    laptop_sales = sales_df.filter(pl.col("product") == "Laptop")

    pivot_laptop_monthly = laptop_sales.pivot(
        values="total_sales",
        index="year",
        columns="month",
        aggregate_function="sum"
    ).sort("year")

    # Fill null values with 0 (months with no sales)
    pivot_laptop_monthly_display = pivot_laptop_monthly.fill_null(0)

    pivot_laptop_monthly_display
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Advanced Techniques

    ### 1. Multiple Index Columns

    You can use multiple columns for the index to create hierarchical groupings:

    ```python
    df.pivot(
        values="total_sales",
        index=["region", "year"],  # Multiple index columns
        columns="product",
        aggregate_function="sum"
    )
    ```

    ### 2. Handling Missing Values

    Pivot tables may contain null values when certain combinations don't exist
    in the data. Use `.fill_null()` to replace them:

    ```python
    pivot_table = df.pivot(...).fill_null(0)
    ```

    ### 3. Post-Pivot Calculations

    After creating a pivot table, you can add calculated columns:

    ```python
    pivot_table = pivot_table.with_columns([
        (pl.col("2023") - pl.col("2022")).alias("year_over_year_change")
    ])
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Example 7: Complex Pivot with Multiple Indices

    Let's create a more complex pivot showing sales by region and product,
    broken down by year.
    """
    )
    return


@app.cell
def _(pl, sales_df):
    pivot_complex = sales_df.pivot(
        values="total_sales",
        index=["region", "product"],
        columns="year",
        aggregate_function="sum"
    ).sort(["region", "product"])

    pivot_complex_display = pivot_complex.with_columns([
        pl.col(str(year)).round(2) for year in range(2019, 2024)
    ])

    pivot_complex_display
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Key Takeaways

    **Polars Pivot Table Advantages:**
    1. **Performance**: Polars is built on Rust and optimized for speed
    2. **Memory Efficiency**: Handles large datasets efficiently
    3. **Clean Syntax**: Intuitive API similar to pandas but more explicit
    4. **Lazy Evaluation**: Can be combined with lazy operations for better performance

    **Common Aggregate Functions:**
    - `"sum"`: Total of values
    - `"mean"`: Average of values
    - `"count"`: Number of occurrences
    - `"min"`: Minimum value
    - `"max"`: Maximum value
    - `"median"`: Median value
    - `"first"`: First value encountered
    - `"last"`: Last value encountered

    **Best Practices:**
    1. Choose appropriate aggregate functions for your data type
    2. Consider using `.fill_null()` for missing combinations
    3. Sort your results for better readability
    4. Use meaningful column names
    5. Round numerical values for cleaner presentation
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Summary Statistics

    Let's conclude with some overall statistics from our dataset.
    """
    )
    return


@app.cell
def _(mo, pl, sales_df):
    total_revenue = sales_df.select(pl.col("total_sales").sum()).item()
    total_units = sales_df.select(pl.col("quantity").sum()).item()
    avg_transaction = sales_df.select(pl.col("total_sales").mean()).item()
    _num_transactions = len(sales_df)

    mo.md(
        f"""
        **Overall Sales Performance (2019-2023):**

        - **Total Revenue**: ${total_revenue:,.2f}
        - **Total Units Sold**: {total_units:,}
        - **Average Transaction Value**: ${avg_transaction:,.2f}
        - **Total Transactions**: {_num_transactions:,}
        """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
