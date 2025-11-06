import marimo

__generated_with = "0.16.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import pathlib
    import chardet

    sales_file_path = './stores_sales_forecasting.csv'
    sales2_file_path = './stores_sales_forecasting2.csv'
    return chardet, mo, pathlib, pl, sales2_file_path, sales_file_path


@app.cell
def _(chardet, pathlib, sales_file_path):
    with open(pathlib.Path(sales_file_path), 'rb') as file:
        # Read first 100,000 bytes or entire file if smaller
        raw_data = file.read(100000)

    # Detect encoding
    results = chardet.detect(raw_data)
    results
    return


@app.cell
def _(pl, sales_file_path):
    df = (pl.read_csv(sales_file_path,
                    encoding='cp1252',
                    schema_overrides={
                            'Row ID':pl.UInt16,
                            'Postal Code':pl.String,
                            'Quantity':pl.UInt16,
                            'Sales':pl.Decimal(scale=2),
                            'Profit':pl.Decimal(scale=2),
                            'Discount':pl.Decimal(scale=2)
                            }
                     ).drop(['Country', 'Row ID'])
         )
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""# Examine the data""")
    return


@app.cell
def _(df):
    df.shape
    return


@app.cell
def _():
    # Examine: df.head(), tail(), sample()
    return


@app.cell
def _(df):
    # Examine: check data types with .glimpse()
    df.glimpse()
    return


@app.cell
def _(df):
    # Examine: get null counts
    df.null_count()
    return


@app.cell
def _(mo):
    mo.md(r"""# Selecting Data""")
    return


@app.cell
def _(df):
    # Select single column
    df.select('Customer Name')
    return


@app.cell
def _(df):
    # Select multiple columns
    df.select('Order ID', 'Customer Name', 'Sales')
    return


@app.cell
def _(df, pl):
    # Select using pl.col()
    df.select(pl.col('Customer Name'), pl.col('Sales'))
    return


@app.cell
def _(df, pl):
    # Select all columns except some
    df.select(pl.exclude('Order ID', 'Product ID'))
    return


@app.cell
def _(df, pl):
    # Select by data type
    df.select(pl.col(pl.Decimal))  # All decimal columns
    return


@app.cell
def _(df, pl):
    df.select(pl.col(pl.Utf8))     # All string columns
    return


@app.cell
def _(df, pl):
    # Select with wildcards
    df.select(pl.col('^Customer*'))  # All columns starting with "Customer"
    return


@app.cell
def _(df, pl):
    df.select(pl.col('.*ID$'))        # All columns ending with "ID"
    return


@app.cell
def _(mo):
    mo.md(r"""# Select and Transform""")
    return


@app.cell
def _(df, pl):
    # Select and rename
    df.select(
        pl.col('Customer Name').alias('customer'),
        pl.col('Sales').alias('revenue')
    )

    # Select and calculate
    df.select(
        pl.col('Product Name'),
        pl.col('Sales'),
        (pl.col('Sales') * pl.col('Quantity')).alias('total_revenue')
    )

    # Select with conditional logic
    df.select(
        pl.col('Customer Name'),
        pl.when(pl.col('Profit') > 0)
          .then(pl.lit('Profitable'))
          .otherwise(pl.lit('Loss'))
          .alias('profit_status')
    )

    # Select with aggregations
    df.select(
        pl.col('Sales').sum().alias('total_sales'),
        pl.col('Profit').mean().alias('avg_profit'),
        pl.col('Order ID').n_unique().alias('unique_orders')
    )
    return


@app.cell
def _(mo):
    mo.md(r"""# Basic Filtering""")
    return


@app.cell
def _(df, pl):
    # Filter by exact match
    df.filter(pl.col('Region') == 'South')
    df.filter(pl.col('Ship Mode') == 'First Class')
    df.select('Order Date','Customer Name','Region','Ship Mode').sort('Customer Name')
    return


@app.cell
def _(df, pl):
    # Filter by numeric comparison
    df.filter(pl.col('Sales') > 1000)
    df.filter(pl.col('Profit') < 0)  # Loss-making orders
    df.filter(pl.col('Discount') >= 0.20)

    # Filter by string contains
    df.filter(pl.col('Customer Name').str.contains('Tracy'))
    df.filter(pl.col('Product Name').str.contains('Chair'))

    # Filter by string starts/ends with
    df.filter(pl.col('State').str.starts_with('C'))  # California, etc.
    df.filter(pl.col('Product ID').str.ends_with('577'))
    return


@app.cell
def _(mo):
    mo.md(r"""# Advanced Filtering""")
    return


@app.cell
def _(df, pl):
    # Multiple conditions with AND
    df.filter(
        (pl.col('Region') == 'South') & 
        (pl.col('Sales') > 500)
    )

    # Multiple conditions with OR
    df.filter(
        (pl.col('Ship Mode') == 'First Class') | 
        (pl.col('Ship Mode') == 'Second Class')
    )

    # Filter using .is_in()
    df.filter(pl.col('Region').is_in(['South', 'West']))
    df.filter(pl.col('Category').is_in(['Furniture', 'Technology']))

    # Filter for null values
    df.filter(pl.col('Segment').is_null())
    df.filter(pl.col('Segment').is_not_null())

    # Filter by negative profit
    df.filter(pl.col('Profit') < 0)

    # Filter by discount range
    df.filter(pl.col('Discount').is_between(0.20, 0.50))

    # Complex condition
    df.filter(
        (pl.col('Sales') > 1000) &
        (pl.col('Discount') > 0) &
        (pl.col('Profit') > 0)
    )
    return


@app.cell
def _(mo):
    mo.md(r"""# Pattern-based Filters""")
    return


@app.cell
def _(df, pl):
    # Inconsistent state names (California vs Californya)
    df.filter(pl.col('State').str.contains('Califor'))

    # Orders with high discounts
    df.filter(pl.col('Discount') >= 0.40)

    # Date-based filtering (after parsing dates)
    df.filter(pl.col('Order Date').str.contains('2016'))

    # Customer IDs starting with specific patterns
    df.filter(pl.col('Customer ID').str.starts_with('CG'))

    # Products with specific keywords
    df.filter(
        pl.col('Product Name').str.to_lowercase().str.contains('chair')
    )
    return


@app.cell
def _(mo):
    mo.md(r"""# Statistical Filters""")
    return


@app.cell
def _(df, pl):
    # Top 10% of sales
    threshold = df.select(pl.col('Sales').quantile(0.90)).item()
    df.filter(pl.col('Sales') > threshold)

    # Orders above average profit
    avg_profit = df.select(pl.col('Profit').mean()).item()
    df.filter(pl.col('Profit') > avg_profit)

    # High quantity orders
    df.filter(pl.col('Quantity') >= 5)
    return


@app.cell
def _(mo):
    mo.md(r"""# Business-type queries""")
    return


@app.cell
def _(df, pl):
    # Loss-making orders by customer
    df.filter(pl.col('Profit') < 0).select(
        'Customer Name',
        'Product Name',
        'Sales',
        'Profit',
        'Discount'
    ).sort('Profit')

    # High-discount, low-profit orders (potential problem)
    df.filter(
        (pl.col('Discount') >= 0.40) &
        (pl.col('Profit') < 50)
    ).select('Order ID', 'Product Name', 'Discount', 'Profit')

    # Regional performance
    df.group_by('Region').agg(
        pl.col('Sales').sum().alias('total_sales'),
        pl.col('Profit').sum().alias('total_profit')
    ).sort('total_sales', descending=True)

    # Customer segment analysis
    df.filter(pl.col('Segment').is_not_null()).group_by('Segment').agg(
        pl.col('Sales').mean().alias('avg_sales'),
        pl.count().alias('order_count')
    )

    # Products with consistent losses
    df.group_by('Product Name').agg(
        pl.col('Profit').sum().alias('total_profit'),
        pl.count().alias('times_ordered')
    ).filter(pl.col('total_profit') < 0).sort('total_profit')
    return


@app.cell
def _(mo):
    mo.md(r"""# Combining SELECT and FILTER""")
    return


@app.cell
def _(df, pl):
    # Filter then select
    df.filter(pl.col('Region') == 'South').select('Customer Name', 'Sales', 'Profit')

    # Select then filter (on derived column)
    df.select(
        pl.col('Customer Name'),
        pl.col('Sales'),
        (pl.col('Sales') - pl.col('Profit')).alias('cost')
    ).filter(pl.col('cost') > 500)

    # Complex example: High-value unprofitable orders
    df.filter(
        (pl.col('Sales') > 1000) &
        (pl.col('Profit') < 0)
    ).select(
        'Order ID',
        'Customer Name',
        'Product Name',
        'Sales',
        'Profit',
        'Discount'
    ).sort('Profit')
    return


@app.cell
def _(df):
    # Select a single column
    df.select('Customer Name', "Sub-Category").unique().sort('Customer Name','Sub-Category')
    return


@app.cell
def _(mo):
    mo.md(r"""# Cleaning data""")
    return


@app.cell
def _(pl, sales2_file_path):
    df2 = (pl.read_csv(sales2_file_path,
                    encoding='cp1252',
                    schema_overrides={
                            'Row ID':pl.UInt16,
                            'Postal Code':pl.String,
                            'Quantity':pl.UInt16,
                            # 'Sales':pl.Decimal(scale=2), # Error b/c of string
                            'Profit':pl.Decimal(scale=2),
                            'Discount':pl.Decimal(scale=2)
                            }
                     ).drop(['Country', 'Row ID'])
         )
    df2.glimpse()
    return (df2,)


@app.cell
def _(df2):
    # Find inconsistent state names
    df2.select('State').unique().sort('State')
    return


@app.cell
def _(df2):
    # Find date format inconsistencies
    df2.select('Order Date').unique().head(20)
    return


@app.cell
def _(df2, pl):
    # Customers with missing segment
    df2.filter(pl.col('Segment').is_null()).select('Customer Name', 'Customer ID')
    return


@app.cell
def _(df2, pl):
    # Find columns that fail to cast
    problem_sales = df2.with_columns(
        pl.col('Sales').str.strip_chars().cast(pl.Float64, strict=False).alias('Sales_test')
    ).filter(
        pl.col('Sales_test').is_null() & pl.col('Sales').is_not_null()
    ).select('Sales')
    print(problem_sales)
    return


@app.cell
def _(df2, pl):
    df2_clean = (df2
                .with_columns(pl.col('Sales')
                .str.replace_all(r'[^\d.]', '')
                .cast(pl.Decimal(scale=2)))
                # .drop('Row ID','Country')
                )
    return (df2_clean,)


@app.cell
def _(df2_clean):
    df2_clean.glimpse()
    return


@app.cell
def _(df2, pl):
    # Try to cast and find what returns null
    df_with_test = df2.with_columns(
        pl.col('Order Date').str.to_date(format='%m/%d/%Y', strict=False).alias('Order_Date_parsed')
    )

    # Find rows where parsing failed
    failed_dates = df_with_test.filter(
        pl.col('Order_Date_parsed').is_null() & pl.col('Order Date').is_not_null()
    ).select('Order Date')

    print(failed_dates)
    return (df_with_test,)


@app.cell
def _(df_with_test):
    df_with_test.glimpse()
    return


@app.cell
def _():
    # df = df.with_columns(
    #     pl.coalesce(
    #         pl.col('date_col').str.to_date(format='%m/%d/%Y', strict=False),
    #         pl.col('date_col').str.to_date(format='%B, %d %Y', strict=False)
    #     ).alias('date_col')
    # )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
