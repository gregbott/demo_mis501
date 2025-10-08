import marimo

__generated_with = "0.16.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import pathlib
    import chardet
    return chardet, pathlib, pl


@app.cell
def _(pl):
    grades_df = pl.read_excel('data/grades.xlsx')
    return (grades_df,)


@app.cell
def _(grades_df):
    grades_df.shape
    return


@app.cell
def _(grades_df):
    grades_df.head()
    return


@app.cell
def _(grades_df):
    grades_df.describe()
    return


@app.cell
def _(grades_df):
    grades_df.tail(50)
    return


@app.cell
def _(grades_df):
    grades_df.estimated_size()
    return


@app.cell
def _(chardet, pathlib):
    with open(pathlib.Path('data/stores_sales_forecasting.csv'), 'rb') as file:
        # Read first 100,000 bytes or entire file if smaller
        raw_data = file.read()

    # Detect encoding
    results = chardet.detect(raw_data)
    results
    return


@app.cell
def _(pl):
    df = (pl.read_csv('data/stores_sales_forecasting2.csv',
                                schema_overrides={'Postal Code':pl.String,
                                                  'Profit':pl.Float32,
                                                  'Discount':pl.Float32,
                                                  'Sales':pl.Float32,
                                                  'Quantity':pl.Int16
                                                 },
                                encoding='cp1252',
                             ).drop(["Country", "Row ID"])
                  )
    return (df,)


@app.cell
def _(df, pl):
    # Keep original string column while creating parsed version
    df_sales = df.with_columns([
        pl.col("Order Date")
        .str.strptime(pl.Date, format="%m/%d/%Y", strict=False)
        .alias("Order Date Parsed")
    ])

    # Filter for nulls in parsed column (these are failures)
    parse_failures = df_sales.filter(
        pl.col("Order Date Parsed").is_null() & 
        pl.col("Order Date").is_not_null()  # Original had a value
    )

    print(f"Rows that failed to parse: {len(parse_failures)}")
    print(parse_failures.select(["Order Date", "Order Date Parsed"]))
    print(parse_failures.select(["Order Date", "Order Date Parsed"]))
    return


@app.cell
def _(employee_df):
    employee_df.estimated_size()
    return


@app.cell
def _(employee_df):
    employee_df.head()
    return


@app.cell
def _(employee_df):
    employee_df.glimpse()
    return


@app.cell
def _(employee_df):
    employee_df.null_count()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
