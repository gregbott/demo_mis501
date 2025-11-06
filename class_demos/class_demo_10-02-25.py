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
    return chardet, pathlib, pl, sales_file_path


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
def _(df):
    df.glimpse()
    return


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df.tail()

    return


@app.cell
def _(df):
    df.sample(20)
    return


@app.cell
def _(df):
    df.null_count()
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df):
    df.select('Customer Name', 'Region')
    return


@app.cell
def _(df):
    df.select('State').unique().sort('State')
    return


@app.cell
def _(df, pl):
    df.select(pl.col('Customer Name','Sales'))
    return


@app.cell
def _(df, pl):
    df.select(pl.exclude('Order ID', 'Product ID'))
    return


@app.cell
def _(df, pl):
    df.select(pl.col(pl.Decimal)).describe()
    return


@app.cell
def _(df, pl):
    df.select(pl.col(pl.String))
    return


@app.cell
def _(df, pl):
    _df = df.select(
        pl.col('Customer Name').alias('customer'),
        pl.col('Sales').alias('revenue')    
    )
    _df.glimpse()
    return


@app.cell
def _(df):
    df.glimpse()
    return


@app.cell
def _(df, pl):
    df.select(
        pl.col('Customer Name'),
        pl.col('Sales'),
        (pl.col('Sales') * pl.col('Quantity')).alias('total_revenue')
    )
    return


@app.cell
def _(df, pl):
    df.filter(pl.col('Region') == 'South')
    df.filter(pl.col('Ship Mode') == 'First Class')
    return


@app.cell
def _(df, pl):
    df.filter(pl.col('Sales') > 1000)
    return


@app.cell
def _(df, pl):
    df.filter(pl.col('Customer Name').str.contains('Tracy'))
    return


@app.cell
def _(df, pl):
    df.filter(
        ((pl.col('Ship Mode')=='First Class') |
        (pl.col('Ship Mode')=='Second Class')) &
        (pl.col('Region').is_in(['South', 'West']))   
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
