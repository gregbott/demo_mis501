import marimo

__generated_with = "0.16.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    sales_url = 'https://raw.githubusercontent.com/gregbott/data_501/master/stores_sales_forecasting.parquet'
    netflix_url = 'https://raw.githubusercontent.com/gregbott/data_501/master/netflix_titles.csv'
    return netflix_url, pl, sales_url


@app.cell
def _(pl, sales_url):
    df = pl.read_parquet(sales_url)
    return (df,)


@app.cell
def _(netflix_url, pl):
    ndf = pl.read_csv(netflix_url, encoding='latin-1')
    ndf.shape
    return


@app.cell
def _(df):
    df.shape
    return


@app.cell
def _(df):
    df.glimpse()
    return


@app.cell
def _(df):
    df.null_count()
    return


@app.cell
def _(df, pl):
    avg_sale = df.select(
        pl.col('Sales').mean().round(0).alias('average_sale'),
        pl.col('Order ID').count().alias('count')
    )
    avg_sale
    return


@app.cell
def _(df):
    round(df['Sales'].mean(),0)
    return


@app.cell
def _(df, pl):
    regional_sales = df.group_by('Region','Sub-Category').agg([
        pl.col('Sales').sum().round(0).alias(('total_sales')),
        pl.col('Order ID').count().alias('count')
    ]).sort('Sub-Category','total_sales',descending=[False,True])
    regional_sales
    return


@app.cell
def _(df):
    df['Sub-Category'].value_counts()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
