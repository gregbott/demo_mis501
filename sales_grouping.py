import marimo

__generated_with = "0.16.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    sales_url = 'https://raw.githubusercontent.com/gregbott/data_501/master/stores_sales_forecasting.parquet'
    return pl, sales_url


@app.cell
def _(pl, sales_url):
    df = pl.read_parquet(sales_url)
    return (df,)


@app.cell
def _(df):
    df.glimpse()
    return


@app.cell
def _(df, pl):
    regional_sales = df.group_by('Region','Sub-Category').agg([
        pl.col('Sales').sum().round(0).alias('total_sales'),
        pl.col('Order ID').count().alias('count')
    ]).sort(['Sub-Category','Region'])
    return (regional_sales,)


@app.cell
def _(df, pl):
    # Select returns a 
    avg_sale = df.select(
        pl.col('Sales').mean().round(0).alias('average_sale'))

    avg_sale
    return


@app.cell
def _(df):
    df['Sales'].mean()
    return


@app.cell
def _(df, pl):
    avg_sale_by_sub = df.group_by('Sub-Category').agg([
        pl.col('Sales').mean().round(0).alias('avg')
    ])
    avg_sale_by_sub
    return


@app.cell
def _():
    return


@app.cell
def _(regional_sales):
    regional_sales
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
