import marimo

__generated_with = "0.16.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    return (pl,)


@app.cell
def _(pl):
    df = pl.read_csv('data/stores_sales_forecasting.csv',
                    encoding='MacRoman')


    return (df,)


@app.cell
def _(df):
    df.columns
    return


@app.cell
def _(df, pl):
    regional_sales = df.group_by("Region").agg([
                     pl.col('Sales')   
    ])
    return


if __name__ == "__main__":
    app.run()
