import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    return (pl,)


@app.cell
def _(pl):
    sales_df = pl.read_csv('stores_sales_forecasting.csv')
    return (sales_df,)


@app.cell
def _(sales_df):
    sales_df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
