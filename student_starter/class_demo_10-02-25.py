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
def _(chardet, pathlib, sales_file_path):
    with open(pathlib.Path(sales_file_path), 'rb') as file:
        # Read first 100,000 bytes or entire file if smaller
        raw_data = file.read(100000)

    # Detect encoding
    results = chardet.detect(raw_data)
    results
    return


@app.cell
def _(pl):
    sales_file_path = './stores_sales_forecasting.csv'

    df = (pl.read_csv(sales_file_path,
                    encoding='cp1252',
                    schema_overrides={
                            'Row ID':pl.UInt16,
                            'Postal Code':pl.String,
                            'Quantity':pl.UInt16,
                            'Sales':pl.Float32,
                            'Profit':pl.Float32,
                            'Discount':pl.Float32
                            }
                     ).drop(['Country', 'Row ID'])
         )
    return (sales_file_path,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
