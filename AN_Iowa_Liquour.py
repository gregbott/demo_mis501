import marimo

__generated_with = "0.16.3"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""## Task 1.0: Import modules and initialize variables""")
    return


@app.cell
def _():
    import marimo as mo
    import polars as pl
    return mo, pl


@app.cell
def _(mo):
    mo.md(r"""## Task 2.0: Ingest data""")
    return


@app.cell
def _(pl):
    ia_df = pl.read_csv('./data/Iowa_Liquor_Sales-26M.csv.gz',
                        schema_overrides={'Zip Code':pl.String,
                                            'Item Number':pl.String}
                        )
    return (ia_df,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Get the size of the DataFrame and store it in a comment
    Answer: 8349497028

    Convert this number to megabytes. How large is the DataFrame in megabytes?
    Answer: 7,963.48 MB
    """
    )
    return


@app.cell
def _(ia_df):
    ia_df.estimated_size()
    return


@app.cell
def _(pl):
    # If you want to specify certain dtypes while inferring others
    lazy_df_with_dtypes = pl.scan_csv(
        './data/Iowa_Liquor_Sales-26M.csv.gz',
        # dtypes={"id": pl.Int64, "date": pl.Utf8},  # Specify some types
        infer_schema_length=1000  # Scan first 1000 rows to infer schema
    )

    # You can also get column names only
    ia_names = lazy_df_with_dtypes.collect_schema().items()

    # Or check specific column types
    # print(lazy_df_with_dtypes.collect_schema().dtypes()
    return (ia_names,)


@app.cell
def _(ia_names):
    dict(ia_names)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
