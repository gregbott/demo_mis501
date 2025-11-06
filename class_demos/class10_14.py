import marimo

__generated_with = "0.16.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    return (pl,)


@app.cell
def _():
    state_abbrev = {
        "Alabama": "AL",
        "Alaska": "AK",
        "Arizona": "AZ",
        "Arkansas": "AR",
        "California": "CA",
        "Californya":"CA",
        "Colorado": "CO",
        "Connecticut": "CT",
        "District of Columbia":"DC",
        "Delaware": "DE",
        "Florida": "FL",
        "Georgia": "GA",
        "Hawaii": "HI",
        "Idaho": "ID",
        "Illinois": "IL",
        "Indiana": "IN",
        "Iowa": "IA",
        "Kansas": "KS",
        "Kentucky": "KY",
        "Louisiana": "LA",
        "Maine": "ME",
        "Maryland": "MD",
        "Massachusetts": "MA",
        "Michigan": "MI",
        "Minnesota": "MN",
        "Mississippi": "MS",
        "Missouri": "MO",
        "Montana": "MT",
        "Nebraska": "NE",
        "Nevada": "NV",
        "New Hampshire": "NH",
        "New Jersey": "NJ",
        "New Mexico": "NM",
        "New York": "NY",
        "North Carolina": "NC",
        "North Dakota": "ND",
        "Ohio": "OH",
        "Oklahoma": "OK",
        "Oregon": "OR",
        "Pennsylvania": "PA",
        "Rhode Island": "RI",
        "South Carolina": "SC",
        "South Dakota": "SD",
        "Tennessee": "TN",
        "Texas": "TX",
        "Utah": "UT",
        "Vermont": "VT",
        "Virginia": "VA",
        "Washington": "WA",
        "West Virginia": "WV",
        "Wisconsin": "WI",
        "Wyoming": "WY"
    }
    return (state_abbrev,)


@app.cell
def _():
    state_corrections = {'Californya':'CA'}
    return


@app.cell
def _(pl, state_abbrev):
    df_sales2 = pl.read_csv('data/stores_sales_forecasting2.csv',
                           encoding='cp1252')
    df_sales2 = df_sales2.with_columns(
        pl.col('State')
            .replace_strict(state_abbrev, default=pl.col('State')),
        pl.coalesce([
            pl.col('Order Date').str.strptime(pl.Date, format="%m/%d/%Y", strict=False), # # Consider creating a function for this that includes all common date formats being careful to discren non-US formats
            pl.col('Order Date').str.strptime(pl.Date, format="%B %d, %Y", strict=False)
        ]).alias('Order Date Parsed'),        
                                      )
    return (df_sales2,)


@app.cell
def _(df_sales2, pl):
    parse_failures = df_sales2.filter(
        pl.col("Order Date Parsed").is_null() & 
        pl.col("Order Date").is_not_null()  # Original had a value
    )

    print(f"Rows that failed to parse: {len(parse_failures)}")
    print(parse_failures.select(["Order Date", "Order Date Parsed"]))
    print(parse_failures.select(["Order Date", "Order Date Parsed"]))
    return


@app.cell
def _(df_sales2):
    df_sales2[''].unique().sort()
    return


@app.cell
def _(df_sales2):
    df_sales2.glimpse()
    return


@app.cell
def _(pl):
    ia_df = pl.read_csv('data/Iowa_Liquor_Sales-26M.csv.gz',
                        schema_overrides={'Zip Code':pl.String,
                                         'Item Number':pl.String})
    return (ia_df,)


@app.cell
def _(ia_df):
    ia_df.shape
    return


@app.cell
def _(ia_df):
    ia_df.glimpse()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
