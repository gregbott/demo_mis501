import marimo

__generated_with = "0.16.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import pathlib
    from datetime import datetime, date

    sales_file_path = 'data/stores_sales_forecasting2.csv'
    return pl, sales_file_path


@app.cell
def _():
    print(r"hello\tworld")
    return


@app.cell
def _():
    state_abbrev = {
        'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
        'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
        'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
        'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
        'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
        'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
        'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
        'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
        'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
        'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
        'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
        'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
        'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia'
    }
    return (state_abbrev,)


@app.cell
def _(state_abbrev):
    list(state_abbrev.values())
    return


@app.cell
def _(pl, sales_file_path):
    store_sales2 = (pl.read_csv(sales_file_path,
                    encoding='cp1252',
                    schema_overrides={
                        'Postal Code':pl.String
                    }
                               )
               
                   ).drop(['Row ID', 'Country','Category'])

    store_sales2 = store_sales2.with_columns(
        pl.col('Postal Code')
           .str.zfill(5)
           .alias('Postal Code'),
        pl.col('Sales')
            .cast(pl.String)
            .str.replace_all(r'[^\d.-]','')
            .cast(pl.Decimal, strict=False)
            .alias('Sales')
    )
    store_sales2
    return (store_sales2,)


@app.cell
def _(store_sales2):
    store_sales2.glimpse()
    return


@app.cell
def _(pl, store_sales2):
    (store_sales2
        .filter(pl.col('Postal Code').str
         .len_chars()<5)['Postal Code']
            .unique()    
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
