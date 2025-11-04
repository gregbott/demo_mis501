import marimo

__generated_with = "0.16.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    return mo, pl


@app.cell
def _(mo):
    mo.md(r"""You may use any and all AI tools for this assignment. You may NOT collaborate with any other human.""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Task 1: Read the Data
    Use the chicago_crime_2001_2025.parquet file.
    """
    )
    return


@app.cell
def _(pl):
    df = (
        pl.read_parquet('./data/chicago_crime_2001_2025.parquet')
        .with_columns(
            pl.col('Date').str.to_datetime('%m/%d/%Y %I:%M:%S %p')
        )
        .with_columns(
            pl.col('Date').dt.year().alias('year'),
            pl.col('Date').dt.month().alias('month'),
            pl.col('Date').dt.truncate('1mo').cast(pl.Date).alias('month_period'),
            pl.col('Date').dt.truncate('1q').cast(pl.Date).alias('quarter_period'),
            pl.col('Date').dt.day().alias('day'),
            pl.col('Date').dt.hour().alias('hour'),
            pl.col('Date').dt.minute().alias('minute'),                                  
        )
    )
    return (df,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Task 2: Prep the Data
    Perform Necessary Modifications for the data set. Convert data types, add columns required for analysis
    """
    )
    return


@app.cell
def _(df):
    df.glimpse()
    return


@app.cell
def _(df):
    df['Primary Type'].unique()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Task 3: Most Violent Districts
    List the Top 5 districts with the most Violent Crimes in 2024

    Add a markdown cell explaining which crimes you chose as violent and why.
    """
    )
    return


@app.cell
def _():
    violent_crimes = [
        'HOMICIDE',
        'CRIM SEXUAL ASSAULT',
        'CRIMINAL SEXUAL ASSAULT',
        'ROBBERY',
        'ASSAULT',
        'BATTERY',
        'KIDNAPPING',
        'ARSON',
        'SEX OFFENSE',
        'HUMAN TRAFFICKING',
        'INTIMIDATION',
        'STALKING',
        'WEAPONS VIOLATION'
    ]
    return (violent_crimes,)


@app.cell
def _(df, pl, violent_crimes):
    top_5_violent_districts = (
        df
        .filter(pl.col('year')==2024)
        .filter(pl.col('Primary Type').is_in(violent_crimes))
        .group_by('District')
        .agg(pl.len().alias('violent_crime_count'))
        .sort('violent_crime_count', descending=True)
        .head(5)
    )
    top_5_violent_districts
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Task 4: Breakdown by Crime Type
    Provide the Same list as above but include a breakdown of crime type
    """
    )
    return


@app.cell
def _(df, mo):
    # Get unique districts from the dataframe
    districts = df['District'].unique().sort().to_list()

    # Create dropdown
    district_dropdown = mo.ui.dropdown(
        options=districts,
        value=districts[0] if districts else None,
        label="Select District:"
    )

    district_dropdown
    return (district_dropdown,)


@app.cell
def _(df, district_dropdown, pl, violent_crimes):
    # Access the selected district
    selected_district = district_dropdown.value

    # Filter data for selected district
    district_violent_crimes = (
        df
        .filter(pl.col('District') == selected_district)
        .filter(pl.col('year') == 2024)
        .filter(pl.col('Primary Type').is_in(violent_crimes))
        .group_by('Primary Type')
        .agg(pl.len().alias('count'))
        .sort('count', descending=True)
    )

    print(f"Violent Crimes in District {selected_district} (2024):")
    district_violent_crimes
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Task 5: Murder Beats
    Create a polars dataframe showing the top 10 beats by most homicides. Include the District, Community Area, and Homicide count as columns. Sort by homicide count most to least.
    """
    )
    return


@app.cell
def _(df, pl):
    top_10_homicide_beats_detailed = (
        df
        .filter(pl.col('Primary Type') == 'HOMICIDE')
        .group_by(['Beat', 'District', 'Community Area'])
        .agg(pl.len().alias('homicide_count'))
        .sort('homicide_count', descending=True)
        .head(10)
    )

    print("Top 10 Beats with Most Homicides (with details):")
    print(top_10_homicide_beats_detailed)
    return


@app.cell
def _(mo):
    mo.md(r"""Bonus Task 6: Create a heat map of the areas in Chicago with the highest murder rate""")
    return


@app.cell
def _(df, mo):
    # Get available years from the data
    available_years = df['year'].unique().sort().to_list()

    year_slider = mo.ui.slider(
        start=min(available_years),
        stop=max(available_years),
        value=2024,
        step=1,
        label="Select Year:"
    )

    year_slider
    return


if __name__ == "__main__":
    app.run()
