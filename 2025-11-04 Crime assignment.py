import marimo

__generated_with = "0.16.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""You may use any and all AI tools for this assignment. You may NOT collaborate with any other human.""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Task 1: Read the Data
    Use the chicago_crime_2001_2025.parquet file.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Task 2: Prep the Data
    Perform Necessary Modifications for the data set. Convert data types, add columns required for analysis
    """
    )
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
def _(mo):
    mo.md(
        r"""
    ## Task 4: Breakdown by Crime Type
    Provide the Same list as above, but include a breakdown of crime type
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Task 5: Murder Beats
    Create a polars dataframe showing the top 10 beats by most homicides. Include the District, Community Area, and Homicide count as columns. Sort by homicide count from most to least.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Bonus Task 6: Create a heat map of the areas in Chicago with the highest murder rate""")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
