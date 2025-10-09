import marimo

__generated_with = "0.16.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    return (mo,)


@app.cell
def _():
    netflix_url = 'https://raw.githubusercontent.com/gregbott/data_501/master/netflix_titles.csv'
    return


@app.cell
def _():
    import chardet
    import requests

    def detect_encoding(url, sample_size=100000):
        response = requests.get(url)
        raw_data = response.content
        sample = raw_data[:sample_size]
        result = chardet.detect(sample)

        return result
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Task 1 - Ingest the Netflix data. What is the shape of the data?
    Answer: 

        Number of columns: 

        Number of rows:
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Task 2 - Examine the data

    Question: Which columns should you modify and why?

    Answer:
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Task 3 - Characterize the data

    Write code to answer the following questions. 

    ### Task 3.1
    How many of types are in the data? How many of each type?

    Answer:
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Task 3.2

    List the top three ratings and how many there are of each. 

    Answer:
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Task 3.3
    What year had the most releases (how many, if a tie, state the most recent year)? 

    Answer:
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Task 3.4
    What year had the fewest releases (how many, if a tie, state the most recent year)? 

    Answer:
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Task 3.5
    What was the average number of releases per year (round to 1 digit of precision)? 

    Answer:
    """
    )
    return


if __name__ == "__main__":
    app.run()
