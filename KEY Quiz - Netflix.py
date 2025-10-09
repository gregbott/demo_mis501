import marimo

__generated_with = "0.16.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    return mo, pl


@app.cell
def _():
    netflix_url = 'https://raw.githubusercontent.com/gregbott/data_501/master/netflix_titles.csv'
    return (netflix_url,)


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
    Number of columns: 26
    Number of rows: 8809

    """
    )
    return


@app.cell
def _(netflix_url, pl):
    # utf-8 is most common, but fails for this data set
    columns_to_drop = [
      "",
      "_duplicated_0",
      "_duplicated_1",
      "_duplicated_2",
      "_duplicated_3",
      "_duplicated_4",
      "_duplicated_5",
      "_duplicated_6",
      "_duplicated_7",
      "_duplicated_8",
      "_duplicated_9",
      "_duplicated_10",
      "_duplicated_11",
      "_duplicated_12"
    ]
    df = pl.read_csv(netflix_url, encoding='latin-1')
    print(df.shape)
    df = df.drop(columns_to_drop)

    return (df,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Task 2 - Examine the data

    Question: Which columns should you modify and why?

    Answer: 
    1. date_added dtype is string and should be date
    2. The blank column and all the columns with 'duplicated' in them should be deleted since they're null
    3. Rating has invalid entries (minutes, blank, etc.)
    4. Bonus: The country column has multiple values in each row and needs to be split to be efficiently accessed
    5. Bonus: Duration has both minutes and seasons, which may need to be separated
    6. Bonus: Listed in has multiple values per row
    7. Bonus: various columns could be converted to a category dtype
    8. Bonus: Director has characters listed that may be indicative of bad encoding, multi-valued
    """
    )
    return


@app.cell
def _(df):
    df.glimpse()
    return


@app.cell
def _(df):
    df['date_added'].unique()
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(df, pl):
    releases_by_yr = df.group_by('release_year').agg([
        pl.col('show_id').count().alias('count')
    ])
    releases_by_yr
    return (releases_by_yr,)


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Task 3 - Characterize the data

    Write code to answer the following questions. 

    ### Task 3.1
    How many of types are in the data? How many of each type?

    Answer: 2 TV Show (2,677) and Movie (6,132)
    """
    )
    return


@app.cell
def _(df):
    df['type'].value_counts()
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Task 3.2

    List the top three ratings and how many there are of each. Answer: TV-MA (3208), TV-14 (2160), TV-PG (863)
    """
    )
    return


@app.cell
def _(df, pl):
    by_rating = df.group_by('rating').agg(pl.col('show_id').count().alias('count')).sort('count',descending=True)
    by_rating
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Task 3.3
    What year had the most releases (how many, if a tie, state the most recent year)? Answer: 2018 (1147)
    """
    )
    return


@app.cell
def _(df, pl):
    df.group_by('release_year').agg([
        pl.col('show_id').count().alias('count')
    ]).sort('count',descending=True)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Task 3.4
    What year had the fewest releases (how many, if a tie, state the most recent year)? 

    Answer: 2024 (1)
    """
    )
    return


@app.cell
def _(df, pl):
    df.group_by('release_year').agg(pl.col('show_id').count().alias('count')).sort('count',descending=False)
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Task 3.5
    What was the average number of releases per year (round to 1 digit of precision)? 117.5
    """
    )
    return


@app.cell
def _(releases_by_yr):
    round(releases_by_yr['count'].mean(),1)
    return


if __name__ == "__main__":
    app.run()
