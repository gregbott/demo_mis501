import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    from datetime import datetime, date
    import pathlib
    import chardet

    data_path = pathlib.Path('/mnt/expansion16TB/Dropbox/3_Resources/marimo_data')
    return chardet, mo, pathlib, pl


@app.cell
def _(mo):
    mo.md(
        r"""
    # 1. Foundations & Philosophy

    **Why Polars?** Performance, memory efficiency, expressive API

    **Core concepts:** Immutability, lazy vs eager execution, expressions

    **Key differences from pandas:** No index, column-oriented, query optimization
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Introduction and Setup

    ```bash
    pixi add polars
    ```
    """
    )
    return


@app.cell
def _(pl):
    print(f"Polars version: {pl.__version__}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## What is Polars?
    Polars is a blazingly fast DataFrame library for Python (and Rust).

    Key characteristics:

    - Built in Rust for maximum performance
    - Lazy evaluation with query optimization
    - Memory efficient with Apache Arrow backend
    - Expressive API similar to pandas but more consistent
    - Multi-threaded operations by default
    - No index-based operations (unlike pandas)

    Pandas is another DataFrame library released in 2008 by Wes McKinney. Polars is designed to handle larger-than-memory datasets efficiently and provides a more predictable performance profile than pandas.

    ## DataFrame vs. Series
    """
    )
    return


@app.cell
def _(pl):

    ## DataFrame vs Series


    ### Understanding DataFrame vs Series

    ### A Series is a 1-dimensional data structure (like a column)
    _series_example = pl.Series("temperatures", [20.5, 22.1, 19.8, 23.4, 21.0])
    print("Series example:")
    print(f"Series data type: {_series_example.dtype}")
    print(f"Series length: {len(_series_example)}")
    print(f"Object type: {type(_series_example)}")
    _series_example
    return


@app.cell
def _(pl):
    ### A DataFrame is a 2-dimensional data structure (like a table)
    df_example = pl.DataFrame({
        "city": ["New York", "London", "Tokyo", "Sydney", "Paris"],
        "temperature": [20.5, 22.1, 19.8, 23.4, 21.0],
        "humidity": [65, 70, 80, 55, 60]
    })


    print(f"DataFrame shape: {df_example.shape}")
    print(f"DataFrame columns: {df_example.columns}")
    print("DataFrame example:")
    df_example
    return (df_example,)


@app.cell
def _(df_example):
    df_example
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Polars Has No Index

    Key difference from pandas:

    - Polars DataFrames have NO INDEX
    - Rows are identified by their position (0, 1, 2, ...)
    - No .loc, .iloc, or index-based operations (however you can use .item(x, y))
    - This simplifies operations and improves performance
    - Use .with_row_index() if you need row numbers as a column
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # 2. Loading & Inspecting Data

    Reading files: `read_csv()`, `read_parquet()`, `read_excel()`

    Schema control: `dtypes`, `schema_overrides`

    Initial exploration: `head()`, `tail()`, `glimpse()`, `describe()`, `schema`
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## From a Dictionary""")
    return


@app.cell
def _(pl):
    _dict_data = {
        "name": ["Alice", "Bob", "Charlie", "Diana"],
        "age": [25, 30, 35, 28],
        "salary": [50000, 60000, 70000, 55000],
        "department": ["Engineering", "Marketing", "Engineering", "Sales"]
    }

    _df_from_dict = pl.DataFrame(_dict_data)
    _df_from_dict
    return


@app.cell
def _(mo):
    mo.md(r"""## From Lists""")
    return


@app.cell
def _(pl):
    _list_data = [
        ["Product A", 100, 25.99],
        ["Product B", 150, 19.99],
        ["Product C", 75, 35.50],
        ["Product D", 200, 15.99]
    ]

    _df_from_lists = pl.DataFrame(
        _list_data,
        schema=["product_name", "quantity", "price"],
        orient="row"
    )

    _df_from_lists
    return


@app.cell
def _(mo):
    mo.md(r"""## From files (csv, xlsx, json, parquet,...)""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Schema control and Encoding
    Encoding converts human-readable text into a machine-readable format, usually binary data, for storage or transmission
    . A specific encoding standard, like UTF-8, assigns each character a unique numerical value. When you open a text file, your computer uses that same encoding standard to interpret the numbers and display the correct letters. A mismatch between how a file was saved and how it is read can result in unreadable or "garbled" text.

    ### dtypes

    Most Commonly Used:

    Integers: pl.Int64, pl.Int32
    Floats: pl.Float64
    Text: pl.String
    Dates: pl.Date, pl.Datetime
    Boolean: pl.Boolean

    Aliases:

    * ```pl.Int = pl.Int64```
    * ```pl.Float = pl.Float64```
    * ```pl.Utf8 = pl.String```

    Rule of thumb for currency values:
    * Analytics/reporting/visualizations → Float64
    * Accounting/invoicing/legal compliance → Decimal
    * High-frequency trading/mission-critical → Store as integer cents

    For your sales forecasting data, pl.Float64 is probably the best choice unless you have specific requirements for exact decimal arithmetic.


    ### schema_overrides
    Polars
    ```schema_overrides``` is a feature that allows you to specify or change the data types for individual columns when reading a file, overriding Polars' default type inference. You should use it when Polars incorrectly infers a column's data type, such as interpreting a datetime column as a string or a postal code (which should be a string) interpreted as an integer. This ensures that your data is correctly parsed from the start, which is especially important for lazy evaluations.
    """
    )
    return


@app.cell
def _(store_sales_df):
    store_sales_df.estimated_size()
    return


@app.cell
def _(pl):
    # When possible, have a good idea of the data you're ingesting. How many columns? How many rows? What fields? Types? Etc.

    # Error: First attempt. In its original format, the csv file was encoded using 'cp1252' but is read using a different (incorrect) encoding.
    #      No exception is raised, but it is misread. Note ZERO rows and 42,441 columns. 

    sales_file_path = './stores_sales_forecasting.csv'

    # store_sales_df = pl.read_csv(sales_file_path) # First attempt - ERROR, 0 rows

    # Second attempt
    store_sales_df = (pl.read_csv(sales_file_path,
                    encoding='cp1252', # Unless you specify the proper encoding, the data will not ingest correctly.
                    schema_overrides={
                            'Row ID':pl.UInt16,
                            'Postal Code':pl.String,
                            'Quantity':pl.UInt16,
                            'Sales':pl.Decimal(scale=4),
                            'Profit':pl.Decimal(scale=4),
                            'Discount':pl.Decimal(scale=4)
                            }
                     ).drop(['Country', 'Row ID'])
         )

    store_sales_df.shape
    return sales_file_path, store_sales_df


@app.cell
def _(mo):
    mo.md(r"""### How to determine encoding""")
    return


@app.cell
def _(chardet, pathlib, sales_file_path):
    with open(pathlib.Path(sales_file_path), 'rb') as file:
        # Read first 100,000 bytes or entire file, () if smaller
        raw_data = file.read(100000)

    # Detect encoding
    results = chardet.detect(raw_data)
    results
    return


@app.cell
def _(store_sales_df):
    store_sales_df.head()
    return


@app.cell
def _(store_sales_df):
    store_sales_df['Segment'].value_counts()
    return


@app.cell
def _(store_sales_df):
    # If you see only types and no column names, you have a problem. In this case, it's an encoding error.
    dict(store_sales_df.schema) # wrapped in dict() constructor so Marimo prints vertically
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### glimpse()

    ```python
    glimpse()
    ```
    Returns a formatted string showing a summary of the DataFrame including:

    - Number of rows and columns
    - Column names
    - data types
    - a preview of the first few values
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Return a single item""")
    return


@app.cell
def _(store_sales_df):
    # store_sales_df.item(1,2) = 'November 8, 2016' # Not allowed in Polars, DataFrames are immutable
    store_sales_df.item(1,2)
    return


@app.cell
def _(mo):
    mo.md(r"""### Check Unique values""")
    return


@app.cell
def _(store_sales_df):
    store_sales_df["Segment"].unique()
    return


@app.cell
def _(mo):
    mo.md(r"""### Get count of unique values""")
    return


@app.cell
def _(store_sales_df):
    store_sales_df["Segment"].value_counts()
    return


@app.cell
def _(store_sales_df):
    store_sales_df['State'].value_counts(sort=True, parallel=True, name="count")
    return


@app.cell
def _(store_sales_df):
    store_sales_df['State'].unique().sort()
    return


@app.cell
def _():
    return


@app.cell
def _():
    # clean_stores.filter(pl.col('Postal Code').str.starts_with('0'))['Postal Code'].unique().sort()
    return


@app.cell
def _(store_sales_df):
    store_sales_df.glimpse()
    return


@app.cell
def _(mo):
    mo.md(r"""## Summary Statistics""")
    return


@app.cell
def _(store_sales_df):
    store_sales_df.describe()
    return


@app.cell
def _(pl, store_sales_df):
    # Since using Select, results in a DataFrame
    sorted_zip = (
                    store_sales_df
                    .filter(pl.col('Postal Code').str.len_chars()<5)
                    .select('Postal Code')
                    .unique()
                    .sort('Postal Code')
    )
    sorted_zip
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # 3. Selecting & Filtering

    Column selection: `select()`, `pl.col()`, `pl.all()`, `pl.exclude()`

    Row filtering: `filter()`, boolean expressions

    Slicing: row/column indexing with `[]`

    Getting values: `item()`, `row()`
    """
    )
    return


@app.cell
def _(store_sales_df):
    _single_col_df = store_sales_df.select("Customer Name") #.select() returns a dataframe, not a series
    print(type(_single_col_df))
    _single_col_df
    return


@app.cell
def _(store_sales_df):
    store_sales_df.columns
    return


@app.cell
def _(store_sales_df):
    # Select multiple columns:
    _multi_cols = store_sales_df.select(["Customer Name", "Segment", "State"])
    _multi_cols
    return


@app.cell
def _(pl, store_sales_df):
    # Row filtering
    print("Filter rows where profit < 0:")
    _filtered_df = store_sales_df.filter(pl.col("Profit") < 0)
    _filtered_df
    return


@app.cell
def _(pl, store_sales_df):
    print("\nFilter rows with multiple conditions: Corporate sales with profit loss")
    _complex_filter = store_sales_df.filter(
        (pl.col("Profit") < 0) & (pl.col("Segment") == "Corporate")
    )[['Order ID', 'Customer Name','Segment','Profit']].sort('Profit')
    _complex_filter
    return


@app.cell
def _(mo):
    mo.md(r"""## Sorting and Grouping""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Grouping and aggregation""")
    return


@app.cell
def _(store_sales_df):
    store_sales_df.columns
    return


@app.cell
def _(pl, store_sales_df):
    print("Group by department and calculate mean salary:")
    _grouped = store_sales_df.group_by("Sub-Category").agg(
        pl.col("Sales").mean().alias("avg_sale"),
        pl.col("Profit").mean().alias("avg_profit"),
        pl.len().alias("count")
    ).sort('avg_profit',descending=True)
    _grouped
    return


@app.cell
def _(pl, store_sales_df):
    store_sales_df.group_by('Sub-Category').agg([
        pl.col('Sales').sum().alias('total_sales'),
        pl.col('Profit').sum().alias('total_profit'),
        pl.col('Sales').mean().alias('avg_sales'),
        pl.col('Profit').mean().alias('avg_profit')
    ])
    return


@app.cell
def _(df_sales2, pl):
    # Total sales by region
    df_sales2.group_by('Region').agg(pl.col('Sales').sum())
    return


@app.cell
def _(df_sales2, pl):
    # Average sales by category
    df_sales2.group_by('Category').agg(pl.col('Sales').mean())
    return


@app.cell
def _(df_sales2, pl):
    # Count orders by ship mode
    df_sales2.group_by('Ship Mode').agg(pl.len().alias('count'))
    return


@app.cell
def _(pl, store_sales_df):
    store_sales_df.select(pl.col('Sales').sum().round(3)) # The DataFrame object has a round() method
    return


@app.cell
def _(store_sales_df):
    # Using this syntax results in a float, which does NOT have a .round() method, so we wrap the statement in the round() function and specify the number of digits precision after the decimal point.
    round(store_sales_df['Sales'].sum(),3) 
    return


@app.cell
def _(store_sales_df):
    store_sales_df['Sub-Category'].unique()
    return


@app.cell
def _(store_sales_df):
    store_sales_df.columns
    return


@app.cell
def _(mo):
    mo.md(r"""Adding and Modifying Columns""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### List average sale and profit by category
    Include a count of items.
    """
    )
    return


@app.cell
def _(pl, store_sales_df):
    ## Method Chaining Example


    # Polars supports method chaining for complex operations
    print("Method chaining example:")
    print("Filter → Group → Aggregate → Sort")

    _chained_result = (
        store_sales_df
        .filter(pl.col("Profit") < 0)  # Filter loss
        .group_by("Sub-Category")       # Group by department
        .agg([                        # Multiple aggregations
            pl.col("Sales").mean().alias("avg_sale"),
            pl.col("Profit").mean().alias("avg_profit"),
            pl.len().alias("count")
        ])
        .sort("avg_sale", descending=True)  # Sort by average salary
    )

    print(_chained_result)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 4. Data Transformation

    Adding/modifying columns: `with_columns()`

    Expressions: `pl.col()` operations (math, string, date methods)
    sum_horizontal()
    max_horizontal()
    min_horizontal()
    mean_horizontal()

    Casting types: `cast()`

    Renaming: `alias()`, `rename()`

    Dropping: `drop()`
    """
    )
    return


@app.cell
def _(df_grades):
    df_grades.glimpse()
    return


@app.cell
def _(mo):
    mo.md(r"""### Analyze Grades""")
    return


@app.cell
def _(pl):
    df_grades = pl.read_excel('data/grades.xlsx')

    # sum quiz points
    grades_with_totals = df_grades.with_columns([
        pl.mean_horizontal('essay_1','essay_2','essay_3').round(0).alias('avg_essay'),
        pl.mean_horizontal(pl.selectors.contains('quiz')).round(0).alias('avg_quiz'), # could import pl.selectors as cs instead
        pl.mean_horizontal(pl.selectors.contains('exam')).round(0).alias('avg_exam'),
        pl.mean_horizontal(pl.selectors.numeric()).round(0).alias('non-weighted_avg'),

        # Worst quiz score
        pl.min_horizontal(pl.selectors.contains('quiz')).alias('worst_quiz'),

        # Worst essay score
        pl.min_horizontal(pl.selectors.contains('essay')).alias('worst_essay'),

        # Worst exam score
        pl.min_horizontal(pl.selectors.contains('exam')).alias('worst_exam'),

        # Worst score overall
        pl.min_horizontal(pl.selectors.numeric()).alias('worst_score_overall')

    ]).with_columns([
        # Weighted final grade: 20% quizzes, 30% essays, 20% midterm, 30% final
        (
            pl.col('avg_quiz') * 0.20 +
            pl.col('avg_essay') * 0.30 +
            pl.col('midterm_exam') * 0.20 +
            pl.col('final_exam') * 0.30
        ).round(2).alias('final_grade')
    ]).with_columns([
        # Assign letter grades
        pl.when(pl.col('final_grade') >= 90).then(pl.lit('A'))
        .when(pl.col('final_grade') >= 80).then(pl.lit('B'))
        .when(pl.col('final_grade') >= 70).then(pl.lit('C'))
        .when(pl.col('final_grade') >= 60).then(pl.lit('D'))
        .otherwise(pl.lit('F'))
        .alias('letter_grade')
    ])
    grades_with_totals
    # grades_with_totals[['avg_essay', 'avg_quiz','avg_exam','non-weighted_avg']]
    return (df_grades,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 5. String & Date Operations

    String methods: ```str.to_lowercase(), str.contains(), str.replace(), str.len_chars()```

    Date parsing: ```str.strptime(), str.to_date()```

    Date manipulation: ```dt.year(), dt.month(), date arithmetic, strptime(), strftime()```
    """
    )
    return


@app.cell
def _():
    # Grok this!! "Grok, create a Python dictionary with two-digit state abbreviations as the key and the full state name as the value."
    state_abbrev = {
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
        'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
        'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
        'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
        'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
        'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
        'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
        'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
        'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
        'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
        'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
        'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
        'Wisconsin': 'WI', 'Wyoming': 'WY', 'District of Columbia': 'DC'
    }
    return (state_abbrev,)


@app.cell
def _():
    print("hello\nworld")
    return


@app.cell
def _():
    state_corrections = {'Californya':'CA'}
    return (state_corrections,)


@app.cell
def _(pl, state_abbrev, state_corrections):
    df_sales2 = pl.read_csv('data/stores_sales_forecasting2.csv',
                           encoding='cp1252')
    df_sales2 = df_sales2.with_columns(
        pl.col('State')
            .replace_strict(state_abbrev, default=pl.col('State')) # _strict leaves value unchanged
            .replace_strict(state_corrections, default=pl.col('State')),
        pl.col('Sales')
            .cast(pl.String) # Ensure it's a string so we can run replace_all/Regex on it
            .str.replace_all(r'[^\d.-]', '')  # Remove all non-numeric characters except digits, decimal, and minus
            .cast(pl.Decimal, strict=False) # Convert to float, True - raises error, Fales-non-convertible become null
            .alias('Sales'), # If changing in-place, this is not necessary, included for readability,
        pl.col('Profit')
            .cast(pl.Decimal),
        # pl.col("Order Date") # ERROR: This fails because there are two date formats and only one is accounted for
        #     .str.strptime(pl.Date, format="%m/%d/%Y", strict=False)
        #     .alias("Order Date Parsed"), # making a new column so that we can compare old and new
        pl.coalesce([
            pl.col('Order Date').str.strptime(pl.Date, format="%m/%d/%Y", strict=False), # # Consider creating a function for this that includes all common date formats being careful to discren non-US formats
            pl.col('Order Date').str.strptime(pl.Date, format="%B %d, %Y", strict=False)
        ]).alias('Order Date Parsed'),
        pl.col('Postal Code')
            .cast(pl.String)
            .str.zfill(5)
            .alias('Postal Code')
    )
    return (df_sales2,)


@app.cell
def _(df_sales2):
    df_sales2.glimpse()
    return


@app.cell
def _(df_sales2):
    df_sales2['State'].unique().sort()
    return


@app.cell
def _(df_sales2, pl):
    # Display zip codes that likely had leading zeros 
    (df_sales2
        .filter(pl.col('Postal Code').str
            .len_chars()<5)['Postal Code']
        .unique()
        .sort())
    return


@app.cell
def _(df_sales2, pl):
    # Keep original string column while creating parsed version
    # df_sales = store_sales_df.with_columns([
    #     pl.col("Order Date")
    #     .str.strptime(pl.Date, format="%m/%d/%Y", strict=False)
    #     .alias("Order Date Parsed")
    # ])

    # Filter for nulls in parsed column (these are failures)
    parse_failures = df_sales2.filter(
        pl.col("Order Date Parsed").is_null() & 
        pl.col("Order Date").is_not_null()  # Original had a value
    )

    print(f"Rows that failed to parse: {len(parse_failures)}")
    print(parse_failures.select(["Order Date", "Order Date Parsed"]))
    print(parse_failures.select(["Order Date", "Order Date Parsed"]))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 6. Aggregations & Grouping

    Basic aggregations: `sum()`, `mean()`, `min()`, `max()`, `count()`

    Grouping: `group_by()` + `agg()`

    Multiple aggregations at once

    Window functions
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 7. Sorting & Ranking

    Sorting: `sort()`, multiple columns, ascending/descending

    Ranking: `rank()`, `row_number()`
    """
    )
    return


@app.cell
def _(df_grades, pl):
    df_grades_ranking = df_grades.with_columns([
        pl.col('final_exam').rank(method='average', descending=True).alias('rank_average'),
        pl.col('final_exam').rank(method='min', descending=True).alias('rank_min'),
        pl.col('final_exam').rank(method='max', descending=True).alias('rank_max'),
        pl.col('final_exam').rank(method='dense', descending=True).alias('rank_dense'),
        pl.col('final_exam').rank(method='ordinal', descending=True).alias('rank_ordinal')
    ])
    return (df_grades_ranking,)


@app.cell
def _(df_grades_ranking):
    df_grades_ranking.columns
    return


@app.cell
def _(df_grades_ranking):
    df_grades_ranking[['first_name','final_exam','rank_min']].sort('rank_min')
    return


@app.cell
def _(df_grades_ranking):
    df_grades_ranking[['first_name','final_exam','rank_max']].sort('rank_max')
    return


@app.cell
def _(df_grades_ranking):
    df_grades_ranking[['first_name','final_exam','rank_dense']].sort('rank_dense')
    return


@app.cell
def _(df_grades_ranking):
    df_grades_ranking[['first_name','final_exam','rank_ordinal']].sort('rank_ordinal')
    return


@app.cell
def _(df_grades_ranking):
    df_grades_ranking[['first_name','final_exam','rank_average']].sort('rank_average')
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 8. Joins & Concatenation

    Join types: `join()` (inner, left, outer, cross, semi, anti)

    Concatenating: `pl.concat()` (vertical/horizontal)

    Union operations
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 9. Reshaping Data

    Pivoting: `pivot()`

    Melting: `melt()` (wide to long)

    Exploding lists: `explode()`
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 10. Conditional Logic

    ```when().then().otherwise()``` chains

    Complex conditional transformations

    ```pl.lit()``` for literal values

    * **Order matters**: Conditions are evaluated top-to-bottom, and the first matching condition wins
    * **Always use** ```.alias()```: Give your new column a name
    * **Use with** ```.with_columns()```: This adds/modifies columns in your dataframe
    * **Conditions must return boolean**: Use comparison operators like >, <, ==, !=
    * **Combine conditions**: Use & (AND), | (OR), ~ (NOT) for complex logic
    * **Parentheses required**: When combining conditions, wrap each in parentheses: (cond1) & (cond2)
    """
    )
    return


@app.cell
def _(df_sales2, pl):
    df_sales2.with_columns(
        pl.when(pl.col('Sales')>1000)
        .then(pl.lit('High'))
        .otherwise(pl.lit('Low'))
        .alias('Sales Level')
    )[['Sales', 'Sales Level']]
    return


@app.cell
def _(df_sales2, pl):
    df_sales2.with_columns(
        pl.when(pl.col('Sales') > 4000)
        .then(pl.lit('High'))
        .when(pl.col('Sales') > 1000)
            .then(pl.lit('Med'))
        .otherwise(pl.lit('Low'))
        .alias('Sales_level')
    )['Sales_level'].value_counts()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 11. Missing Data

    Detecting: `is_null()`, `is_not_null()`, `null_count()`

    Handling: `fill_null()`, `drop_nulls()`, `forward_fill()`, `backward_fill()`
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 12. Advanced Topics

    Lazy evaluation: `pl.LazyFrame`, `collect()`, query optimization

    Custom functions: `map_elements()` (use sparingly!)

    List columns: nested data structures

    Performance optimization techniques
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 13. I/O & Export

    Writing: `write_csv()`, `write_parquet()`, `write_excel()`

    Interoperability: converting to/from pandas, numpy, arrow
    """
    )
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Key Polars Concepts Summary:

    1. DataFrame: 2D table structure (rows & columns)
    2. Series: 1D column structure  
    3. No Index: Rows identified by position only
    4. Lazy Evaluation: Use scan_* functions for large datasets
    5. Expressions: Use pl.col() for column references
    6. Method Chaining: Operations can be chained together
    7. Performance: Built for speed with Rust backend

    Common patterns:
    - ```df.select()``` for column selection
    - ```df.filter()``` for row filtering  
    - ```df.with_columns()``` for adding/modifying columns
    - ```df.group_by().agg()``` for aggregations
    - Chain methods together for complex operations
    """
    )
    return


if __name__ == "__main__":
    app.run()
