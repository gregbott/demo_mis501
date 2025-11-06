# Polars Proficiency Skills Checklist

## Fundamentals

- [ ] Understand the difference between eager and lazy execution
- [ ] Know when to use `DataFrame` vs `LazyFrame`
- [ ] Understand the `Series` data structure
- [ ] Understand the expression API and method chaining
- [ ] Know the benefits of Polars over pandas (performance, type safety, etc.)

## Data Loading and Saving

- [ ] Read data from CSV files (`read_csv`, `scan_csv`)
- [ ] Read data from Parquet files (`read_parquet`, `scan_parquet`)
- [ ] Read data from JSON files (`read_json`, `scan_ndjson`)
- [ ] Read data from Excel files
- [ ] Read from databases and SQL queries
- [ ] Write data to various formats (CSV, Parquet, JSON)
- [ ] Understand lazy scanning vs eager reading
- [ ] Work with streaming data for large datasets

## Data Inspection

- [ ] Use `head()`, `tail()`, and `glimpse()` for quick inspection
- [ ] Check schema and data types with `schema` and `dtypes`
- [ ] Get shape with `shape` property
- [ ] Use `describe()` for statistical summaries
- [ ] Check for null values with `null_count()`
- [ ] Sample data with `sample()`

## Data Selection

- [ ] Select columns using `select()`
- [ ] Select columns by data type
- [ ] Use column selectors (`cs.numeric()`, `cs.string()`, etc.)
- [ ] Exclude columns with `exclude()`
- [ ] Rename columns with `rename()` and `alias()`
- [ ] Reorder columns

## Data Filtering

- [ ] Filter rows using `filter()` with expressions
- [ ] Use comparison operators (`>`, `<`, `==`, `!=`)
- [ ] Combine filters with logical operators (`&`, `|`, `~`)
- [ ] Filter using `is_null()` and `is_not_null()`
- [ ] Use `is_in()` for membership tests
- [ ] Use `is_between()` for range filtering

## Expressions

- [ ] Understand the `pl.col()` expression
- [ ] Use `pl.lit()` for literal values
- [ ] Chain expressions with method calls
- [ ] Use `when().then().otherwise()` for conditional logic
- [ ] Apply functions with `map_elements()` (formerly `apply`)
- [ ] Use built-in expression functions
- [ ] Create custom expressions

## Data Transformation

- [ ] Add new columns with `with_columns()`
- [ ] Modify existing columns
- [ ] Cast data types with `cast()`
- [ ] Handle missing values with `fill_null()`, `fill_nan()`, `drop_nulls()`
- [ ] Replace values with `replace()`
- [ ] Use `unique()` and `n_unique()`
- [ ] Sort data with `sort()` and `sort_by()`

## Aggregations

- [ ] Use basic aggregations (`sum()`, `mean()`, `min()`, `max()`, `count()`)
- [ ] Use `median()`, `std()`, `var()` for statistics
- [ ] Use `first()`, `last()`, `head()`, `tail()` in aggregations
- [ ] Use `n_unique()` and `unique()` in aggregations
- [ ] Apply multiple aggregations at once
- [ ] Use `agg()` for custom aggregations

## Grouping

- [ ] Group data with `group_by()` (formerly `groupby`)
- [ ] Perform aggregations on grouped data
- [ ] Use multiple grouping columns
- [ ] Understand the difference between `group_by()` and `partition_by()`
- [ ] Use `over()` for window functions with grouping
- [ ] Maintain group context with expressions

## Joins and Concatenations

- [ ] Perform inner joins with `join()`
- [ ] Perform left, right, and outer joins
- [ ] Use semi and anti joins
- [ ] Handle join key conflicts
- [ ] Concatenate DataFrames vertically with `concat()` or `vstack()`
- [ ] Concatenate DataFrames horizontally with `hstack()`
- [ ] Join on multiple columns
- [ ] Perform cross joins

## Window Functions

- [ ] Use `over()` for window operations
- [ ] Apply rolling window functions (`rolling_mean()`, `rolling_sum()`, etc.)
- [ ] Use ranking functions (`rank()`, `row_number()`)
- [ ] Calculate cumulative aggregations (`cum_sum()`, `cum_max()`, etc.)
- [ ] Use `shift()` and `diff()` for time series
- [ ] Partition windows with `over(partition_by=...)`

## String Operations

- [ ] Access string methods with `.str` namespace
- [ ] Use `str.contains()` for pattern matching
- [ ] Extract substrings with `str.slice()` and `str.extract()`
- [ ] Use `str.replace()` and `str.replace_all()`
- [ ] Change case with `str.to_lowercase()` and `str.to_uppercase()`
- [ ] Strip whitespace with `str.strip_chars()`
- [ ] Split and join strings
- [ ] Parse strings to other types

## Date and Time Operations

- [ ] Work with date/datetime data types
- [ ] Parse strings to dates with `str.strptime()`
- [ ] Extract date components (year, month, day) with `.dt` namespace
- [ ] Calculate date differences
- [ ] Use `dt.truncate()` for date rounding
- [ ] Work with time zones
- [ ] Use date ranges with `date_range()`
- [ ] Perform date arithmetic

## Reshaping Data

- [ ] Pivot data with `pivot()`
- [ ] Unpivot data with `unpivot()` (formerly `melt`)
- [ ] Explode lists into rows with `explode()`
- [ ] Create lists from rows with aggregations
- [ ] Transpose DataFrames with `transpose()`

## Working with Lists and Arrays

- [ ] Create list columns
- [ ] Use `.list` namespace for list operations
- [ ] Use `list.lengths()` to get list sizes
- [ ] Explode lists with `explode()`
- [ ] Apply operations to list elements with `list.eval()`
- [ ] Access list elements with indexing
- [ ] Filter and transform nested data

## Working with Structs

- [ ] Create struct columns
- [ ] Access struct fields with `.struct` namespace
- [ ] Unnest structs into separate columns
- [ ] Create structs from multiple columns

## Performance Optimization

- [ ] Prefer lazy evaluation for large datasets
- [ ] Use query optimization with `LazyFrame.explain()`
- [ ] Minimize data type sizes for memory efficiency
- [ ] Use projection pushdown (select only needed columns early)
- [ ] Use predicate pushdown (filter early)
- [ ] Avoid unnecessary `collect()` calls in lazy mode
- [ ] Use `streaming=True` for out-of-core processing
- [ ] Understand query plan optimization

## Advanced Topics

- [ ] Use SQL context with `pl.SQLContext()`
- [ ] Create custom functions with `map_elements()`
- [ ] Work with categorical data efficiently
- [ ] Use parallel processing capabilities
- [ ] Handle large files with lazy and streaming
- [ ] Debug queries with `explain()` and `show_graph()`
- [ ] Integrate Polars with other libraries (NumPy, pandas, Arrow)
- [ ] Optimize memory usage for large datasets

## Data Quality and Validation

- [ ] Check for duplicates with `is_duplicated()` and `is_unique()`
- [ ] Remove duplicates with `unique()`
- [ ] Validate data types
- [ ] Handle missing data appropriately
- [ ] Detect and handle outliers
- [ ] Use assertions for data validation

## Integration and Interoperability

- [ ] Convert between Polars and pandas DataFrames
- [ ] Work with PyArrow tables
- [ ] Convert to/from NumPy arrays
- [ ] Export data to various formats for reporting
- [ ] Use Polars with visualization libraries

## Best Practices

- [ ] Write readable and maintainable query chains
- [ ] Use meaningful column aliases
- [ ] Handle edge cases and nulls appropriately
- [ ] Document complex transformations
- [ ] Profile and benchmark queries
- [ ] Use type hints in custom functions
- [ ] Follow naming conventions
- [ ] Write idiomatic Polars code (avoid pandas patterns)
