import marimo

__generated_with = "0.16.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import seaborn as sns
    import plotly.express as px
    sns.set_style('darkgrid') #'darkgrid' or 'whitegrid'
    import matplotlib.pyplot as plt
    return mo, pl, plt, px, sns


@app.cell
def _(pl):
    df = pl.read_parquet('/mnt/expansion16TB/OnedriveUA/1_Projects/701-MIS 501 FL25/shared/student_shared/data/chicago_crime_2001_2025.parquet')
    df.shape
    return (df,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Review the data
    Learn what each field means. See the table below, which was copied from the https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2/about_data site.
    """
    )
    return


@app.cell
def _(df):
    df.glimpse()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Here's the data converted to a markdown table:

    | Column Name | Description | API Field Name | Data Type |
    |:-------------|:-------------|:----------------|:-----------|
    | ID | Unique identifier for the record. | id | Number |
    | Case Number | The Chicago Police Department RD Number (Records Division Number), which is unique to the incident. | case_number | Text |
    | Date | Date when the incident occurred. this is sometimes a best estimate. | date | Floating Timestamp |
    | Block | The partially redacted address where the incident occurred, placing it on the same block as the actual address. | block | Text |
    | IUCR | The Illinois Uniform Crime Reporting code. This is directly linked to the Primary Type and Description. See the list of IUCR codes at https://data.cityofchicago.org/d/c7ck-438e. | iucr | Text |
    | Primary Type | The primary description of the IUCR code. | primary_type | Text |
    | Description | The secondary description of the IUCR code, a subcategory of the primary description. | description | Text |
    | Location Description | Description of the location where the incident occurred. | location_description | Text |
    | Arrest | Indicates whether an arrest was made. | arrest | Checkbox |
    | Domestic | Indicates whether the incident was domestic-related as defined by the Illinois Domestic Violence Act. | domestic | Checkbox |
    | Beat | Indicates the beat where the incident occurred. A beat is the smallest police geographic area – each beat has a dedicated police beat car. Three to five beats make up a police sector, and three sectors make up a police district. The Chicago Police Department has 22 police districts. See the beats at https://data.cityofchicago.org/d/aerh-rz74. | beat | Text |
    | District | Indicates the police district where the incident occurred. See the districts at https://data.cityofchicago.org/d/fthy-xz3r. | district | Text |
    | Ward | The ward (City Council district) where the incident occurred. See the wards at https://data.cityofchicago.org/d/sp34-6z76. | ward | Number |
    | Community Area | Indicates the community area where the incident occurred. Chicago has 77 community areas. See the community areas at https://data.cityofchicago.org/d/cauq-8yn6. | community_area | Text |
    | FBI Code | Indicates the crime classification as outlined in the FBI's National Incident-Based Reporting System (NIBRS). See the Chicago Police Department listing of these classifications at https://gis.chicagopolice.org/pages/crime_details. | fbi_code | Text |
    | X Coordinate | The x coordinate of the location where the incident occurred in State Plane Illinois East NAD 1983 projection. This location is shifted from the actual location for partial redaction but falls on the same block. | x_coordinate | Number |
    | Y Coordinate | The y coordinate of the location where the incident occurred in State Plane Illinois East NAD 1983 projection. This location is shifted from the actual location for partial redaction but falls on the same block. | y_coordinate | Number |
    | Year | Year the incident occurred. | year | Number |
    | Updated On | Date and time the record was last updated. | updated_on | Floating Timestamp |
    | Latitude | The latitude of the location where the incident occurred. This location is shifted from the actual location for partial redaction but falls on the same block. | latitude | Number |
    | Longitude | The longitude of the location where the incident occurred. This location is shifted from the actual location for partial redaction but falls on the same block. | longitude | Number |
    | Location | The location where the incident occurred in a format that allows for creation of maps and other geographic operations on this data portal. This location is shifted from the actual location for partial redaction but falls on the same block. | location | Location |
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Prep the data
    * Which fields, if any, need to be casted?
    * Given our questions, what additional fields do we need?
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Date formats

    * %m - Month as number (10)

    * %d - Day of month (13)

    * %Y - 4-digit year (2025)

    * %I - Hour in 12-hour format (12)

    * %M - Minutes (00)

    * %S - Seconds (00)

    * %p - AM/PM indicator
    """
    )
    return


@app.cell
def _(df, pl):
    _df = df.with_columns(pl.col('Date').str.to_datetime("%m/%d/%Y %I:%M:%S %p"))
    _df.glimpse()
    return


@app.cell
def _():
    # df.write_parquet('chicago_crime_2001_2025.parquet')
    return


@app.cell
def _(df, pl, plt, sns):
    df2 = df.with_columns(pl.col('Date').str.to_datetime('%m/%d/%Y %I:%M:%S %p')) # Note 'datetime', NOT 'date'
    print(df2.schema)
    df2 = df2.with_columns(
        pl.col('Date').dt.year().alias('year'),
        pl.col('Date').dt.month().alias('month'),
        pl.col('Date').dt.truncate('1mo').cast(pl.Date).alias('month_period'),
        pl.col('Date').dt.day().alias('day'),
        pl.col('Date').dt.hour().alias('hour'),
        pl.col('Date').dt.minute().alias('minute'),
    )
    df2.glimpse()
    print(df2['hour'].value_counts().sort('hour'))

    # Count occurrences by hour
    hour_counts = df2.group_by("hour").len().sort("hour")

    # Create bar plot
    sns.barplot(data=hour_counts, x="hour", y="len")
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Crimes")
    plt.title("Distribution of Crimes by Hour")
    plt.show()
    return (df2,)


@app.cell
def _(df2):
    df2.glimpse()
    return


@app.cell
def _(df2, plt, sns):
    # Count occurrences by hour
    minute_counts = df2.group_by("minute").len().sort("minute")

    # Create bar plot
    plt.figure(figsize=(14,6))
    sns.barplot(data=minute_counts, x="minute", y="len")
    plt.xlabel("Minute of Hour")
    plt.ylabel("Number of Crimes")
    plt.title("Distribution of Crimes by Minute of Hour")
    plt.show()
    return


@app.cell
def _(df2, pl):
    unique_observations_by_month = (df2.group_by('month')
                                    .agg(pl.col('ID')
                                    .n_unique()
                                    .alias('unique_ids'))
                                    .sort('month')
                                   )
    return (unique_observations_by_month,)


@app.cell
def _(px, unique_observations_by_month):
    # Which months typically have more crime?
    fig = px.bar(unique_observations_by_month, 
                  x='month', 
                  y='unique_ids',
                  title='Unique Crimes by Month',
                  labels={'ID': 'Number of Unique Crimes', 'Month': 'Month'},
                  # markers=True
                )  # Add markers to points

    fig.show()
    return


@app.cell
def _(df2, pl):
    crimes_year_month = (df2
                        .group_by('month_period')
                        .agg(pl.col('ID')
                            .n_unique()
                            .alias('unique_ids'))
                        .sort('month_period')
                        )
    return (crimes_year_month,)


@app.cell
def _(crimes_year_month, px):
    fig_year_month = px.line(crimes_year_month, 
                  x='month_period', 
                  y='unique_ids',
                  title='Crimes by Year, Month',
                  labels={'ID': 'Number of Unique Crimes', 'Month Period': 'month_period'},
                  markers=True
                )  # Add markers to points

    fig_year_month.show()
    return


@app.cell
def _(df2, pl):
    crime_by_type = (
        df2
        .filter(pl.col('year') == 2025)
        .group_by('Primary Type')
        .agg(
            pl.len().alias('total_crimes')
        )
        .sort('total_crimes', descending=True)
    )

    crime_by_type
    return


@app.cell
def _(df2, pl, px):
    # Filter by specific crime types and sum by year
    crime_types = ['HOMICIDE', 'CRIMINAL SEXUAL ASSAULT', 'PROSTITUTION']

    filtered_crimes = (
        df2
        .filter(pl.col('Primary Type').is_in(crime_types))
        .group_by(['year', 'Primary Type'])
        .agg(pl.len().alias('total_crimes'))
        .sort('year')
    )

    # Create line chart with separate line for each crime type
    serious_crime_fig = px.line(filtered_crimes, 
                  x='year', 
                  y='total_crimes',
                  color='Primary Type',
                  title='Serious Crimes by Year',
                  labels={'total_crimes': 'Number of Crimes', 
                          'year': 'Year',
                          'Primary Type': 'Crime Type'},
                  markers=True)

    serious_crime_fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Number of Crimes",
        legend_title="Crime Type"
    )

    serious_crime_fig.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
