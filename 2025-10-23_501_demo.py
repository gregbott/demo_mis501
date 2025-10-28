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
    return mo, pl, px


@app.cell
def _(pl):
    df = pl.read_parquet('data/chicago_crime_2001_2025.parquet')
    df.shape
    return (df,)


@app.cell
def _(df):
    df.glimpse()
    return


@app.cell
def _(mo):
    mo.md(r"""## Nice to have a data dictionary""")
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
    mo.md(r"""## How would we extract the quarter from the date into a new field called 'quater_period'?""")
    return


@app.cell
def _(df, pl):
    df2 = df.with_columns(pl.col('Date').str.to_datetime("%m/%d/%Y %I:%M:%S %p"))
    df2 = df2.with_columns(
        pl.col('Date').dt.year().alias('year'),
        pl.col('Date').dt.month().alias('month'),
        pl.col('Date').dt.day().alias('day'),
        pl.col('Date').dt.hour().alias('hour'),
        pl.col('Date').dt.minute().alias('minute'),
        pl.col('Date').dt.truncate('1mo').cast(pl.Date).alias('month_period')    
    )
    df2.glimpse()
    return (df2,)


@app.cell
def _(df2, pl):
    crimes_year_month = (df2
                        .group_by('month_period')
                        .agg(pl.col('ID')
                        .n_unique()
                        .alias('unique_ids')).sort('month_period')
                        )
    return (crimes_year_month,)


@app.cell
def _(crimes_year_month, px):
    fig_year_month = px.bar(crimes_year_month,
                            x='month_period',
                            y='unique_ids',
                            title='Crimes by Month',
                            labels={'ID':'Number of Unique Crimes', 'Month Period':'month_period'},
                            # markers=True
                            )
    fig_year_month.show()
    return


@app.cell
def _(df2):
    df2['hour'].value_counts()
    return


@app.cell
def _(mo):
    mo.md(r"""## Standardize the months""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Overlay temp on crimes per month""")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
