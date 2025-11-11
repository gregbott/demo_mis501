import marimo

__generated_with = "0.9.14"
app = marimo.App(width="medium")


@app.cell
def __():
    import polars as pl
    import marimo as mo
    import plotly.express as px
    import plotly.graph_objects as go
    return go, mo, pl, px


@app.cell
def __(pl):
    # Read the Chicago crime data
    df = pl.read_parquet('./data/chicago_crime_2001_2025.parquet')

    # Display basic information about the dataset
    print(f"Total records: {df.shape[0]:,}")
    print(f"Columns: {df.columns}")
    df.head()
    return df,


@app.cell
def __(df, pl):
    # Filter for homicides only
    # Note: Adjust the column name if different in your dataset
    # Common column names: 'Primary Type', 'primary_type', 'PrimaryType'

    homicides = df.filter(
        pl.col('Primary Type').str.to_uppercase() == 'HOMICIDE'
    )

    print(f"Total homicides: {homicides.shape[0]:,}")
    homicides.head()
    return homicides,


@app.cell
def __(homicides, pl):
    # Group homicides by beat and include district and community
    # Assuming columns: Beat, District, Community Area
    # Adjust column names as needed based on your actual data

    homicides_by_beat = (
        homicides
        .group_by(['Beat', 'District', 'Community Area'])
        .agg([
            pl.count().alias('Homicide Count'),
        ])
        .sort('Homicide Count', descending=True)
    )

    homicides_by_beat
    return homicides_by_beat,


@app.cell
def __(homicides_by_beat):
    # Display summary statistics
    print(f"Total unique beats with homicides: {homicides_by_beat.shape[0]:,}")
    print(f"\nTop 10 beats by homicide count:")
    homicides_by_beat.head(10)
    return


@app.cell
def __(homicides_by_beat):
    # Additional analysis: Group by district
    district_summary = (
        homicides_by_beat
        .group_by('District')
        .agg([
            pl.sum('Homicide Count').alias('Total Homicides'),
            pl.count().alias('Number of Beats')
        ])
        .sort('Total Homicides', descending=True)
    )

    print("Homicides by District:")
    district_summary
    return district_summary,


@app.cell
def __(homicides_by_beat, px):
    # Histogram: Distribution of homicide counts across beats
    fig_histogram = px.histogram(
        homicides_by_beat.to_pandas(),
        x='Homicide Count',
        nbins=50,
        title='Distribution of Homicide Counts Across Beats',
        labels={'Homicide Count': 'Number of Homicides', 'count': 'Number of Beats'},
        color_discrete_sequence=['#E74C3C']
    )

    fig_histogram.update_layout(
        xaxis_title='Number of Homicides',
        yaxis_title='Number of Beats',
        showlegend=False,
        height=500
    )

    fig_histogram
    return fig_histogram,


@app.cell
def __(homicides_by_beat, px):
    # Bar chart: Top 20 beats by homicide count
    top_20_beats = homicides_by_beat.head(20).to_pandas()

    # Create a combined label for better readability
    top_20_beats['Beat Label'] = (
        'Beat ' + top_20_beats['Beat'].astype(str) +
        ' (Dist ' + top_20_beats['District'].astype(str) + ')'
    )

    fig_top_beats = px.bar(
        top_20_beats,
        x='Homicide Count',
        y='Beat Label',
        orientation='h',
        title='Top 20 Beats by Homicide Count',
        labels={'Homicide Count': 'Number of Homicides', 'Beat Label': ''},
        color='Homicide Count',
        color_continuous_scale='Reds',
        text='Homicide Count'
    )

    fig_top_beats.update_traces(textposition='outside')
    fig_top_beats.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=600,
        showlegend=False
    )

    fig_top_beats
    return fig_top_beats, top_20_beats


@app.cell
def __(district_summary, px):
    # Bar chart: Homicides by district
    fig_district = px.bar(
        district_summary.to_pandas(),
        x='District',
        y='Total Homicides',
        title='Total Homicides by District',
        labels={'Total Homicides': 'Number of Homicides', 'District': 'Police District'},
        color='Total Homicides',
        color_continuous_scale='YlOrRd',
        text='Total Homicides'
    )

    fig_district.update_traces(textposition='outside')
    fig_district.update_layout(
        xaxis={'type': 'category'},
        height=500,
        showlegend=False
    )

    fig_district
    return fig_district,


@app.cell
def __(homicides_by_beat):
    # Save results to CSV for further analysis
    output_file = './homicides_by_beat.csv'
    homicides_by_beat.write_csv(output_file)
    print(f"Results saved to: {output_file}")
    return output_file,


if __name__ == "__main__":
    app.run()
