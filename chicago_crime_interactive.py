import marimo

__generated_with = "0.8.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __():
    import polars as pl
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from datetime import datetime, date

    # Parquet file path
    parquet_path = "./data/chicago_crime_2001_2025.parquet"

    return pl, pd, px, go, datetime, date, parquet_path


@app.cell
def __(mo):
    return mo.md("""
    # 🗺️ Chicago Crime Interactive Explorer

    Explore Chicago crime data with interactive visualizations. Select crime types and date ranges to see patterns on the map.
    """)


@app.cell
def __(pl, parquet_path):
    # Load data with necessary columns
    crime_df = (
        pl.scan_parquet(parquet_path)
        .select([
            "Primary Type",
            "Date",
            "Year",
            "Latitude",
            "Longitude",
            "Arrest",
            "Domestic",
            "Description",
            "Location Description"
        ])
        .filter(
            (pl.col("Latitude").is_not_null()) &
            (pl.col("Longitude").is_not_null())
        )
        .collect()
    )

    # Get unique crime types for dropdown
    crime_types = sorted(crime_df.select("Primary Type").unique().to_series().to_list())

    # Get date range from data
    min_year = crime_df.select(pl.col("Year").min()).item()
    max_year = crime_df.select(pl.col("Year").max()).item()

    print(f"Loaded {crime_df.shape[0]:,} crime records")
    print(f"Date range: {min_year} - {max_year}")
    print(f"Crime types: {len(crime_types)}")

    crime_df.head()


@app.cell
def __(mo, crime_types, min_year, max_year):
    # Interactive UI elements

    # Crime type dropdown
    crime_type_dropdown = mo.ui.dropdown(
        options=["All Crime Types"] + crime_types,
        value="All Crime Types",
        label="Select Crime Type:"
    )

    # Start year slider
    start_year_slider = mo.ui.slider(
        start=min_year,
        stop=max_year,
        value=max_year - 5,  # Default to last 5 years
        label="Start Year:",
        show_value=True
    )

    # End year slider
    end_year_slider = mo.ui.slider(
        start=min_year,
        stop=max_year,
        value=max_year,
        label="End Year:",
        show_value=True
    )

    # Sample size slider (for performance)
    sample_size_slider = mo.ui.slider(
        start=1000,
        stop=50000,
        step=1000,
        value=10000,
        label="Max Points to Display:",
        show_value=True
    )

    return crime_type_dropdown, start_year_slider, end_year_slider, sample_size_slider


@app.cell
def __(mo, crime_type_dropdown, start_year_slider, end_year_slider, sample_size_slider):
    return mo.md(f"""
    ## 🎛️ Controls

    Use the controls below to filter the crime data visualization.

    {crime_type_dropdown}

    {start_year_slider}

    {end_year_slider}

    {sample_size_slider}
    """)


@app.cell
def __(pl, crime_df, crime_type_dropdown, start_year_slider, end_year_slider, sample_size_slider):
    # Filter data based on UI selections

    filtered_df = crime_df.filter(
        (pl.col("Year") >= start_year_slider.value) &
        (pl.col("Year") <= end_year_slider.value)
    )

    # Filter by crime type if not "All Crime Types"
    if crime_type_dropdown.value != "All Crime Types":
        filtered_df = filtered_df.filter(
            pl.col("Primary Type") == crime_type_dropdown.value
        )

    # Sample data if needed for performance
    total_records = filtered_df.shape[0]
    if total_records > sample_size_slider.value:
        # Use polars sample for random sampling
        filtered_df = filtered_df.sample(n=sample_size_slider.value, seed=42)
        sampled = True
    else:
        sampled = False

    # Group by location to get frequency (for color intensity)
    location_freq = (
        filtered_df
        .group_by(["Latitude", "Longitude"])
        .agg([
            pl.len().alias("Count"),
            pl.col("Primary Type").first().alias("Crime_Type"),
            pl.col("Description").first().alias("Description"),
            pl.col("Location Description").first().alias("Location_Desc")
        ])
        .sort("Count", descending=True)
    )

    print(f"Filtered records: {total_records:,}")
    if sampled:
        print(f"Displaying sample of {filtered_df.shape[0]:,} points")
    print(f"Unique locations: {location_freq.shape[0]:,}")


@app.cell
def __(mo, crime_type_dropdown, start_year_slider, end_year_slider, total_records, sampled, filtered_df):
    # Display statistics
    return mo.md(f"""
    ### 📊 Filtered Data Statistics

    - **Crime Type**: {crime_type_dropdown.value}
    - **Date Range**: {start_year_slider.value} - {end_year_slider.value}
    - **Total Records**: {total_records:,}
    - **Displayed**: {filtered_df.shape[0]:,} {'(sampled)' if sampled else '(all)'}
    """)


@app.cell
def __(px, location_freq, crime_type_dropdown):
    # Create interactive map with frequency coloring

    map_df = location_freq.to_pandas()

    # Create the scatter mapbox
    fig_map = px.scatter_mapbox(
        map_df,
        lat='Latitude',
        lon='Longitude',
        color='Count',
        size='Count',
        hover_data={
            'Crime_Type': True,
            'Description': True,
            'Location_Desc': True,
            'Count': True,
            'Latitude': ':.4f',
            'Longitude': ':.4f'
        },
        color_continuous_scale='Reds',
        size_max=15,
        zoom=10,
        height=700,
        title=f'Crime Locations: {crime_type_dropdown.value}',
        labels={
            'Count': 'Frequency',
            'Crime_Type': 'Crime Type',
            'Description': 'Description',
            'Location_Desc': 'Location'
        }
    )

    # Update layout
    fig_map.update_layout(
        mapbox_style="open-street-map",
        mapbox_center={"lat": 41.8781, "lon": -87.6298},  # Chicago center
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        coloraxis_colorbar={
            'title': 'Incident<br>Frequency',
            'thicknessmode': 'pixels',
            'thickness': 15,
            'lenmode': 'pixels',
            'len': 300,
            'yanchor': 'top',
            'y': 1
        }
    )

    fig_map


@app.cell
def __(mo):
    return mo.md("""
    ---

    ## 📈 Crime Statistics
    """)


@app.cell
def __(pl, filtered_df):
    # Calculate statistics from filtered data

    stats_by_type = (
        filtered_df
        .group_by("Primary Type")
        .agg([
            pl.len().alias("Count"),
            (pl.col("Arrest").sum() / pl.len() * 100).alias("Arrest_Rate_%"),
            (pl.col("Domestic").sum() / pl.len() * 100).alias("Domestic_%")
        ])
        .sort("Count", descending=True)
        .limit(15)
    )

    stats_by_type


@app.cell
def __(px, stats_by_type):
    # Bar chart of crime types

    stats_pd = stats_by_type.to_pandas()

    fig_bar = px.bar(
        stats_pd,
        x='Primary Type',
        y='Count',
        title='Top Crime Types in Filtered Data',
        labels={'Count': 'Number of Incidents', 'Primary Type': 'Crime Type'},
        color='Count',
        color_continuous_scale='Blues',
        height=500
    )

    fig_bar.update_xaxes(tickangle=-45)
    fig_bar.update_layout(
        showlegend=False,
        xaxis_title='',
        margin={"b": 120}
    )

    fig_bar


@app.cell
def __(px, stats_by_type):
    # Scatter plot: Arrest Rate vs Domestic Crime Rate

    stats_pd_scatter = stats_by_type.to_pandas()

    fig_scatter = px.scatter(
        stats_pd_scatter,
        x='Arrest_Rate_%',
        y='Domestic_%',
        size='Count',
        hover_data=['Primary Type', 'Count'],
        title='Arrest Rate vs Domestic Crime Rate',
        labels={
            'Arrest_Rate_%': 'Arrest Rate (%)',
            'Domestic_%': 'Domestic Crime Rate (%)',
            'Count': 'Total Incidents'
        },
        color='Count',
        color_continuous_scale='Viridis',
        height=500
    )

    fig_scatter.update_layout(
        hovermode='closest',
        showlegend=False
    )

    fig_scatter


@app.cell
def __(pl, filtered_df):
    # Time series analysis

    crimes_by_year = (
        filtered_df
        .group_by("Year")
        .agg([
            pl.len().alias("Total_Crimes"),
            (pl.col("Arrest").sum()).alias("Arrests")
        ])
        .sort("Year")
    )

    crimes_by_year


@app.cell
def __(px, crimes_by_year):
    # Line chart of crimes over time

    time_pd = crimes_by_year.to_pandas()

    fig_line = px.line(
        time_pd,
        x='Year',
        y=['Total_Crimes', 'Arrests'],
        title='Crime Trends Over Time',
        labels={'value': 'Number of Incidents', 'variable': 'Metric'},
        markers=True,
        height=500
    )

    fig_line.update_layout(
        hovermode='x unified',
        legend=dict(
            title='',
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )

    fig_line


@app.cell
def __(mo):
    return mo.md("""
    ---

    ## 🌡️ Heatmap Analysis

    Density-based heatmap showing crime concentration areas.
    """)


@app.cell
def __(go, filtered_df):
    # Create density heatmap

    heat_df = filtered_df.to_pandas()

    fig_heatmap = go.Figure(
        go.Densitymapbox(
            lat=heat_df['Latitude'],
            lon=heat_df['Longitude'],
            radius=10,
            colorscale='Hot',
            showscale=True,
            colorbar=dict(title="Density"),
            hoverinfo='skip'
        )
    )

    fig_heatmap.update_layout(
        mapbox_style="open-street-map",
        mapbox_center={"lat": 41.8781, "lon": -87.6298},
        mapbox_zoom=10,
        height=600,
        title='Crime Density Heatmap',
        margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )

    fig_heatmap


@app.cell
def __(mo):
    return mo.md("""
    ---

    ## 📍 Top Crime Locations

    Most frequent crime locations in the filtered dataset.
    """)


@app.cell
def __(pl, filtered_df):
    # Top locations by frequency

    top_locations = (
        filtered_df
        .group_by(["Location Description", "Primary Type"])
        .agg(pl.len().alias("Count"))
        .sort("Count", descending=True)
        .limit(20)
    )

    top_locations


@app.cell
def __(px, top_locations):
    # Horizontal bar chart of top locations

    loc_pd = top_locations.to_pandas()

    # Combine location and crime type for better labels
    loc_pd['Location_Crime'] = loc_pd['Location Description'] + ' (' + loc_pd['Primary Type'] + ')'

    fig_locations = px.bar(
        loc_pd,
        x='Count',
        y='Location_Crime',
        orientation='h',
        title='Top 20 Location-Crime Type Combinations',
        labels={'Count': 'Number of Incidents', 'Location_Crime': ''},
        color='Count',
        color_continuous_scale='Oranges',
        height=600
    )

    fig_locations.update_layout(
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )

    fig_locations


@app.cell
def __(mo):
    return mo.md("""
    ---

    ## 💡 Key Insights

    - **Interactive Map**: Points are colored by frequency (darker = more incidents at that location)
    - **Point Size**: Larger points indicate higher crime frequency
    - **Hover Details**: Click on any point to see crime type, description, and location details
    - **Performance**: Use the "Max Points to Display" slider to control map performance
    - **Heatmap**: Shows overall crime density patterns across Chicago

    ### Tips for Exploration:
    1. Select specific crime types to see their geographic patterns
    2. Adjust date ranges to see temporal changes
    3. Compare arrest rates across different crime types
    4. Use the heatmap to identify high-crime density areas
    """)


@app.cell
def __(mo, crime_df):
    # Summary statistics card

    total_crimes = crime_df.shape[0]
    unique_locations = crime_df.select(["Latitude", "Longitude"]).unique().shape[0]

    return mo.md(f"""
    ## 📋 Dataset Overview

    - **Total Crimes with Location Data**: {total_crimes:,}
    - **Unique Geographic Locations**: {unique_locations:,}
    - **Interactive Controls**: Dropdown, Date Range, Sample Size
    """)


if __name__ == "__main__":
    app.run()
