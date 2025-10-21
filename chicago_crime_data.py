import marimo

__generated_with = "0.16.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    import requests
    from typing import Optional

    def download_chicago_data(
        endpoint: str = "https://data.cityofchicago.org/resource/ijzp-q8t2.csv",
        limit: int = 50000,
        output_file: Optional[str] = None
    ) -> pl.DataFrame:
        """
        Download all data from a Chicago Data Portal SODA2 endpoint.
    
        Args:
            endpoint: The API endpoint URL
            limit: Number of records to fetch per request (max 50000 for SODA2)
            output_file: Optional path to save the data as CSV
    
        Returns:
            Polars DataFrame with all the data
        """
        all_data = []
        offset = 0
    
        print(f"Downloading data from {endpoint}")
    
        while True:
            # SODA2 uses $limit and $offset parameters
            params = {
                "$limit": limit,
                "$offset": offset
            }
        
            print(f"Fetching records {offset} to {offset + limit}...")
        
            try:
                response = requests.get(endpoint, params=params)
                response.raise_for_status()
            
                # Read the CSV response into a Polars DataFrame
                from io import StringIO
                df_chunk = pl.read_csv(StringIO(response.text))
            
                # If we got no rows, we've reached the end
                if df_chunk.height == 0:
                    print("No more data to fetch.")
                    break
            
                all_data.append(df_chunk)
                print(f"  Retrieved {df_chunk.height} records")
            
                # If we got fewer rows than the limit, we've reached the end
                if df_chunk.height < limit:
                    print("Reached end of dataset.")
                    break
            
                offset += limit
            
            except requests.exceptions.RequestException as e:
                print(f"Error fetching data: {e}")
                break
    
        # Combine all chunks into a single DataFrame
        if all_data:
            final_df = pl.concat(all_data)
            print(f"\nTotal records downloaded: {final_df.height}")
            print(f"Columns: {final_df.columns}")
        
            # Optionally save to file
            if output_file:
                final_df.write_csv(output_file)
                print(f"Data saved to {output_file}")
        
            return final_df
        else:
            print("No data was downloaded.")
            return pl.DataFrame()
    return (download_chicago_data,)


@app.cell
def _(download_chicago_data):
    # Download the data
    df = download_chicago_data(
        endpoint="https://data.cityofchicago.org/resource/ijzp-q8t2.csv",
        output_file="chicago_data.csv"
    )

    # Display basic info about the dataset
    print("\nDataset shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    return (df,)


@app.cell
def _(df):
    df.shape
    return


@app.cell
def _(df):
    df.shape
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
