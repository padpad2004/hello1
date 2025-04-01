---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.7
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from scipy.interpolate import make_interp_spline
import requests
import re
```

```python
# create the base url which i can use as a base and add the dates to so i can access the different pages
base_url = "https://www.transfermarkt.co.uk/premier-league/marktwerteverein/wettbewerb/GB1/stichtag/"

# Dictionary to store DataFrames for each date
dfs_prem_value = {}

# Custom headers to mimic a real browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/109.0",
    "Accept-Language": "en-US,en;q=0.5"
}

# Loop through each year from 2011 to 2024
for year in range(2011, 2025):
    date_str = f"{year}-03-15"
    url = base_url + date_str 
    print(f"Scraping: {url}")
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Error fetching page for {date_str}: {response.status_code}")
        continue

    soup = BeautifulSoup(response.text, 'html.parser')
    
    # First, try to find the table with class "items"
    table = soup.find('table', class_="items")
    
    # If not found, look inside HTML comments
    if not table:
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for comment in comments:
            comment_soup = BeautifulSoup(comment, "html.parser")
            table = comment_soup.find('table', class_="items")
            if table:
                break

    if table:
        try:
            df = pd.read_html(str(table))[0]
            dfs_prem_value[date_str] = df
            print(f"Found items table for {date_str} with {len(df)} rows.")
        except Exception as e:
            print(f"Error parsing table for {date_str}: {e}")
    else:
        print(f"No items table found for {date_str}.")
```

```python
# Define a regex pattern_top6 matching the desired club names in lower-case
pattern_top6 = r'manchester city|manchester united|chelsea fc|tottenham hotspur|liverpool fc|arsenal fc'
pattern_bottom6 = r'southampton fc|leicester city|ipswich town|fulham fc|everton fc|wolverhampton wanderers'

# Dictionary to store the filtered DataFrames
filtered_dataframes_top_6 = {}

# Loop through each date and DataFrame in the scraped data
for date, df in dfs.items():
    # Check that the expected column is present
    if 'Club' in df.columns:
        # Filter rows by matching the pattern_top6 in a case-insensitive manner.
        filtered_df_top = df[df['Club'].str.lower().str.contains(pattern_top6, na=False)]
        filtered_dataframes_top_6[date] = filtered_df_top
        print(f"{date}: {len(filtered_df_top)} rows retained.")
    else:
        print(f"DataFrame for {date} does not contain a 'Club' column. Available columns: {df.columns}")


# Dictionary to store the filtered DataFrames
filtered_dataframes_bottom_6 = {}

# Loop through each date and DataFrame in your scraped data
for date, df in dfs.items():
    # Check that the expected column is present
    if 'Club' in df.columns:
        # Filter rows by matching the pattern_top6 in a case-insensitive manner.
        filtered_df_bottom = df[df['Club'].str.lower().str.contains(pattern_bottom6, na=False)]
        filtered_dataframes_bottom_6[date] = filtered_df_bottom
        print(f"{date}: {len(filtered_df_bottom)} rows retained.")
    else:
        print(f"DataFrame for {date} does not contain a 'Club' column. Available columns: {df.columns}")

```

```python
filtered_dataframes_top_6['2022-03-15']
```

```python
filtered_dataframes_bottom_6['2022-03-15']
```

```python
# List of column names to drop
columns_to_drop = ['#', 'wappen', 'Club.1', 'Current value', '%', 'Unnamed: 8','Unnamed: 9' ]  # Replace with your actual column names

# Loop through each DataFrame in your dictionary (e.g., filtered_dataframes)
for date, df in filtered_dataframes_top_6.items():
    # Drop the columns and update the DataFrame in the dictionary
    # Using errors='ignore' ensures that if a column is missing, it won't raise an error
    filtered_dataframes_top_6[date] = df.drop(columns=columns_to_drop, errors='ignore')

for date, df in filtered_dataframes_bottom_6.items():
    # Drop the columns and update the DataFrame in the dictionary
    # Using errors='ignore' ensures that if a column is missing, it won't raise an error
    filtered_dataframes_bottom_6[date] = df.drop(columns=columns_to_drop, errors='ignore')
```

```python
# Assuming your dictionary of DataFrames is called 'filtered_dataframes'
for date, df in filtered_dataframes_top_6.items():
    # Check if the column 'League' exists and then rename it to the date
    if 'League' in df.columns:
        df.rename(columns={'League': "Value_" + date}, inplace=True)
    else:
        print(f"'League' column not found in DataFrame for {date}")

for date, df in filtered_dataframes_bottom_6.items():
    # Check if the column 'League' exists and then rename it to the date
    if 'League' in df.columns:
        df.rename(columns={'League': "Value_" + date}, inplace=True)
    else:
        print(f"'League' column not found in DataFrame for {date}")


```

```python
filtered_dataframes_top_6['2014-03-15'].head(6)
```

```python
filtered_dataframes_bottom_6['2014-03-15'].head(6)
```

```python
# New dictionary to store the subset DataFrames
filtered_dataframes_top_6_v1 = {}

for date, df in filtered_dataframes_top_6.items():
    # Construct the value column name based on the date
    value_col = "Value_" + date
    if 'Club' in df.columns and value_col in df.columns:
        # Select only the 'Club' and the 'Value_(date)' columns
        subset_df = df[['Club', value_col]].copy()
        filtered_dataframes_top_6_v1[date] = subset_df
        print(f"For {date}: Retained columns: {subset_df.columns.tolist()}")
    else:
        print(f"DataFrame for {date} does not have the required columns: 'Club' and {value_col}")

filtered_dataframes_bottom_6_v1 = {}

for date, df in filtered_dataframes_bottom_6.items():
    # Construct the value column name based on the date
    value_col = "Value_" + date
    if 'Club' in df.columns and value_col in df.columns:
        # Select only the 'Club' and the 'Value_(date)' columns
        subset_df = df[['Club', value_col]].copy()
        filtered_dataframes_bottom_6_v1[date] = subset_df
        print(f"For {date}: Retained columns: {subset_df.columns.tolist()}")
    else:
        print(f"DataFrame for {date} does not have the required columns: 'Club' and {value_col}")
```

```python
filtered_dataframes_top_6_v1['2017-03-15'].head(10)
```

```python
filtered_dataframes_bottom_6_v1['2017-03-15'].head(10)
```

```python
# Start with an empty combined dataframe
combined_df_top6 = None
combined_df_bottom6 = None

# Loop through each date and merge on the 'Club' column
for date, df in filtered_dataframes_top_6_v1.items():
    if combined_df_top6 is None:
        combined_df_top6 = df
    else:
        combined_df_top6 = pd.merge(combined_df_top6, df, on='Club', how='outer')

# Display the first few rows of the combined dataframe
print(combined_df_top6.head())

for date, df in filtered_dataframes_bottom_6_v1.items():
    if combined_df_bottom6 is None:
        combined_df_bottom6 = df
    else:    
        combined_df_bottom6 = pd.merge(combined_df_bottom6, df, on='Club', how='outer') 
print(combined_df_bottom6.head())

```

```python

def convert_value(value_str):
    """
    Convert a monetary string (e.g., "€310.75m", "€1.19bn") into a numeric value.
    """
    if isinstance(value_str, str):
        # Remove the euro symbol and extra spaces, then convert to lower case
        value_str = value_str.replace("€", "").strip().lower()
        if "m" in value_str:
            try:
                # Remove "m", convert to float, and multiply by 1e6
                return float(value_str.replace("m", "")) * 1_000_000
            except:
                return None
        elif "bn" in value_str:
            try:
                # Remove "bn", convert to float, and multiply by 1e9
                return float(value_str.replace("bn", "")) * 1_000_000_000
            except:
                return None
        else:
            try:
                return float(value_str)
            except:
                return None
    return value_str

# Assuming your combined dataframe is named 'combined_df_top6'
# Loop through all columns and apply conversion on columns that start with "Value_"
for col in combined_df_top6.columns:
    if col.startswith("Value_"):
        combined_df_top6[col] = combined_df_top6[col].apply(convert_value)

for col in combined_df_bottom6.columns:
    if col.startswith("Value_"):
        combined_df_bottom6[col] = combined_df_bottom6[col].apply(convert_value)  



```

```python
df_transposed_top6 = combined_df_top6.set_index('Club').transpose()

# Remove the "Value_" prefix from the index and convert to datetime objects.
df_transposed_top6.index = pd.to_datetime(df_transposed_top6.index.str.replace("Value_", "", regex=True))
df_transposed_top6 = df_transposed_top6.sort_index()

df_transposed_bottom6 = combined_df_bottom6.set_index('Club').transpose()

# Remove the "Value_" prefix from the index and convert to datetime objects.
df_transposed_bottom6.index = pd.to_datetime(df_transposed_bottom6.index.str.replace("Value_", "", regex=True))
df_transposed_bottom6 = df_transposed_bottom6.sort_index()
```

```python
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
from scipy.interpolate import make_interp_spline

# ---------------------- Custom Y-Axis Formatter ---------------------- #
def custom_y_formatter(x, pos):
    if x < 1e9:
        return f"{x/1e6:,.0f} million"
    else:
        return f"{x/1e9:,.1f} billion"

# Set Seaborn style and palette for aesthetics.
sns.set_style("whitegrid")
sns.set_palette("Set2")

# Create a figure with 2 subplots (vertical layout).
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True)

# ---------------------- Plotting Function ---------------------- #
def plot_data(ax, df_transposed, title):
    # Convert datetime index to numeric for spline interpolation.
    x_dates = mdates.date2num(df_transposed.index.to_pydatetime())
    
    # Plot each club's data.
    for club in df_transposed.columns:
        y = df_transposed[club].values
        if len(x_dates) >= 3:
            spline = make_interp_spline(x_dates, y, k=3)  # Cubic spline for smoothness.
            x_dense = np.linspace(x_dates.min(), x_dates.max(), 300)
            y_smooth = spline(x_dense)
            x_dense_dates = mdates.num2date(x_dense)
            ax.plot(x_dense_dates, y_smooth, label=club, linewidth=2)
        else:
            ax.plot(df_transposed.index, y, marker='o', label=club, linewidth=2)
    
    # Format the x-axis: one tick per year, display only the year.
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', rotation=45)
    
    # Disable scientific notation/offsets on the y-axis.
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    
    # Limit the number of y-axis ticks.
    ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
    
    # Apply the custom y-axis formatter.
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_y_formatter))
    
    # Set titles and labels.
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Investment Value", fontsize=12)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

# ---------------------- Plot for Top 6 Teams ---------------------- #
plot_data(axes[0], df_transposed_top6, "Top 6 Teams Values Over Time (2011-2024)")

# ---------------------- Plot for Bottom 6 Teams ---------------------- #
plot_data(axes[1], df_transposed_bottom6, "Bottom 6 Teams Values Over Time (2011-2024)")

plt.tight_layout()
plt.show()

```
