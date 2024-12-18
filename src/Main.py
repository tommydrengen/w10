# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# Load data
file_path = r'C:\Users\spac-23\Documents\w10\vgsales.csv'
data = pd.read_csv(file_path, header=0)

# Handle missing values in numerical columns
numerical_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
imputer = SimpleImputer(strategy='mean')
data[numerical_cols] = imputer.fit_transform(data[numerical_cols])

# Clean Year column
data['Year'] = pd.to_numeric(data['Year'], errors='coerce')  # Convert to numeric
data = data.dropna(subset=['Year'])  # Remove rows with missing years
data['Year'] = data['Year'].astype(int)  # Convert Year to integer for grouping

# Sales by Release Year Across Regions
def plot_sales_by_release_year(data):
    yearly_sales = data.groupby('Year')[numerical_cols].sum()
    yearly_sales.plot(figsize=(12, 8), marker='o')
    plt.title('Yearly Sales Trends by Region')
    plt.ylabel('Total Sales (in Millions)')
    plt.xlabel('Release Year')
    plt.legend(loc='best', title='Regions')
    plt.grid()
    plt.show()

# Sales by Release Year Across Publishers
def plot_sales_by_publisher(data, top_n=10):
    publisher_sales = (
        data.groupby(['Publisher', 'Year'])[numerical_cols].sum()
        .reset_index()
        .groupby('Publisher').sum()['Global_Sales']
        .sort_values(ascending=False)
        .head(top_n)
    )
    top_publishers = publisher_sales.index.tolist()

    filtered_data = data[data['Publisher'].isin(top_publishers)]
    publisher_yearly_sales = filtered_data.groupby(['Year', 'Publisher'])['Global_Sales'].sum().unstack()

    publisher_yearly_sales.plot(figsize=(12, 8), marker='o')
    plt.title(f'Yearly Sales Trends by Publisher (Top {top_n})')
    plt.ylabel('Total Sales (in Millions)')
    plt.xlabel('Release Year')
    plt.legend(loc='best', title='Publishers')
    plt.grid()
    plt.show()

# Sales by Release Year Across Genres
def plot_sales_by_genre(data):
    genre_sales = data.groupby(['Year', 'Genre'])['Global_Sales'].sum().unstack()
    genre_sales.plot(figsize=(12, 8), marker='o')
    plt.title('Yearly Sales Trends by Genre')
    plt.ylabel('Total Sales (in Millions)')
    plt.xlabel('Release Year')
    plt.legend(loc='best', title='Genres')
    plt.grid()
    plt.show()

# Sales by Release Year Across Platforms
def plot_sales_by_platform(data, top_n=10):
    platform_sales = (
        data.groupby(['Platform', 'Year'])[numerical_cols].sum()
        .reset_index()
        .groupby('Platform').sum()['Global_Sales']
        .sort_values(ascending=False)
        .head(top_n)
    )
    top_platforms = platform_sales.index.tolist()

    filtered_data = data[data['Platform'].isin(top_platforms)]
    platform_yearly_sales = filtered_data.groupby(['Year', 'Platform'])['Global_Sales'].sum().unstack()

    platform_yearly_sales.plot(figsize=(12, 8), marker='o')
    plt.title(f'Yearly Sales Trends by Platform (Top {top_n})')
    plt.ylabel('Total Sales (in Millions)')
    plt.xlabel('Release Year')
    plt.legend(loc='best', title='Platforms')
    plt.grid()
    plt.show()

# Call the functions
plot_sales_by_release_year(data)  # Sales by release year across regions
plot_sales_by_publisher(data, top_n=10)  # Top 10 publishers
plot_sales_by_genre(data)  # Sales by genre
plot_sales_by_platform(data, top_n=10)  # Top 10 platforms



def plot_sales_vs_platforms_boxplot(data):
    # Group by title and aggregate
    title_sales = (
        data.groupby('Name')
        .agg({'Platform': 'nunique', 'Global_Sales': 'sum'})
        .reset_index()
        .rename(columns={'Platform': 'Num_Platforms', 'Global_Sales': 'Total_Sales'})
    )

    # Boxplot of Total Sales vs. Number of Platforms
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=title_sales, x='Num_Platforms', y='Total_Sales')
    plt.title('Distribution of Total Sales by Number of Platforms')
    plt.xlabel('Number of Platforms')
    plt.ylabel('Total Sales (in Millions)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Call the function
plot_sales_vs_platforms_boxplot(data)


def calculate_platform_sales_correlation(data):
    # Group by title and aggregate
    title_sales = (
        data.groupby('Name')
        .agg({'Platform': 'nunique', 'Global_Sales': 'sum'})
        .reset_index()
        .rename(columns={'Platform': 'Num_Platforms', 'Global_Sales': 'Total_Sales'})
    )

    # Compute correlation
    correlation = title_sales['Num_Platforms'].corr(title_sales['Total_Sales'])
    print(f"Correlation between Number of Platforms and Total Sales: {correlation:.4f}")

    return correlation

# Call the function
correlation = calculate_platform_sales_correlation(data)


def plot_sales_by_platforms_and_region(data, region):
    """
    Plots sales by number of platforms for a specific region.
    """
    # Group by game title and calculate the number of platforms and total sales for the region
    title_sales_region = (
        data.groupby('Name')
        .agg({'Platform': 'nunique', region: 'sum'})
        .reset_index()
        .rename(columns={'Platform': 'Num_Platforms', region: 'Region_Sales'})
    )

    # Remove games with no sales in the region
    title_sales_region = title_sales_region[title_sales_region['Region_Sales'] > 0]

    # Plot a boxplot for the region
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=title_sales_region, x='Num_Platforms', y='Region_Sales', palette='viridis')
    plt.title(f'Sales by Number of Platforms for {region}', fontsize=16)
    plt.xlabel('Number of Platforms', fontsize=14)
    plt.ylabel(f'{region} Sales (in Millions)', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Calculate and print correlation
    correlation = title_sales_region['Num_Platforms'].corr(title_sales_region['Region_Sales'])
    print(f"Correlation between Number of Platforms and {region} Sales: {correlation:.4f}")

# Example: Plot for North America
plot_sales_by_platforms_and_region(data, 'NA_Sales')

# Example: Plot for Europe
plot_sales_by_platforms_and_region(data, 'EU_Sales')

# Example: Plot for Japan
plot_sales_by_platforms_and_region(data, 'JP_Sales')

# Example: Plot for Other regions
plot_sales_by_platforms_and_region(data, 'Other_Sales')


