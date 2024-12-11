#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns 
import altair as alt
from scipy.stats import randint
import os

# prep
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler, StandardScaler, label_binarize
from sklearn.preprocessing import OneHotEncoder

#Featureimportance
# need to install if you don't have 
# pip3 install yellowbrick
from yellowbrick.model_selection import FeatureImportances


# Validation libraries
from sklearn import metrics
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve

#models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

#%%
# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the CSV file
csv_file_path = os.path.join(current_dir, "Mental Health Dataset.csv")

# Read the CSV file
health_data = pd.read_csv(csv_file_path)

print("\nReady to continue.")
# %%
print(health_data.info())
health_data.head()
# %%
## Removing null values
print(health_data.isnull().sum())
print(health_data.dropna(inplace=True))
health_data.info()
# %%
# Clean Timestamp - Convert time and date to year

def process_timestamp_to_year(df, column_name):
    """
    Converts a Timestamp column in the DataFrame to numeric years.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the timestamp column.
        column_name (str): The name of the timestamp column.

    Returns:
        pd.DataFrame: The DataFrame with the processed column as numeric years.
    """
    # Convert the column to datetime
    df[column_name] = pd.to_datetime(df[column_name], format='%m/%d/%Y %H:%M', errors='coerce')
    
    # Extract the year
    df[column_name] = df[column_name].dt.year
    
    # Convert the years to numeric
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    
    return df

# Example usage
health_data = process_timestamp_to_year(health_data, 'Timestamp')

# Preview the result
print(health_data[['Timestamp']].head())
print(health_data['Timestamp'].info())

#%%
print(health_data['Timestamp'].unique())

# %%
# Cleaning Gender variable - Now Male -> 0 and Female -> 1

import pandas as pd

def process_gender_column(df, column_name):
    """
    Processes a Gender column in a DataFrame:
    - Maps "Female" to 1 and "Male" to 0.
    - Handles unexpected values by setting them to NaN or a default value.
    - Prints unique values and column info.

    Args:
        df (pd.DataFrame): The input DataFrame containing the Gender column.
        column_name (str): The name of the Gender column to process.

    Returns:
        pd.DataFrame: The updated DataFrame with the Gender column processed.
    """
    # Map Gender to numeric values, setting unexpected values to NaN
    df[column_name] = df[column_name].map({'Female': 1, 'Male': 0})
    
    # Print unique values and basic column info
    print(f"Unique values in '{column_name}': {df[column_name].unique()}")
    print(df[[column_name]].head())
    print(df[column_name].info())
    
    return df

# Example usage for processing the Gender column
health_data = process_gender_column(health_data, 'Gender')

# %%
# Cleaning the self_employed data
print(health_data['self_employed'].unique())
def process_self_employed_column(df, column_name):
    """
    Processes a self_employed column in a DataFrame:
    - Maps "Yes" to 1 and "No" to 0.
    - Handles unexpected values by setting them to NaN.
    - Prints unique values, first few rows, and column info.

    Args:
        df (pd.DataFrame): The input DataFrame containing the self_employed column.
        column_name (str): The name of the self_employed column to process.

    Returns:
        pd.DataFrame: The updated DataFrame with the self_employed column processed.
    """
    # Map 'Yes' to 1 and 'No' to 0, handle unexpected values by setting to NaN
    df[column_name] = df[column_name].map({'Yes': 1, 'No': 0})
    
    # Print unique values, first few rows, and column info
    print(f"Unique values in '{column_name}': {df[column_name].unique()}")
    print(df[[column_name]].head())
    print(df[column_name].info())
    
    return df

# Example usage for processing the self_employed column
health_data = process_self_employed_column(health_data, 'self_employed')


# %%
# Cleaning the family_history column
health_data['family_history'].unique()

def process_family_history_column(df, column_name):
    """
    Processes a family_history column in a DataFrame:
    - Maps "Yes" to 1 and "No" to 0.
    - Handles unexpected values by setting them to NaN.
    - Prints unique values, first few rows, and column info.

    Args:
        df (pd.DataFrame): The input DataFrame containing the family_history column.
        column_name (str): The name of the family_history column to process.

    Returns:
        pd.DataFrame: The updated DataFrame with the family_history column processed.
    """
    # Map 'Yes' to 1 and 'No' to 0, handle unexpected values by setting to NaN
    df[column_name] = df[column_name].map({'Yes': 1, 'No': 0})
    
    # Print unique values, first few rows, and column info
    print(f"Unique values in '{column_name}': {df[column_name].unique()}")
    print(df[[column_name]].head())
    print(df[column_name].info())
    
    return df

# Example usage for processing the family_history column
health_data = process_family_history_column(health_data, 'family_history')

# %%
# Clean the Days_Indoors
health_data['Days_Indoors'].unique()


def process_days_indoors_column(df, column_name):
    """
    Processes the 'Days_Indoors' column in a DataFrame by mapping each category 
    to a corresponding numerical value representing the approximate number of days.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the column to process.
        column_name (str): The name of the column to process.
    
    Returns:
        pd.DataFrame: The DataFrame with the processed column.
    """
    # Define the mapping
    mapping = {
        'Go out Every day': 365,
        'More than 2 months': 60,
        '31-60 days': 45,
        '15-30 days': 22.5,
        '1-14 days': 7.5
    }

    # Map the categories to numerical values
    df[column_name] = df[column_name].map(mapping)

    # Handle missing or unmapped values by setting a default (e.g., NaN or 0)
    df[column_name].fillna(-1, inplace=True)  # Replace NaN with -1 for clarity
    
    # Print the unique values, first few rows, and column info
    print(f"Unique values in '{column_name}': {df[column_name].unique()}")
    print(df[[column_name]].head())
    print(df[column_name].describe())
    
    return df

# Example usage of the function
health_data = process_days_indoors_column(health_data, 'Days_Indoors')


# %%
# Clean Treatment column
health_data['treatment'].unique()
def process_treatment_column(df, column_name):
    """
    Maps the 'treatment' column in a DataFrame to 0 and 1.
    'Yes' -> 1, 'No' -> 0.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the column to process.
        column_name (str): The name of the column to process.
    
    Returns:
        pd.DataFrame: The DataFrame with the processed column.
    """
    # Define the mapping
    mapping = {
        'Yes': 1,
        'No': 0
    }

    # Map the categories to numerical values
    df[column_name] = df[column_name].map(mapping)

    # Handle missing or unmapped values if necessary (e.g., replace NaN with a default value)
    df[column_name].fillna(-1, inplace=True)  # Replace NaN with -1 for clarity
    
    # Print unique values, first few rows, and column description
    print(f"Unique values in '{column_name}': {df[column_name].unique()}")
    print(df[[column_name]].head())
    print(df[column_name].describe())
    
    return df
# Apply the function to the treatment column
health_data = process_treatment_column(health_data, 'treatment')


# %%
# Clean changes_habits column
health_data['Changes_Habits'].unique()
def process_changes_habits_column(df, column_name):
    """
    Maps the 'Changes_Habits' column in a DataFrame to numerical values.
    'No' -> 0, 'Yes' -> 1, 'Maybe' -> 2.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the column to process.
        column_name (str): The name of the column to process.
    Returns:
        pd.DataFrame: The DataFrame with the processed column.
    """
    # Define the mapping
    mapping = {
        'No': 0,
        'Yes': 1,
        'Maybe': 2
    }

    # Map the categories to numerical values
    df[column_name] = df[column_name].map(mapping)


    # Print unique values, first few rows, and column description
    print(f"Unique values in '{column_name}': {df[column_name].unique()}")
    print(df[[column_name]].head())
    print(df[column_name].describe())
    
    return df
# Apply the function to the Changes_Habits column
health_data = process_changes_habits_column(health_data, 'Changes_Habits')



# %%
# Clean the Mood_Swings
health_data['Mood_Swings'].unique()

def map_mood_swings_column(df, column_name):
    """
    Maps the 'Mood_Swings' column in the DataFrame to numerical values.

    Parameters:
    - df (DataFrame): The DataFrame containing the column to be processed.
    - column_name (str): The name of the column to map.

    Returns:
    - df (DataFrame): The DataFrame with the column mapped to numerical values.
    """
    # Define the mapping
    mapping = {
        'Low': 0,
        'Medium': 1,
        'High': 2
    }
    
    # Map the column
    df[column_name] = df[column_name].map(mapping)
    
    # Display unique values, first rows, and info
    print(f"Unique values in '{column_name}': {df[column_name].unique()}")
    print(df[[column_name]].head())
    print(df[column_name].describe())
    
    return df

# Apply the function to the 'Mood_Swings' column
health_data = map_mood_swings_column(health_data, 'Mood_Swings')

# %%
# Clean the Mental_Health_History
health_data['Mental_Health_History'].unique()

def map_mental_health_history_column(df, column_name):
    """
    Maps the 'Mental_Health_History' column in the DataFrame to numerical values.

    Parameters:
    - df (DataFrame): The DataFrame containing the column to be processed.
    - column_name (str): The name of the column to map.

    Returns:
    - df (DataFrame): The DataFrame with the column mapped to numerical values.
    """
    # Define the mapping
    mapping = {
        'Yes': 1,
        'No': 0,
        'Maybe': 2
    }
    
    # Map the column
    df[column_name] = df[column_name].map(mapping)
    
    # Display unique values, first rows, and info
    print(f"Unique values in '{column_name}': {df[column_name].unique()}")
    print(df[[column_name]].head())
    print(df[column_name].describe())
    
    return df

# Apply the function to the 'Mental_Health_History' column
health_data = map_mental_health_history_column(health_data, 'Mental_Health_History')


# %%
# Clean Coping_Struggles column

health_data['Coping_Struggles'].unique()

def map_coping_struggles_column(df, column_name):
    """
    Maps the 'Coping_Struggles' column in the DataFrame to numerical values.

    Parameters:
    - df (DataFrame): The DataFrame containing the column to be processed.
    - column_name (str): The name of the column to map.

    Returns:
    - df (DataFrame): The DataFrame with the column mapped to numerical values.
    """
    # Define the mapping
    mapping = {
        'No': 0,
        'Yes': 1
    }
    
    # Map the column
    df[column_name] = df[column_name].map(mapping)
    
    # Display unique values, first rows, and info
    print(f"Unique values in '{column_name}': {df[column_name].unique()}")
    print(df[[column_name]].head())
    print(df[column_name].describe())
    
    return df

# Apply the function to the 'Coping_Struggles' column
health_data = map_coping_struggles_column(health_data, 'Coping_Struggles')

# %%
# Clean Work_Interest column

health_data['Work_Interest'].unique()

def map_work_interest_column(df, column_name):
    """
    Maps the 'Work_Interest' column in the DataFrame to numerical values.

    Parameters:
    - df (DataFrame): The DataFrame containing the column to be processed.
    - column_name (str): The name of the column to map.

    Returns:
    - df (DataFrame): The DataFrame with the column mapped to numerical values.
    """
    # Define the mapping
    mapping = {
        'No': 0,
        'Yes': 1,
        'Maybe': 2
    }
    
    # Map the column
    df[column_name] = df[column_name].map(mapping)
    
    # Display unique values, first rows, and info
    print(f"Unique values in '{column_name}': {df[column_name].unique()}")
    print(df[[column_name]].head())
    print(df[column_name].describe())
    
    return df

# Apply the function to the 'Work_Interest' column
health_data = map_work_interest_column(health_data, 'Work_Interest')
# %%
# Clean Social_Weakness
health_data['Social_Weakness'].unique()

def map_social_weakness_column(df, column_name):
    """
    Maps the 'Social_Weakness' column in the DataFrame to numerical values.

    Parameters:
    - df (DataFrame): The DataFrame containing the column to be processed.
    - column_name (str): The name of the column to map.

    Returns:
    - df (DataFrame): The DataFrame with the column mapped to numerical values.
    """
    # Define the mapping
    mapping = {
        'No': 0,
        'Maybe': 2,
        'Yes': 1
    }
    
    # Map the column
    df[column_name] = df[column_name].map(mapping)
    
    # Display unique values, first rows, and info
    print(f"Unique values in '{column_name}': {df[column_name].unique()}")
    print(df[[column_name]].head())
    print(df[column_name].describe())
    
    return df

# Apply the function to the 'Social_Weakness' column
health_data = map_social_weakness_column(health_data, 'Social_Weakness')

# %%
print(health_data['mental_health_interview'].unique())

def map_mental_health_interview(df, column_name):
    """
    Maps the 'Social_Weakness' column in the DataFrame to numerical values.

    Parameters:
    - df (DataFrame): The DataFrame containing the column to be processed.
    - column_name (str): The name of the column to map.

    Returns:
    - df (DataFrame): The DataFrame with the column mapped to numerical values.
    """
    # Define the mapping
    mapping = {
        'No': 0,
        'Maybe': 2,
        'Yes': 1
    }
    
    # Map the column
    df[column_name] = df[column_name].map(mapping)
    
    # Display unique values, first rows, and info
    print(f"Unique values in '{column_name}': {df[column_name].unique()}")
    print(df[[column_name]].head())
    print(df[column_name].describe())
    
    return df

# Apply the function to the 'Social_Weakness' column
health_data = map_social_weakness_column(health_data, 'mental_health_interview')

# %%
# Clean care_options column
health_data['care_options'].unique()

def map_care_options_column(df, column_name):
    """
    Maps the 'care_options' column in the DataFrame to numerical values.

    Parameters:
    - df (DataFrame): The DataFrame containing the column to be processed.
    - column_name (str): The name of the column to map.

    Returns:
    - df (DataFrame): The DataFrame with the column mapped to numerical values.
    """
    # Define the mapping
    mapping = {
        'No': 0,
        'Not sure': 2,
        'Yes': 1
    }
    
    # Map the column
    df[column_name] = df[column_name].map(mapping)
    
    # Display unique values, first rows, and info
    print(f"Unique values in '{column_name}': {df[column_name].unique()}")
    print(df[[column_name]].head())
    print(df[column_name].describe())
    
    return df

# Apply the function to the 'care_options' column
health_data = map_care_options_column(health_data, 'care_options')
#%%
# Clean the Growing_Stress column
health_data['Growing_Stress'].unique()

def map_growing_stress_column(df, column_name):
    """
    Maps the 'Growing_Stress' column in the DataFrame to numerical values.

    Parameters:
    - df (DataFrame): The DataFrame containing the column to be processed.
    - column_name (str): The name of the column to map.

    Returns:
    - df (DataFrame): The DataFrame with the column mapped to numerical values.
    """
    # Define the mapping
    mapping = {
        'No': 0,
        'Maybe': 2,
        'Yes': 1
    }
    
    # Map the column
    df[column_name] = df[column_name].map(mapping)
    
    # Display unique values, first rows, and info
    print(f"Unique values in '{column_name}': {df[column_name].unique()}")
    print(df[[column_name]].head())
    print(df[column_name].describe())
    
    return df

# Apply the function to the 'Growing_Stress' column
health_data = map_growing_stress_column(health_data, 'Growing_Stress')

# %%
#Verifying the cleaned dataset
health_data.describe()
print(health_data.info())
# %%[markdown]
# Visualisation for Proportion of each feature in dataset

sns.set_style("whitegrid")

# Columns to visualize
cols_to_visualize = ['Gender', 'self_employed', 'family_history', 'treatment', 'Days_Indoors',
                     'Growing_Stress', 'Changes_Habits', 'Mental_Health_History', 'Mood_Swings', 
                     'Coping_Struggles', 'Work_Interest', 'Social_Weakness','Occupation']

# Collect value counts for each column
counts = [health_data[col].value_counts() for col in cols_to_visualize]

# Define a refined darker palette
color_palette = [
    '#E07A5F',  # Terra Cotta
    '#81B29A',  # Sage Green
    '#F2CC8F',  # Sandy Gold
    '#3D405B',  # Charcoal
    '#F4A261'   # Peach Orange
]

# Create a 4x4 grid using GridSpec
fig = plt.figure(figsize=(20, 20))
gs = gridspec.GridSpec(4, 4, figure=fig)

# Generate pie charts
for i, (col, count) in enumerate(zip(cols_to_visualize, counts)):
    ax = fig.add_subplot(gs[i])  # Create a subplot in the ith grid cell
    ax.pie(
        count,
        labels=count.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=color_palette[:len(count)]  # Use palette based on the number of unique values
    )
    ax.set_title(col, fontsize=14, fontweight='bold', color='darkslategray')
    ax.grid(False)

# Adjust layout
plt.tight_layout()
plt.show()



# %%


bright_palette = ["#FFB74D", "#64B5F6", "#81C784", "#E57373", "#FFD54F", "#4DD0E1"]

# Set the Seaborn style to 'whitegrid' for a clean look
sns.set_style("whitegrid")

# Create the countplot
plt.figure(figsize=(20, 10))
ax = sns.countplot(x='Country', data=health_data, palette=bright_palette)  # Brighter custom palette

# Rotate x-axis labels for readability
plt.xticks(rotation=45, ha="right", fontsize=12)

# Add bar labels
for container in ax.containers:
    ax.bar_label(container, fontsize=14, padding=8, color='black', weight='bold')

# Add a clean, modern title and axis labels
ax.set_title('Distribution of Countries', fontsize=22, fontweight='bold', color='teal')
ax.set_xlabel('Country', fontsize=18, color='darkslategray')
ax.set_ylabel('Count', fontsize=18, color='darkslategray')

# Adjust the gridlines to be subtle and light
ax.grid(visible=True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

# Improve layout to prevent clipping of labels
plt.tight_layout()
plt.show()

# %%[markdown]
# Growing stress vs allvariables now

#%%[markdown]
#Growing stress by gender
df = pd.crosstab(health_data.Growing_Stress,health_data.Gender).plot(kind = 'bar',color = ['m','c'])
df.grid(False)
df.set_title('Growing stress by Gender ')

#%%[markdown]
# Most of the males seems to have a 'Maybe' category in Growing stress and most of the
# female have 'Yes' category in Growing category 
#%%[markdown]

# Group the data by 'Growing_Stress' and 'Days_Indoors' and count the occurrences
#Icount = health_data.groupby(['Growing_Stress', 'Days_Indoors']).size().reset_index(name='Count')

# Create the bar plot using Altair
#import altair as alt

#chart = alt.Chart(Icount).mark_bar().encode(
#    x='Count:Q',
#    y='Days_Indoors:N',
#    color='Growing_Stress:N',
#    tooltip=['Growing_Stress', 'Count']
#).properties(
#    title='Distribution of Growing Stress by Days Indoors',
#    width=600,
#    height=400
#).configure_mark(
#    opacity=0.7  # Subtle opacity for the bars
#).configure_title(
#    fontSize=18, 
#    font='Arial', 
#    anchor='middle', 
#    color='gray'
#)

# Show the chart
#chart.show()
# %%[markdown]
# Growing stress by Days spent in indoors

count_data = health_data.groupby(['Growing_Stress', 'Days_Indoors']).size().reset_index(name='Count')

custom_palette = ['#FF6F81',
                  '#6DAED9',
                  '#D17000']

# Create the barplot
plt.figure(figsize=(15, 8))
ax = sns.barplot(
    data=count_data,
    x='Days_Indoors',
    y='Count',
    hue='Growing_Stress',
    palette=custom_palette
)

# Rotate x-axis labels for better readability
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add bar labels for clarity
for container in ax.containers:
    ax.bar_label(container, fmt='%.0f', fontsize=10, padding=3)

# Set titles and labels
ax.set_title('Distribution of Growing Stress by Days Indoors', fontsize=18, color='darkslategray', weight='bold')
ax.set_xlabel('Days Indoors', fontsize=14, color='gray')
ax.set_ylabel('Count', fontsize=14, color='gray')
ax.legend(title='Growing Stress', fontsize=12, title_fontsize=12)

# Remove gridlines for a cleaner look
sns.despine()
plt.tight_layout()

# Show the plot
plt.show()

#%%[markdown]
# Growing stress vs occupation

count_data = health_data.groupby(['Growing_Stress', 'Occupation']).size().reset_index(name='Count')

# Set Seaborn style
sns.set_style("whitegrid")

# Create custom color palette
custom_palette = {0: '#FF6F81', 1: '#6DAED9', 2: '#D17000'}

# Create the bar plot
plt.figure(figsize=(15, 8))
ax = sns.barplot(
    data=count_data,
    x='Count',  # Count for the x-axis
    y='Occupation',  # Occupation for the y-axis
    hue='Growing_Stress',  # Hue for 'Growing_Stress'
    palette=custom_palette
)

# Add labels to the bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.0f', fontsize=10, padding=3)

# Customize plot title and labels
ax.set_title('Distribution of Growing Stress by Occupations', fontsize=18, fontweight='bold', color='darkslategray')
ax.set_xlabel('Count', fontsize=14, color='gray')
ax.set_ylabel('Occupation', fontsize=14, color='gray')
ax.legend(title='Growing Stress', fontsize=12, title_fontsize=12)

# Display the plot
plt.tight_layout()
plt.show()
#%%[markdown]
# Corporate employees report the highest levels of growing stress, while housewives and students experience the 
# most uncertainty, often categorized as 'maybe' stress
# %%[markdown]
# Growing stress vs work interest

color_map = {1: '#FF6F81', 0: '#6DAED9', 2: '#D17000'}

# Create the plot
plt.figure(figsize=(8, 8))
ax1 = sns.countplot(x='Work_Interest', hue='Growing_Stress', data=health_data, palette=color_map)

# Add title and remove grid
plt.title('Growing Stress by Work Interest')
plt.grid(False)

# Add labels to bars
for container in ax1.containers:
    ax1.bar_label(container, label_type='edge')

# Move the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.show()

#%%[markdown]
# The people with maybe work interest seem to have the highest growing stress followed by 
# people with 'yes' work interest 
# %%[markdown]
# The growing stress vs chnages_habits

plt.figure(figsize=(8, 8))
ax2 = sns.countplot(x='Changes_Habits', hue='Growing_Stress', data=health_data,palette="viridis")
plt.title('Growing Stress by Changes_Habits')
plt.grid(False)
for container in ax2.containers:
    ax2.bar_label(container, label_type='edge')

# Move the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

#%%[markdown]
# The changes_habits 'maybe' category has the highest growing stress population who doesnot change habits.

# %%[markdown]
# Growing stress vs Self_employed
plt.figure(figsize=(8, 8))
ax2 = sns.countplot(x='self_employed', hue='Growing_Stress', data=health_data,palette="viridis")
plt.title('Growing Stress by Cself employed')
plt.grid(False)
for container in ax2.containers:
    ax2.bar_label(container, label_type='edge')

# Move the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

#%%[markdown]
# The data reveals that non-self-employed individuals exhibit higher levels of growing stress compared to no growing stress. 
# Conversely, self-employed individuals display a balanced distribution, with equal proportions experiencing growing stress and no growing stress.

#%%[markdown]
# Growing stress vs Family history
plt.figure(figsize=(8, 8))
ax2 = sns.countplot(x='family_history', hue='Growing_Stress', data=health_data,palette="viridis")
plt.title('Growing Stress by family history')
plt.grid(False)
for container in ax2.containers:
    ax2.bar_label(container, label_type='edge')

# Move the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
# %%[markdown]
# The data reveals that individuals with family history exhibit higher levels of growing stress compared to no growing stress. 
# and individuals with no family history also display high levels of growing stress to no growing stress.
#%%[markdown]
# Growing stress vs Treatment

plt.figure(figsize=(8, 8))
ax2 = sns.countplot(x='treatment', hue='Growing_Stress', data=health_data,palette="viridis")
plt.title('Growing Stress by treatment')
plt.grid(False)
for container in ax2.containers:
    ax2.bar_label(container, label_type='edge')

# Move the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# %%[markdown]
# The data reveals that individuals who had treatment exhibit higher levels of growing stress compared to no growing stress and the 
# individuals with no treatment also display a higher levels of growing stress to no growing stress.

#%%[markdown]

plt.figure(figsize=(8, 8))
ax2 = sns.countplot(x='Mood_Swings', hue='Growing_Stress', data=health_data,palette="viridis")
plt.title('Growing Stress by treatment')
plt.grid(False)
for container in ax2.containers:
    ax2.bar_label(container, label_type='edge')

# Move the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# %%[markdown]
# The people with medium stress levels have highest growing stress compared to people with low and high mood 

#%%[markdown]
# Growing stress vs coping struggles
plt.figure(figsize=(8, 8))
ax2 = sns.countplot(x='Coping_Struggles', hue='Growing_Stress', data=health_data,palette="viridis")
plt.title('Growing Stress by Coping struggles')
plt.grid(False)
for container in ax2.containers:
    ax2.bar_label(container, label_type='edge')

# Move the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# %%[markdown]
# Surprisingly the people who donot struggle with coping exhbit more growing stress levels

#%%[markdown]

# Growing stress vs Care options
plt.figure(figsize=(8, 8))
ax2 = sns.countplot(x='care_options', hue='Growing_Stress', data=health_data,palette="viridis")
plt.title('Growing Stress by care options')
plt.grid(False)
for container in ax2.containers:
    ax2.bar_label(container, label_type='edge')

# Move the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
# %%[markdown]
# The people with care options exhibit growing stress levels compared to others 

#%%[markdown]
plt.figure(figsize=(8, 8))
ax2 = sns.countplot(x='Mental_Health_History', hue='Growing_Stress', data=health_data,palette="viridis")
plt.title('Growing Stress by Mental health history')
plt.grid(False)
for container in ax2.containers:
    ax2.bar_label(container, label_type='edge')

# Move the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
# %%[markdown]

# The people with mental health history have growing stress compared to other people.

#%%[markdown]
# Growing stress vs Timestamp
health_data['Timestamp'].unique()
#%%

# Group by 'Timestamp' and 'Growing_Stress', then count occurrences
grouped_data = health_data.groupby(['Timestamp', 'Growing_Stress']).size().reset_index(name='Count')

# Map the Growing_Stress categories to readable names (for the legend)
category_map = {0: 'No', 1: 'Yes', 2: 'Maybe'}
grouped_data['Growing_Stress_Label'] = grouped_data['Growing_Stress'].map(category_map)

# Create the lineplot
plt.figure(figsize=(12, 6))
sns.lineplot(
    x='Timestamp',
    y='Count',
    hue='Growing_Stress_Label',  # Hue based on the mapped Growing_Stress categories
    data=grouped_data,
    palette={'No': '#6DAED9', 'Yes': '#FF6F81', 'Maybe': '#D17000'}
)

# Customize the plot
plt.title('Growing Stress Over Time by Category', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Count of Growing Stress Cases', fontsize=12)
plt.legend(title='Growing Stress', fontsize=10, title_fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()



#%%[markdown]
# Growing stress vs Social_weakness

plt.figure(figsize=(8, 8))
ax2 = sns.countplot(x='Social_Weakness', hue='Growing_Stress', data=health_data,palette="viridis")
plt.title('Growing Stress by Social_Weakness')
plt.grid(False)
for container in ax2.containers:
    ax2.bar_label(container, label_type='edge')

# Move the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
#%%[markdown]
# The people who have social weakness have growing stress compared to others

#%%[markdown]
# EDA specifically for the Student occupation
# Filter data for 'Student' occupation
student_data = health_data[health_data['Occupation'] == 'Student']

# Check the first few rows of the filtered data
student_data.head()

#%%[markdown]
# Distribution of Days_Indoors for Students
# Plot distribution of 'Days_Indoors' for students
sns.histplot(student_data['Days_Indoors'], kde=True, color='skyblue')
plt.title('Distribution of Days Indoors for Students')
plt.xlabel('Days Indoors')
plt.ylabel('Frequency')
plt.show()

#%%[markdown]
# Treatment vs Growing Stress for students
sns.countplot(x='Growing_Stress', hue='treatment', data=student_data, palette='Set2')
plt.title('Treatment vs Growing Stress for Students')
plt.xlabel('Growing Stress')
plt.ylabel('Count')
plt.show()

#%%[markdown]
# Changes in Habits vs Growing Stress for students
sns.countplot(x='Growing_Stress', hue='Changes_Habits', data=student_data, palette='Set2')
plt.title('Changes in Habits vs Growing Stress for Students')
plt.xlabel('Growing Stress')
plt.ylabel('Count')

# Move the legend (hue label) to the top-left
plt.legend(title='Changes in Habits', loc='upper left', bbox_to_anchor=(0, 1))
plt.show()


#%%[markdown]
# Coping Struggles vs Growing Stress for students
sns.countplot(x='Growing_Stress', hue='Coping_Struggles', data=student_data, palette='Set2')
plt.title('Coping Struggles vs Growing Stress for Students')
plt.xlabel('Growing Stress')
plt.ylabel('Count')
plt.show()

#%%[markdown]
# Social Weakness vs Growing Stress for students
sns.countplot(x='Growing_Stress', hue='Social_Weakness', data=student_data, palette='Set2')
plt.title('Social Weakness vs Growing Stress for Students')
plt.xlabel('Growing Stress')
plt.ylabel('Count')
plt.show()

#%%[markdown]
# Mood Swings vs Growing Stress for students
sns.countplot(x='Growing_Stress', hue='Mood_Swings', data=student_data, palette='Set2')
plt.title('Mood Swings vs Growing Stress for Students')
plt.xlabel('Growing Stress')
plt.ylabel('Count')
plt.show()

#%%[markdown]
# Gender vs Growing Stress for students
sns.countplot(x='Growing_Stress', hue='Gender', data=student_data, palette='Set2')
plt.title('Gender vs Growing Stress for Students')
plt.xlabel('Growing Stress')
plt.ylabel('Count')
plt.show()

#%%[markdown]
# Days Indoors vs Growing Stress for students
sns.boxplot(x='Growing_Stress', y='Days_Indoors', data=student_data, palette='Set2')
plt.title('Days Indoors vs Growing Stress for Students')
plt.xlabel('Growing Stress')
plt.ylabel('Days Indoors')
plt.show()

#%%[markdown]
# Correlation heatmap for variables correlated with Growing Stress for students
student_corr = student_data[['Days_Indoors', 'Growing_Stress', 'Changes_Habits', 'Coping_Struggles', 
                             'Mood_Swings', 'Social_Weakness', 'treatment']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(student_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap for Growing Stress and Related Variables for Students')
plt.show()



#%%[markdown]
# Statistical tests
from scipy.stats import chi2_contingency, chi2

cols= ['Timestamp','Gender','Country','Occupation','self_employed','family_history', 'treatment', 'Days_Indoors',
                    'Changes_Habits', 'Mental_Health_History', 'Mood_Swings','Coping_Struggles', 'Work_Interest', 'Social_Weakness','mental_health_interview','care_options']
def calculate_chi_square(column1,column2 = 'Growing_Stress'):
    print(f"Correlation between **{column1}** and **{column2}**\n")
    crosstab = pd.crosstab(health_data[column1],health_data[column2])
    stat,p,dof,expected = chi2_contingency(crosstab, correction = True)
    print(f'P_value = {p}, degrees of freedom =  {dof}')
    prob = 0.95
    critical = chi2.ppf(prob,dof)
    print(f'probability = %.3f, critical = %.3f, stat = %.3f' %(prob,critical,stat) )
    if stat >= critical:
        print('dependent(reject(Ho))')
    else:
        print('independent(accept(Ho))')
    alpha = 1.0 - prob
    print(f'significance = %.3f, p-value = %.3f' %(alpha,p))
    if p <= alpha:
        print('dependent(reject(Ho))')
    else:
        print('independent(accept(Ho))')
    print('\n-----------------------------------\n')
print('** Chi_square Correlation between Dichotomous features with Target:Growing stress**\n')
for col in cols:
    calculate_chi_square(col)
# %%
# Gender, occupation, family_history,treatment, Days_indoors, changes_habits, mental_health_history,
# mood_swings, coping_struggles, work_interest, social weakness, mental_health_interview, care_options
# have a significant p-value suggesting that the growing stress levels for these groups is different for
# different categories. We are also removing the gender variable as it has unequal distribution


#%%[markdown]
# Before the unknown factor to create the normal model without the unknown factor

#%%[markdown]
# Random Forest -haeyeon

# Create a copy of the health_data to avoid modifying the original
health_data_copy = health_data.copy()

# Filter data to only include students
health_data_students = health_data_copy[health_data_copy['Occupation'] == 'Student']

# Preprocessing the dataset
# Drop 'Timestamp' and other irrelevant columns for prediction
X = health_data_students.drop(['Growing_Stress', 'Timestamp', 'Country', 'Occupation'], axis=1)  # Drop target and irrelevant columns
y = health_data_students['Growing_Stress']  # Target column

# Convert categorical columns to numeric (if any)
X = pd.get_dummies(X, drop_first=True)  # One-hot encoding for categorical features

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features (Important for Random Forest and other models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Get predicted probabilities for each class
y_pred_proba = rf.predict_proba(X_test_scaled)

# One-vs-Rest approach: Binarize the target labels for multiclass classification
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])  # Binarize the target labels (0, 1, 2)

# Compute AUC for each class (One-vs-Rest)
roc_auc = roc_auc_score(y_test_bin, y_pred_proba, average='macro', multi_class='ovr')

# Plot the ROC curve for each class
plt.figure(figsize=(8, 6))
for i in range(3):  # 3 classes: 0, 1, and 2
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc_score(y_test_bin[:, i], y_pred_proba[:, i]):.2f})')

# Plot diagonal line for random classifier
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

# Set plot labels and title
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC Curve (One-vs-Rest) for Students')
plt.legend(loc='lower right')
plt.show()

# Print the Macro AUC score
print(f"Macro AUC-ROC score for Students: {roc_auc:.2f}")
#%%[markdown]
# A Macro AUC-ROC score of 1.00, might be because of Overfitting, Data Imbalance(Growing_Stress (with values 0, 1, 2)
# We tried additioanl validation

#%%[markdown]
# Perform cross-validation (e.g., 5-fold)
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='roc_auc_ovr')
print(f'Cross-validated AUC-ROC scores: {cv_scores}')
print(f'Mean AUC-ROC score from cross-validation: {cv_scores.mean():.2f}')

# Check Class Distribution
print(y.value_counts())

# Confusion Matrix
# Generate confusion matrix
cm = confusion_matrix(y_test, rf.predict(X_test_scaled))
print(cm)

#%%[markdown]
# The AUC scores across folds range from 0.45 to 0.75, with a mean AUC of 0.62
# Class 2: 22,915 samples, Class 1: 21,424 samples, Class 0: 16,348 samples <- little imblanced, but not extreme
# Class 0 (Stress Level 0): The model has made no misclassifications, indicating that it is very confident when predicting the "no stress" class.
# Class 1 (Stress Level 1): The model is performing fairly well, with a few misclassifications (12).
# Class 2 (Stress Level 2): The model is performing quite well here too, with a very small number of misclassifications (17)



# %%
# Logisticregression before Uknown factor
# Logistic Regression after Removing 'Maybe' from Growing_Stress
def logistic_regression_filtered():
    """
    Train and evaluate logistic regression model after removing 'Maybe' from Growing_Stress with detailed evaluation.
    """
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        roc_curve,
        confusion_matrix,
        classification_report
    )
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Filter out 'Maybe' (2) from the Growing_Stress column
    filtered_data = health_data[health_data['Growing_Stress'] != 2]

    # Drop non-numeric columns and target variable
    X_filtered = filtered_data.drop(['Growing_Stress', 'Timestamp', 'Country', 'Occupation'], axis=1, errors='ignore')
    y_filtered = filtered_data['Growing_Stress']

    # Encode categorical variables
    X_filtered = pd.get_dummies(X_filtered, drop_first=True)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)

    # Train Logistic Regression Model
    logit_model = LogisticRegression(max_iter=1000, random_state=42)
    logit_model.fit(X_train, y_train)

    # Predictions
    y_pred = logit_model.predict(X_test)
    y_pred_prob = logit_model.predict_proba(X_test)[:, 1]

    # Evaluate Performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print Results
    print("\nLogistic Regression Results (after removing 'Maybe'):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # ROC Curve Visualization
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Filtered 'Maybe')")
    plt.legend(loc="lower right")
    plt.show()

# Call the function to train and evaluate the logistic regression model
logistic_regression_filtered()





# %%
def knn_before_unknown_factor():
    """
    Train and evaluate KNN model before considering the unknown factor, with detailed metrics and measures to address overfitting.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
    )
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # Filter out 'Maybe' (2) from the Growing_Stress column
    filtered_data = health_data[health_data['Growing_Stress'] != 2]

    # Drop non-numeric columns and target variable
    X_filtered = filtered_data.drop(['Growing_Stress', 'Timestamp', 'Country', 'Occupation'], axis=1, errors='ignore')
    y_filtered = filtered_data['Growing_Stress']

    # Encode categorical variables
    X_filtered = pd.get_dummies(X_filtered, drop_first=True)

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_filtered, test_size=0.2, random_state=42)

    # Train KNN Model with adjusted parameters to reduce overfitting
    knn_model = KNeighborsClassifier(n_neighbors=30, weights='distance')
    knn_model.fit(X_train, y_train)
  
    # Predictions
    y_pred = knn_model.predict(X_test)
    y_pred_prob = knn_model.predict_proba(X_test)[:, 1]

    # Evaluate Performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print Results
    print("\nKNN Results (Before Unknown Factor):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
  # Confusion Matrix Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # ROC Curve Visualization
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Before Unknown Factor)")
    plt.legend(loc="lower right")
    plt.show()
knn_before_unknown_factor()
# %%
def knn_before_unknown_factor():
    """
    Train and evaluate KNN model before considering the unknown factor, with detailed metrics and measures to address overfitting.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
    )
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np

    # Filter out 'Maybe' (2) from the Growing_Stress column
    filtered_data = health_data[health_data['Growing_Stress'] != 2]

    # Drop non-numeric columns and target variable
    X_filtered = filtered_data.drop(['Growing_Stress', 'Timestamp', 'Country', 'Occupation'], axis=1, errors='ignore')
    y_filtered = filtered_data['Growing_Stress']

    # Encode categorical variables
    X_filtered = pd.get_dummies(X_filtered, drop_first=True)

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_filtered, test_size=0.2, random_state=42)

    # Train KNN Model with adjusted parameters to reduce overfitting
    knn_model = KNeighborsClassifier(n_neighbors=30, weights='distance')
    knn_model.fit(X_train, y_train)

    # Cross-Validation
    cross_val_scores = cross_val_score(knn_model, X_scaled, y_filtered, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy (5-Fold): {np.mean(cross_val_scores):.4f}")

    # Predictions
    y_pred = knn_model.predict(X_test)
    y_pred_prob = knn_model.predict_proba(X_test)[:, 1]

    # Evaluate Performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print Results
    print("\nKNN Results (Before Unknown Factor):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # ROC Curve Visualization
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Before Unknown Factor)")
    plt.legend(loc="lower right")
    plt.show()

# Call the functions to train and evaluate KNN models
knn_before_unknown_factor()

# %%

    # Filter out 'Maybe' (2) from the Growing_Stress column
filtered_data = health_data[health_data['Growing_Stress'] != 2]
numeric_data = filtered_data.select_dtypes(include=['number'])

# Calculate correlation
corr_matrix = numeric_data.corr()
print(corr_matrix['Growing_Stress'].sort_values(ascending=False))

# 

#%%[markdown]
# We have a the "unknown factor" quantifies the portion of the variance in Growing_Stress that is unexplained by the selected features.
# (i) Calculate the weighted probability of all features, ensuring the total feature weights sum to 1 and the 
# conditional probability if found for each feature.
# (ii) If the total weights are less than 1, it indicates growing stress due to personal reasons. 
# (iii) This suggests that an unknown factor is impacting the growing stress. And create a new feature unknown factor.
# (iv) This will help us have a more certainity in Growing factor by utilising the unknown factor to bring the 
# Growing stress from 3 classes to 2 classes
# 
#%%[markdown]
#caluclate the conditional probability
cols = health_data[['Occupation', 'self_employed', 'family_history', 'treatment', 'Days_Indoors',
                     'Changes_Habits', 'Mental_Health_History', 'Growing_Stress', 'Mood_Swings', 'Coping_Struggles',
                     'Work_Interest', 'Social_Weakness', 'mental_health_interview', 'care_options']]

def calculate_conditional_pro(feature1, feature2='Growing_Stress'):
    column = pd.crosstab(index=health_data[feature1], columns=health_data[feature2], normalize='columns')
    print(f"Probability of {feature1} given {feature2}:\n")
    print(f'Column conditional probability:\n{column}')
    print('\n------------------------------------------\n')
    return column

for con in cols:
    data = calculate_conditional_pro(con)
#%%[markdown]
# Create DataFrame
import pandas as pd
df = pd.DataFrame(health_data)

# Selected features
selected_features = ['Timestamp','Gender', 'Country', 'Occupation', 'self_employed', 'family_history', 'treatment', 'Days_Indoors',
                     'Changes_Habits', 'Mental_Health_History', 'Growing_Stress', 'Mood_Swings', 'Coping_Struggles',
                     'Work_Interest', 'Social_Weakness', 'mental_health_interview', 'care_options']

# Calculate the frequency of each feature value given each target value
conditional_probs = {feature: {} for feature in selected_features}

for feature in selected_features:
    for target in df['Growing_Stress'].unique():
        conditional_probs[feature][target] = {}
        for value in df[feature].unique():
            prob = df[(df[feature] == value) & (df['Growing_Stress'] == target)].shape[0] / df[df['Growing_Stress'] == target].shape[0]
            conditional_probs[feature][target][value] = prob

# Calculate the average conditional probability for each feature for 'Yes' and 'No'
avg_cond_probs = {}
total_avg_cond_prob = {}
for target in df['Growing_Stress'].unique():
    avg_cond_probs[target] = {}
    total_avg_cond_prob[target] = 0
    for feature in selected_features:
        avg_cond_probs[target][feature] = np.mean(list(conditional_probs[feature][target].values()))
        total_avg_cond_prob[target] += avg_cond_probs[target][feature]
weights = {}
for target in df['Growing_Stress'].unique():
    weights[target] = {}
    for feature in selected_features:
        weights[target][feature] = avg_cond_probs[target][feature] / total_avg_cond_prob[target]

# Create temporary columns for each feature probability
for target in df['Growing_Stress'].unique():
    for feature in selected_features:
        df[f'{feature}_prob_{target}'] = df[feature].map(conditional_probs[feature][target])

# Calculate the combined new feature as a weighted sum of the conditional probabilities for both 'Yes' and 'No'
df['New_Feature'] = sum(df[f'{feature}_prob_1'] * weights[1][feature] + df[f'{feature}_prob_0'] * weights[0][feature] for feature in selected_features)
df['unknown_factor'] = 1 - df['New_Feature']
# Keep only necessary columns
final_df = df[['Occupation',  'family_history', 'treatment', 'Days_Indoors',
                     'Changes_Habits', 'Mental_Health_History', 'Mood_Swings', 'Coping_Struggles',
                     'Work_Interest', 'Social_Weakness', 'mental_health_interview', 'care_options', 'unknown_factor','Growing_Stress']]

final_df.head()

#%%
# Group by Growing_Stress and calculate the mean of unknown_factor
mean_unknown_factor = final_df.groupby('Growing_Stress')['unknown_factor'].mean()
print(mean_unknown_factor)

print(final_df['unknown_factor'].median())
threshold = final_df['unknown_factor'].median()

print(threshold)

#%%
# Set threshold based on the mean or a slightly higher value
  # or use another positive threshold based on analysis

# Convert Growing_Stress from 2 to 0 and 1 based on unknown_factor
final_df['Growing_Stress'] = final_df.apply(lambda row: 1 if row['unknown_factor'] > threshold else 0, axis=1)

# Check the updated values
print(final_df['Growing_Stress'].value_counts())

#%%
final_df = final_df.drop('unknown_factor',axis = 1)
#%%

final_df['Occupation'] = final_df['Occupation'].astype(str)


print(final_df['Occupation'].unique())
final_df.isnull().sum()

#%%
# One-Hot Encoding using Pandas

categorical_columns = final_df.select_dtypes(include=['object']).columns.tolist()

#%%
occupation_encoded = pd.get_dummies(final_df['Occupation'], prefix='Occupation', drop_first=True)

# Step 2: Concatenating the encoded DataFrame with the original DataFrame (excluding the original Occupation column)
encoded_final_df = pd.concat([final_df.drop('Occupation', axis=1), occupation_encoded], axis=1)
encoded_final_df.head()
#%%

data = encoded_final_df.drop('Growing_Stress', axis=1)
f_data = encoded_final_df['Growing_Stress']
X = data
y = f_data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, f_data, test_size=0.2, random_state=42)
methodDict = {}

#%%
logit = LogisticRegression()  # instantiate
logit.fit( Xtrain, Ytrain )
print('Logit model accuracy (with the test set):', logit.score(Xtest, Ytest))
print('Logit model accuracy (with the train set):', logit.score(Xtrain, Ytrain))


#%%[markdown]
# Evaluation of the model
def evalClassModel(model, y_test, y_pred_class, plot=False):
    #Classification accuracy: percentage of correct predictions
    # calculate accuracy
    print('Accuracy:', metrics.accuracy_score(y_test, y_pred_class))
    
    #Null accuracy: accuracy that could be achieved by always predicting the most frequent class
    # examine the class distribution of the testing set (using a Pandas Series method)
    print('Null accuracy:\n', y_test.value_counts())
    
    # calculate the percentage of ones
    print('Percentage of ones:', y_test.mean())
    
    # calculate the percentage of zeros
    print('Percentage of zeros:',1 - y_test.mean())
    
    #Comparing the true and predicted response values
    print('True:', y_test.values[0:25])
    print('Pred:', y_pred_class[0:25])
    
    #Confusion matrix
    # save confusion matrix and slice into four pieces
    confusion = metrics.confusion_matrix(y_test, y_pred_class)
    #[row, column]
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    
    # visualize Confusion Matrix
    sns.heatmap(confusion,annot=True,fmt="d") 
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    #Metrics computed from a confusion matrix
    #Classification Accuracy: Overall, how often is the classifier correct?
    accuracy = metrics.accuracy_score(y_test, y_pred_class)
    print('Classification Accuracy:', accuracy)
    
    #Classification Error: Overall, how often is the classifier incorrect?
    print('Classification Error:', 1 - metrics.accuracy_score(y_test, y_pred_class))
    
    #False Positive Rate: When the actual value is negative, how often is the prediction incorrect?
    false_positive_rate = FP / float(TN + FP)
    print('False Positive Rate:', false_positive_rate)
    
    #Precision: When a positive value is predicted, how often is the prediction correct?
    print('Precision:', metrics.precision_score(y_test, y_pred_class))
    
    
    # IMPORTANT: first argument is true values, second argument is predicted probabilities
    print('AUC Score:', metrics.roc_auc_score(y_test, y_pred_class))
    
    # calculate cross-validated AUC
    # Change cross valisation to k -fold cross vaidation
    print('Cross-validated AUC:', cross_val_score(model, X, y, cv=10, scoring='roc_auc').mean())
    
    ##########################################
    #Adjusting the classification threshold
    ##########################################
    # print the first 10 predicted responses
    print('First 10 predicted responses:\n', model.predict(Xtest)[0:10])

    # print the first 10 predicted probabilities of class membership
    print('First 10 predicted probabilities of class members:\n', model.predict_proba(Xtest)[0:10])

    # print the first 10 predicted probabilities for class 1
    model.predict_proba(Xtest)[0:10, 1]
    
    # store the predicted probabilities for class 1
    y_pred_prob = model.predict_proba(Xtest)[:, 1]
    
    if plot == True:
        # histogram of predicted probabilities
        plt.rcParams['font.size'] = 12
        plt.hist(y_pred_prob, bins=8)
        
        # x-axis limit from 0 to 1
        plt.xlim(0,1)
        plt.title('Histogram of predicted probabilities')
        plt.xlabel('Predicted probability of treatment')
        plt.ylabel('Frequency')
    
    
    # predict treatment if the predicted probability is greater than 0.3
    # it will return 1 for all values above 0.3 and 0 otherwise
    # results are 2D so we slice out the first column
    y_pred_prob = y_pred_prob.reshape(-1,1) 
    y_pred_class = binarize(y_pred_prob, threshold=0.3)[0]
    
    # print the first 10 predicted probabilities
    print('First 10 predicted probabilities:\n', y_pred_prob[0:10])
    
    ##########################################
    #ROC Curves and Area Under the Curve (AUC)
    ##########################################
    
    #AUC is the percentage of the ROC plot that is underneath the curve
    #Higher value = better classifier
    roc_auc = metrics.roc_auc_score(y_test, y_pred_prob)
    
    

    # IMPORTANT: first argument is true values, second argument is predicted probabilities
    # roc_curve returns 3 objects fpr, tpr, thresholds
    # fpr: false positive rate
    # tpr: true positive rate
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
    if plot == True:
        plt.figure()
        
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.rcParams['font.size'] = 12
        plt.title('ROC curve for treatment classifier')
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.legend(loc="lower right")
        plt.show()
    
    # define a function that accepts a threshold and prints sensitivity and specificity
    def evaluate_threshold(threshold):
        #Sensitivity: When the actual value is positive, how often is the prediction correct?
        #Specificity: When the actual value is negative, how often is the prediction correct?print('Sensitivity for ' + str(threshold) + ' :', tpr[thresholds > threshold][-1])
        print('Specificity for ' + str(threshold) + ' :', 1 - fpr[thresholds > threshold][-1])

    # One way of setting threshold
    predict_mine = np.where(y_pred_prob > 0.50, 1, 0)
    confusion = metrics.confusion_matrix(y_test, predict_mine)
    print(confusion)
    
    
    
    return accuracy
#
#%%[markdown]
# Logistic regression

def logisticRegression():
    # train a logistic regression model on the training set
    logreg = LogisticRegression()
    logreg.fit(Xtrain, Ytrain)
    
    # make class predictions for the testing set
    y_pred_class = logreg.predict(Xtest)
    
    accuracy_score = evalClassModel(logreg, Ytest, y_pred_class, True)
    
    #Data for final graph
    methodDict['Log. Regression'] = accuracy_score * 100
    
logisticRegression()

#%%
model = LogisticRegression(max_iter=1000)

# Create the feature importances visualizer
visualizer = FeatureImportances(model)

# Fit the visualizer and the model
visualizer.fit(Xtrain, Ytrain)

# Show the visualization
visualizer.show()




#%%[markdown]
def plot_feature_importance(model, feature_names):
    """
    Plots the feature importance of a trained RandomForest model.
    :param model: Trained RandomForestClassifier
    :param feature_names: List of feature names
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.xlabel("Features")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.show()

#%%[markdown]
# Random Forest
def randomForest():
    # Calculating the best parameters
    forest = RandomForestClassifier(n_estimators = 20)

    featuresSize = Xtrain.shape[1]
    param_dist = {"max_depth": [3, None],
              "max_features": randint(1, featuresSize),
              "min_samples_split": randint(2, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}
    
    # Building and fitting my_forest
    forest = RandomForestClassifier(max_depth = None, min_samples_leaf=8, min_samples_split=2, n_estimators = 20, random_state = 1)
    my_forest = forest.fit(Xtrain, Ytrain)
    
    # make class predictions for the testing set
    y_pred_class = my_forest.predict(Xtest)
    
    accuracy_score = evalClassModel(my_forest, Ytest, y_pred_class, True)

    #Data for final graph
    methodDict['Random Forest'] = accuracy_score * 100

    feature_names = Xtrain.columns if hasattr(Xtrain, 'columns') else [f"Feature {i}" for i in range(Xtrain.shape[1])]
    plot_feature_importance(my_forest, feature_names)
randomForest()







# %%[makrdown]
# Feature importance and selection




# %%[markdown]
# As seen from the summary Gender plays a very important role and contributes the most to the Growing stress followed by
# Mental health interview and family history positive coefficent meaning the ones who have given the interview 
# and have a familty hisroty of mental health could have more growing stress


#%%SVM prep (Yonathan)



#Let's build a scatterplot to visualize the data and understand certain features of SVM to use
#Need to get advice from group
import matplotlib.pyplot as plt
import seaborn as sns



#%%

'''Need to encode occupation since SVM takes numerical variables'''

final_df_encoded = pd.get_dummies(final_df, columns=['Occupation'], drop_first=True)
#First line returns us more variables (breaks variable down) with boolean values (True/False)
#Need to convert into integer
final_df_encoded[final_df_encoded.select_dtypes(include='bool').columns] = final_df_encoded.select_dtypes(include='bool').astype(int)

#Check if it worked
print(final_df_encoded.dtypes)
#Yes it worked as of running the line (12/10). Now move forward to model building

X_clean = X.dropna()
y_clean = y[X_clean.index]


#%%
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler


'''There were some missing values for X so I will remove them in the previous cell'''
X_clean = final_df_encoded.drop('Growing_Stress', axis=1).dropna()
y_clean = final_df_encoded['Growing_Stress'][X_clean.index]

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

scaler = StandardScaler()
Xtrain_scaled = scaler.fit_transform(Xtrain)
Xtest_scaled = scaler.transform(Xtest)

svm_model = SVC(kernel='linear', C=1.0, gamma='scale', probability=True)
svm_model.fit(Xtrain_scaled, Ytrain)

# Predictions
y_pred_class = svm_model.predict(Xtest_scaled)
y_pred_prob = svm_model.predict_proba(Xtest_scaled)[:, 1]

# %%[markdown]
# Is SVM model ideal for the dataset? SVM is more ideal for smaller datasets
# Should I take a percentage of the data and test again to see if computation time lowers?



'''New code to take random sample(10%) of the data'''
X_clean_sampled = X_clean.sample(frac=0.1, random_state=42)
y_clean_sampled = y_clean[X_clean_sampled.index]


from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_clean_sampled, y_clean_sampled, test_size=0.2, random_state=42)

# Scale the data (SVM is sensitive to feature scaling)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Xtrain_scaled = scaler.fit_transform(Xtrain)
Xtest_scaled = scaler.transform(Xtest)

from sklearn.svm import SVC
svm_model = SVC(kernel='linear', C=1.0, probability=True)
svm_model.fit(Xtrain_scaled, Ytrain)


# Predictions
y_pred_class = svm_model.predict(Xtest_scaled)
y_pred_prob = svm_model.predict_proba(Xtest_scaled)[:, 1]
# %%[Abirham]
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Using the processed dataset from the code
X_clean = encoded_final_df.drop('Growing_Stress', axis=1).dropna()
y_clean = encoded_final_df['Growing_Stress'][X_clean.index]

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

# Define the parameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# Instantiate the Random Forest model
rf = RandomForestClassifier(random_state=42)

# Run RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=50,
    scoring='accuracy',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Fit the model to the data
random_search.fit(X_train, y_train)

# Best parameters and best score
best_params = random_search.best_params_
best_score = random_search.best_score_

# Evaluate on the test set
best_model = random_search.best_estimator_
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

best_params, best_score, test_accuracy


# %%
