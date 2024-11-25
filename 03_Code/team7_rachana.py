#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #optional
import altair as alt
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline



#%%
health_data = pd.read_csv("Mental_Health_Dataset.csv")


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


import pandas as pd

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
                     'Coping_Struggles', 'Work_Interest', 'Social_Weakness']

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

# Create subplots for pie charts
fig, axs = plt.subplots(3, 4, figsize=(20, 15))
axs = axs.flatten()

# Generate pie charts
for i, (col, count) in enumerate(zip(cols_to_visualize, counts)):
    axs[i].pie(
        count,
        labels=count.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=color_palette[:len(count)]  # Use palette based on the number of unique values
    )
    axs[i].set_title(col, fontsize=14, fontweight='bold', color='darkslategray')
    axs[i].grid(False)

# Adjust layout
plt.tight_layout()
plt.show()



# %%


bright_palette = ["#FFB74D", "#64B5F6", "#81C784", "#E57373", "#FFD54F", "#4DD0E1"]

# Set the Seaborn style to 'whitegrid' for a clean look
sns.set(style="whitegrid")

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
Icount = health_data.groupby(['Growing_Stress', 'Days_Indoors']).size().reset_index(name='Count')

# Create the bar plot using Altair
import altair as alt

chart = alt.Chart(Icount).mark_bar().encode(
    x='Count:Q',
    y='Days_Indoors:N',
    color='Growing_Stress:N',
    tooltip=['Growing_Stress', 'Count']
).properties(
    title='Distribution of Growing Stress by Days Indoors',
    width=600,
    height=400
).configure_mark(
    opacity=0.7  # Subtle opacity for the bars
).configure_title(
    fontSize=18, 
    font='Arial', 
    anchor='middle', 
    color='gray'
)

# Show the chart
chart.show()
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
# different categories

#%%[markdown]
# We have a the "unknown factor" quantifies the portion of the variance in Growing_Stress that is unexplained by the selected features.
# (i) Calculate the weighted probability of all features, ensuring the total feature weights sum to 1.
# (ii) If the total weights are less than 1, it indicates growing stress due to personal reasons. 
# (iii) This suggests that an unknown factor is impacting the growing stress. And create a new feature unknown factor.
# 
#%%[markdown]
#caluclate the conditional probability
cols = health_data[['Gender', 'Occupation', 'self_employed', 'family_history', 'treatment', 'Days_Indoors',
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
final_df = df[['Gender','Occupation',  'family_history', 'treatment', 'Days_Indoors',
                     'Changes_Habits', 'Mental_Health_History', 'Mood_Swings', 'Coping_Struggles',
                     'Work_Interest', 'Social_Weakness', 'mental_health_interview', 'care_options', 'unknown_factor','Growing_Stress']]

final_df.head()

# %%
from sklearn.model_selection import train_test_split

data = final_df.drop('Growing_Stress',axis = 1)
f_data = final_df['Growing_Stress']
Xtrain,Xtest,Ytrain,Ytest = train_test_split(data,f_data,test_size = 0.2,random_state = 42)
Xtrain.head()
# %%

# Splitting the data into features and labels
data = final_df.drop('Growing_Stress', axis=1)
f_data = final_df['Growing_Stress']

# Identifying numeric and categorical columns
numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = data.select_dtypes(include=['object']).columns.tolist()

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numeric and categorical data
preprocessor = ColumnTransformer(
    transformers=[
#         ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', DecisionTreeClassifier(random_state=42))])

# Splitting the data into training and test sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, f_data, test_size=0.2, random_state=42)

# Fitting the model
clf.fit(Xtrain, Ytrain)

# Making predictions
predictions = clf.predict(Xtest)

# Evaluating the model
accuracy = accuracy_score(Ytest, predictions)
conf_matrix = confusion_matrix(Ytest, predictions)
class_report = classification_report(Ytest, predictions)

# Printing the evaluation results
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")
# %%
from sklearn.model_selection import train_test_split

data = final_df.drop('Growing_Stress',axis = 1)
f_data = final_df['Growing_Stress']
Xtrain,Xtest,Ytrain,Ytest = train_test_split(data,f_data,test_size = 0.2,random_state = 42)
Xtrain.head()
# %%

# Importing necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Splitting the data into features and labels
data = final_df.drop('Growing_Stress', axis=1)
f_data = final_df['Growing_Stress']

# Identifying numeric and categorical columns
numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = data.select_dtypes(include=['object']).columns.tolist()

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numeric and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the Random Forest model
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))])

# Splitting the data into training and test sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, f_data, test_size=0.2, random_state=42)

# Fitting the model
clf.fit(Xtrain, Ytrain)

# Making predictions
predictions = clf.predict(Xtest)

#%%
# Evaluating the model
accuracy = accuracy_score(Ytest, predictions)
conf_matrix = confusion_matrix(Ytest, predictions)
class_report = classification_report(Ytest, predictions)

# Printing the evaluation results
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")

# %%
# Importing necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Splitting the data into features and labels
data = final_df.drop('Growing_Stress', axis=1)
f_data = final_df['Growing_Stress']

# Identifying numeric and categorical columns
numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = data.select_dtypes(include=['object']).columns.tolist()

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numeric and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the Random Forest model
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))])

# Splitting the data into training and test sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, f_data, test_size=0.2, random_state=42)

# Fitting the model
clf.fit(Xtrain, Ytrain)

# Making predictions
predictions = clf.predict(Xtest)

# Evaluating the model
accuracy = accuracy_score(Ytest, predictions)
conf_matrix = confusion_matrix(Ytest, predictions)
class_report = classification_report(Ytest, predictions)

# Printing the evaluation results
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")

# %%
# Importing necessary libraries
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Splitting the data into features and labels
data = final_df.drop('Growing_Stress', axis=1)
f_data = final_df['Growing_Stress']

# Identifying numeric and categorical columns
numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = data.select_dtypes(include=['object']).columns.tolist()

# Preprocessing for categorical data with dense output
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Set sparse_output=False for dense output
])

# Bundle preprocessing for categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the HistGradientBoostingClassifier model
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', HistGradientBoostingClassifier(random_state=42))])

# Splitting the data into training and test sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, f_data, test_size=0.2, random_state=42)

# Fitting the model
clf.fit(Xtrain, Ytrain)

# Making predictions
predictions = clf.predict(Xtest)

# Evaluating the model
accuracy = accuracy_score(Ytest, predictions)
conf_matrix = confusion_matrix(Ytest, predictions)
class_report = classification_report(Ytest, predictions)

# Printing the evaluation results
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")

# %%
