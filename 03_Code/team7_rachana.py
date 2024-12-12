#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
from matplotlib import gridspec
import seaborn as sns 
import altair as alt
from scipy.stats import randint
import os
import plotly.graph_objects as go

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

#Correlation
from scipy.stats import spearmanr
from scipy.stats import chi2_contingency, chi2


# Validation libraries
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix

#models
from sklearn.tree import DecisionTreeClassifier as DTC, plot_tree
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

#%%[markdown]
# Cleaning Grwoing Stress column

# Assuming health_data is your DataFrame
# Step 1: Remove 'Maybe' values from Growing_Stress
health_data = health_data[health_data['Growing_Stress'] != 'Maybe']

print(health_data['Growing_Stress'].unique())

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


# %%
#Verifying the cleaned dataset
health_data.describe()
print(health_data.info())
# %%[markdown]
# Visualisation for Proportion of each feature in dataset

sns.set_style("whitegrid")

# Columns to visualize
cols_to_visualize = ['Gender', 'Occupation', 'self_employed', 'family_history', 'treatment', 
                     'Days_Indoors', 'Growing_Stress', 'Changes_Habits', 'Mental_Health_History', 
                     'Mood_Swings', 'Coping_Struggles', 'Work_Interest', 'Social_Weakness', 
                     'mental_health_interview', 'care_options']

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


# %% [markdown]
# Growing stress vs all variables now

# Define the color palette
color_palette = {
    0: 'rgba(144, 238, 144, 0.6)',  # Light Green for Growing Stress 0 with transparency
    1: 'rgba(255, 0, 0, 0.6)'        # Red for Growing Stress 1 with transparency
}

# Label mapping dictionary
label_mappings = {
    'Growing_Stress': {0: 'No', 1: 'Yes', 2: 'Maybe'},
    'Gender': {0: 'Male', 1: 'Female'},
    'self_employed': {0: 'No', 1: 'Yes'},
    'family_history': {0: 'No', 1: 'Yes'},
    'treatment': {0: 'No', 1: 'Yes'},
    'Days_Indoors' : {7.5: '1-14 days' , 22.5: '15-30 days', 45:'31-60 days', 60: 'More than 2 months', 365: 'Go out Every day'},
    'Changes_Habits': {0: 'No', 1: 'Yes', 2: 'Maybe'},
    'Mood_Swings': {0: 'Low', 1: 'Medium', 2: 'High'},
    'Mental_Health_History': {0: 'No', 1: 'Yes', 2: 'Maybe'},
    'Coping_Struggles': {0: 'No', 1: 'Yes'},
    'Work_Interest': {0: 'No', 1: 'Yes', 2: 'Maybe'},
    'Social_Weakness': {0: 'No', 1: 'Yes', 2: 'Maybe'},
    'mental_health_interview': {0: 'No', 1: 'Yes', 2: 'Maybe'},
    'care_options': {0: 'No', 1: 'Yes', 2: 'Not sure'}
}



# Initialize a list to hold figures
figures = []

# Loop through each feature in health_data
for feature in health_data.columns:
    if feature not in ['Growing_Stress', 'Country']:  # Skip the Growing_Stress column
        # Calculate counts of Growing Stress for each category
        counts = health_data.groupby([feature, 'Growing_Stress']).size().unstack(fill_value=0)
        
        # Calculate percentages
        percentages = counts.div(counts.sum(axis=1), axis=0) * 100
        
        # Create a stacked bar chart
        fig = go.Figure()
        
        # Add traces for each Growing Stress value
        for stress_value in percentages.columns:
            fig.add_trace(go.Bar(
                x=percentages.index.map(lambda x: label_mappings[feature].get(x, x) if feature in label_mappings else x),  # Apply label mapping only if feature is in label_mappings
                y=percentages[stress_value],
                name=f'Growing Stress: {("No" if stress_value == 0 else "Yes")}',  # Change legend labels
                marker_color=color_palette[stress_value],
                text=[f"{val:.1f}%" for val in percentages[stress_value]],  # Percentage text
                textposition='inside'  # Center the text inside the bars
            ))
        
        # Update layout for each figure
        fig.update_layout(
            title=f'Growing Stress Distribution by {feature}',
            xaxis_title=feature,
            yaxis_title='Percentage',
            barmode='stack',
            legend_title='Growing Stress',
            template='plotly_white',
            height=400,  # Adjust height for better visualization
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        # Add grid lines for better readability
        fig.update_xaxes(showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridcolor='lightgray')
        
        # Append the figure to the list
        figures.append(fig)

# Show all figures
for fig in figures:
    fig.show()


#%%[markdown] - Haeyeon
# Insights from ditributions

# Growing Stress vs Timestamp
#The distribution differences across the years 2014, 2015, and 2016 are not significant. In all three years, the proportion of respondents answering "Yes" is slightly higher.

# Growing Stress vs Gender:
# Within the same gender, for female, the proportion of those who answered "Yes" to growing stress is higher than that of men.

# Growing Stress vs Occupation:
# Among occupations, the highest proportion of "Yes" responses regarding growing stress is seen in the business, followed by the student occupation.

# Growing Stress vs Self-employed

# Growing Stress vs Family history

# Growing stress vs Treatment

# Growing Stress vs Days_Indoors

# Growing Stress vs Changes Habits

# Growing Stress vs Mental Health History

# Growing Stress vs Mood Swings

# Growing Stress vs Coping Struggles

# Growing Stress vs Work Interest

# Growing Stress vs Social Weakness

# Growing Stress vs Mental Health Interview

# Growing Stress vs Care options

#$$[markdown]
# EDA for Treatment

# Define a new color palette for treatment (0 for 'No' and 1 for 'Yes')
treatment_color_palette = {
    0: 'rgba(144, 238, 144, 0.6)',  # Light Green for 'No' treatment
    1: 'rgba(255, 0, 0, 0.6)'        # Red for 'Yes' treatment
}

# Label mapping dictionary for various columns
new_label_mappings = {
    'Growing_Stress': {0: 'No', 1: 'Yes'},
    'Gender': {0: 'Male', 1: 'Female'},
    'self_employed': {0: 'No', 1: 'Yes'},
    'family_history': {0: 'No', 1: 'Yes'},
    'treatment': {0: 'No', 1: 'Yes'},
    'Days_Indoors': {7.5: '1-14 days', 22.5: '15-30 days', 45: '31-60 days', 60: 'More than 2 months', 365: 'Go out Every day'},
    'Changes_Habits': {0: 'No', 1: 'Yes', 2: 'Maybe'},
    'Mood_Swings': {0: 'Low', 1: 'Medium', 2: 'High'},
    'Mental_Health_History': {0: 'No', 1: 'Yes', 2: 'Maybe'},
    'Coping_Struggles': {0: 'No', 1: 'Yes'},
    'Work_Interest': {0: 'No', 1: 'Yes', 2: 'Maybe'},
    'Social_Weakness': {0: 'No', 1: 'Yes', 2: 'Maybe'},
    'mental_health_interview': {0: 'No', 1: 'Yes', 2: 'Maybe'},
    'care_options': {0: 'No', 1: 'Yes', 2: 'Not sure'}
}

# Initialize a list to hold the figures for treatment-related plots
treatment_relation_figures = []

# Loop through each feature in health_data (focus on columns that are not 'treatment' and not 'Timestamp')
for feature in health_data.columns:
    if feature not in ['treatment', 'Timestamp']:  # Skip 'treatment' and 'Timestamp' columns
        # Calculate the count of each category within the feature and grouped by treatment status
        feature_counts = health_data.groupby([feature, 'treatment']).size().unstack(fill_value=0)
        
        # Normalize counts to percentages
        feature_percentages = feature_counts.div(feature_counts.sum(axis=1), axis=0) * 100
        
        # Create a new figure for the current feature
        feature_fig = go.Figure()

        # Add traces for each treatment status (0: No, 1: Yes)
        for treatment_status in feature_percentages.columns:
            feature_fig.add_trace(go.Bar(
                # Apply label mapping for the feature if it exists in label_mappings
                x=feature_percentages.index.map(
                    lambda x: label_mappings[feature].get(x, x) if feature in label_mappings else x
                ),  # Use label mapping only for features with mappings
                y=feature_percentages[treatment_status],  # Use the percentage for each treatment status
                name=f'Treatment: {label_mappings["treatment"].get(treatment_status, treatment_status)}',  # Name the trace based on treatment status
                marker_color=treatment_color_palette[treatment_status],  # Color based on treatment status
                text=[f"{val:.1f}%" for val in feature_percentages[treatment_status]],  # Percentage text
                textposition='inside'  # Text inside the bars
            ))

        # Update layout for the figure
        feature_fig.update_layout(
            title=f'Treatment Status by {feature}',
            xaxis_title=feature,
            yaxis_title='Percentage',
            barmode='stack',
            legend_title='Treatment Status',
            template='plotly_white',
            height=400,  # Adjust height for better visualization
            margin=dict(l=40, r=40, t=40, b=40)
        )

        # Add grid lines for readability
        feature_fig.update_xaxes(showgrid=True, gridcolor='lightgray')
        feature_fig.update_yaxes(showgrid=True, gridcolor='lightgray')

        # Append the figure to the list of treatment-related figures
        treatment_relation_figures.append(feature_fig)

# Show all treatment-related figures
for fig in treatment_relation_figures:
    fig.show()

#%%[markdown]
# Statistical Testing for Treatment

# List of columns to consider for the chi-square test
cols = ['Timestamp', 'Gender', 'Country', 'Occupation', 'self_employed', 'family_history', 
        'treatment', 'Days_Indoors', 'Changes_Habits', 'Mental_Health_History', 'Mood_Swings',
        'Coping_Struggles', 'Work_Interest', 'Social_Weakness', 'mental_health_interview', 'care_options']

# Function to calculate Chi-Square Test of Independence
def calculate_chi_square(column1, column2='treatment'):
    print(f"Correlation between **{column1}** and **{column2}**\n")
    # Generate the crosstab for the two columns
    crosstab = pd.crosstab(health_data[column1], health_data[column2])
    
    # Perform the Chi-Square test
    stat, p, dof, expected = chi2_contingency(crosstab, correction=True)
    
    print(f'P-value = {p}, Degrees of freedom = {dof}')
    
    # Critical value for 95% confidence
    prob = 0.95
    critical = chi2.ppf(prob, dof)
    print(f'Probability = %.3f, Critical value = %.3f, Test statistic = %.3f' % (prob, critical, stat))
    
    # Hypothesis test decision
    if stat >= critical:
        print('Dependent (reject H0)')
    else:
        print('Independent (accept H0)')
    
    # Significance testing
    alpha = 1.0 - prob
    print(f'Significance level = %.3f, p-value = %.3f' % (alpha, p))
    
    if p <= alpha:
        print('Dependent (reject H0)')
    else:
        print('Independent (accept H0)')
    
    print('\n-----------------------------------\n')


# Run Chi-Square tests for the correlation between the `treatment` and each feature
print('** Chi-Square Correlation between Treatment and Other Features **\n')
for col in cols:
    # Skip the 'treatment' column itself as it will always be correlated with 'treatment'
    if col != 'treatment':
        calculate_chi_square(col)
        

#%%[markdown]
# Copy dataset for trying different Smart question
health_data_backup = health_data.copy()

# One - Hot Encoding the Occupation column
final_df = health_data.drop(columns=['Country'])
final_df['Occupation'] = final_df['Occupation'].astype(str)

# One-Hot Encoding using Pandas
categorical_columns = final_df.select_dtypes(include=['object']).columns.tolist()
occupation_encoded = pd.get_dummies(final_df['Occupation'], prefix='Occupation', drop_first=True)
encoded_final_df = pd.concat([final_df.drop('Occupation', axis=1), occupation_encoded], axis=1)
encoded_final_df.head()




# %%[markdown]
# Correlation Matrix

# Calculate Spearman correlation matrix
correlation_matrix, _ = spearmanr(encoded_final_df, axis=0)

# Convert the result to a DataFrame for better visualization
correlation_df = pd.DataFrame(correlation_matrix, index=encoded_final_df.columns, columns=encoded_final_df.columns)

# Set up the matplotlib figure
plt.figure(figsize=(14, 10))

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(correlation_df, dtype=bool))

# Create a heatmap with the mask
sns.heatmap(correlation_df, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, 
            square=True, mask=mask, linewidths=.5)

# Set titles and labels
plt.title('Spearman Correlation Matrix of Encoded Final DataFrame (Lower Half)', fontsize=16)
plt.xticks(rotation=90)
plt.yticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()



# %%[markdown]
# It seems that the treatment and Family History have high correlation,
# Except those, all the other variables have do not seem to have a linear relationship with Growing Stress.
# Only Mental Health History has a slight correlation with Growing Stress

# %%[markdown]
# Statistical tests

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


#%%[markdown]
# As the Timestamp, Country, self-employed are independednt of Growing Stress and also have a highly unbalanced data
# it is better to remove those columns for our Modelling.

#%%[markdown]
# Modelling

encoded_final_df = encoded_final_df.drop(columns=['Timestamp', 'self_employed'])
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


# %%[markdown]
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




# %%[markdown]
# %%[markdown]

# Desciion Tree

target = encoded_final_df['Growing_Stress']

# Drop the target variable from the features
features = encoded_final_df.drop(['Growing_Stress'], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

#%%

# Base model with default parameters
base_params = {
    'max_depth': 10,         # No limit by default
    'min_samples_split': 10,    # Default value
    'min_samples_leaf': 20,      # Default value
    'max_features': 'sqrt',      # Consider all features
}

# Define K-Fold cross-validator
kfold = KFold(n_splits=10, shuffle=True, random_state=0)

#%%[markdown]

# Function to vary the parameters
def tune_and_plot(param_name, param_values, base_params, X_train, y_train):
    train_accuracies = []
    test_accuracies = []
    
    for value in param_values:
        # Update the parameter in the base configuration
        params = base_params.copy()
        params[param_name] = value
        
        # Create a DecisionTreeClassifier with the current parameters
        model = DTC(**params, random_state=0)
        
        # Perform K-Fold Cross-Validation
        train_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy', n_jobs=-1)
        train_accuracies.append(np.mean(train_scores))
        
        # Fit on train data and test on the same folds (to get "test-like" score)
        model.fit(X_train, y_train)
        test_accuracies.append(model.score(X_train, y_train))  # Simulated test set accuracy
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(param_values, test_accuracies, label='Test Accuracy', marker='o')
    plt.title(f'Effect of {param_name} on Accuracy', fontsize=16)
    plt.xlabel(param_name, fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()

#%%
# Tune max_depth
tune_and_plot('max_depth', [None, 5, 6, 7, 8, 9, 10], base_params, X_train, y_train)

# At 9 test accuarcy seems to be the highest hence better to set max_depth as 9
# If the tree becomes complex, we can decrease it later to 6 or 7 as test accuracy is still high for them
#%%
# Tune min_samples_split
tune_and_plot('min_samples_split', [2, 5, 10, 15, 20], base_params, X_train, y_train)
#%%
# Tune min_samples_leaf
tune_and_plot('min_samples_leaf', [1, 2, 5, 10, 20], base_params, X_train, y_train)
#%%
# Tune max_features
tune_and_plot('max_features', ['sqrt', 'log2'], base_params, X_train, y_train)
#%%
# Now let's check the final accuracy with these parameters using k-fold cross validation
# Define the final model with optimal parameters
final_params = {
    'max_depth': 9,           # Optimal depth found
    'min_samples_split': 10,  # Higher value to prevent overfitting
    'min_samples_leaf': 10,    # Based on accuracy analysis
    'max_features': 'sqrt',    # Good practice
}

# Initialize the final Decision Tree model
final_model = DTC(**final_params, random_state=0)

# Set up K-Fold cross-validation
kfold = KFold(n_splits=10, shuffle=True, random_state=0)  # 10 folds

# Perform cross-validation and get scores for each fold
cv_scores = cross_val_score(final_model, X_train, y_train, cv=kfold, scoring='accuracy')

# Calculate mean and standard deviation of the cross-validation scores
mean_cv_score = np.mean(cv_scores)
std_cv_score = np.std(cv_scores)

# Fit the final model on the entire training data
final_model.fit(X_train, y_train)

# Evaluate the final model on training and test data
train_accuracy = final_model.score(X_train, y_train)
test_accuracy = final_model.score(X_test, y_test)

# Print results
print("Cross-Validation Accuracies for Each Fold:", cv_scores)
print("Mean Cross-Validation Accuracy:", mean_cv_score)
print("Standard Deviation of Cross-Validation Accuracy:", std_cv_score)
print("Final Training Accuracy:", train_accuracy)
print("Final Test Accuracy:", test_accuracy)

#%%
# For depth = 9 it seems that the model has slightly greater test accuracy than train accuracy but the 
# cross validation accuracy is high hence, this seems to be a good fit

#%%
# Confusion matrix and AUC, ROC
y_pred = final_model.predict(X_test)
y_pred_proba = final_model.predict_proba(X_test)[:, 1]  # Probability of the positive class

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(confusion_matrix=cm)
cmd.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Print AUC score
print("AUC Score:", roc_auc)

# As observed we got an AUC of 0.93 which reflects that most of the classification has been done 
# accurately.






# %%[markdown]

# Random Forest

# Perform GridSearchCV to find the optimal parameters

# Base model with default parameters
base_params = {
    'n_estimators': 10,
    'max_depth': 7,
    'min_samples_split': 20,
    'min_samples_leaf': 20,
    'max_features': 'log2',
    'bootstrap': True
}

# Define K-Fold cross-validator
kfold = KFold(n_splits=10, shuffle=True, random_state=0)


#%%[markdown]
# Function to vary the parameters

def tune_and_plot(param_name, param_values, base_params, X_train, y_train):
    train_accuracies = []
    test_accuracies = []
    
    for value in param_values:
        # Update the parameter in the base configuration
        params = base_params.copy()
        params[param_name] = value
        
        # Create a RandomForestClassifier with the current parameters
        model = RandomForestClassifier(**params, random_state=0)
        
        # Perform K-Fold Cross-Validation
        train_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy', n_jobs=-1)
        train_accuracies.append(np.mean(train_scores))
        
        # Fit on train data and test on the same folds (to get "test-like" score)
        model.fit(X_train, y_train)
        test_accuracies.append(model.score(X_train, y_train))  # Simulated test set accuracy
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(param_values, test_accuracies, label='Test Accuracy', marker='o')
    plt.title(f'Effect of {param_name} on Accuracy', fontsize=16)
    plt.xlabel(param_name, fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()
#%%[markdown]

tune_and_plot('n_estimators', [10, 20, 30, 50], base_params, X_train, y_train)

# After estimators of 10 it seems that the train error rate increases compared to test error rate
# hence best to set n_estimators as 10
#%%

tune_and_plot('max_depth', [5, 6, 7, 8,9, 10], base_params, X_train, y_train)
# It seems that at max_depth = 7 the test accuracy is highest comapred to train hence 
# better to set the max_depth to 7

#%%

tune_and_plot('min_samples_split', [5, 10, 15, 20], base_params, X_train, y_train)

# It has same accuracy for all values hence better to set min_samples_split to 20
# as it will reduce the complexity of the model
#%%
tune_and_plot('min_samples_leaf', [1, 5, 10, 20], base_params, X_train, y_train)

# According to the graph it seems that min_samples_leaf has highest accuarcy hence better to set
# min_samples_leaf to 20
#%%

tune_and_plot('max_features', ['sqrt', 'log2'], base_params, X_train, y_train)

#%%
tune_and_plot('bootstrap', [True, False], base_params, X_train, y_train)

# According to the graph it is better to choose booststrap = TRUE
#%%
# Now lets check the final accuracy with these parameters using k-fold cross validation


from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier

# Define the final model with optimal parameters
final_params = {
    'n_estimators': 10,  # Optimal number of estimators
    'max_depth': 7,      # Chosen based on previous tuning
    'min_samples_split': 20,  # Higher value to prevent overfitting
    'min_samples_leaf': 20,    # Based on accuracy analysis
    'max_features': 'sqrt',     # Good practice
    'bootstrap': True            # Default; can test with False if needed
}

# Initialize the final Random Forest model
final_model = RandomForestClassifier(**final_params, random_state=0)

# Set up K-Fold cross-validation
kfold = KFold(n_splits=10, shuffle=True, random_state=0)  # 10 folds

# Perform cross-validation and get scores for each fold
cv_scores = cross_val_score(final_model, features, target, cv=kfold, scoring='accuracy')

# Calculate mean and standard deviation of the cross-validation scores
mean_cv_score = np.mean(cv_scores)
std_cv_score = np.std(cv_scores)

# Fit the final model on the entire training data
final_model.fit(X_train, y_train)

# Evaluate the final model on training and test data
train_accuracy = final_model.score(X_train, y_train)
test_accuracy = final_model.score(X_test, y_test)

# Print results
print("Cross-Validation Accuracies for Each Fold:", cv_scores)
print("Mean Cross-Validation Accuracy:", mean_cv_score)
print("Standard Deviation of Cross-Validation Accuracy:", std_cv_score)
print("Final Training Accuracy:", train_accuracy)
print("Final Test Accuracy:", test_accuracy)

#%%
# As seen the training testing accuracies along with highest cross validation accuracies seems to be 
# almost same, hence we can choose this fit, if we want to reduce complexity and compromise accuracy it is 
# good to choose depth 6 as well

#%%
# Confusion Matrix and ROC AUC

# Make predictions on the test set from final model fit

y_pred = final_model.predict(X_test)
y_pred_proba = final_model.predict_proba(X_test)[:, 1]  # Probability of the positive class

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(confusion_matrix=cm)
cmd.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Print AUC score
print("AUC Score:", roc_auc)

# As observed we got an AUC of 0.97 which reflects that most of the classification has been done 
# accurately.



























#%% KNN on Whole data set

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Assuming 'Growing_Stress' is our target variable
X = encoded_final_df.drop('Growing_Stress', axis=1)
y = encoded_final_df['Growing_Stress']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the KNN model
knn = KNeighborsClassifier(n_neighbors=20)  # You can adjust the number of neighbors
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# %%
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_knn = grid_search.best_estimator_

# %%
from sklearn.model_selection import cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

cv_scores = cross_val_score(best_knn, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f}")


# %%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Initialize and train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)
y_pred_proba = knn.predict_proba(X_test)[:, 1]

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Cross-validation score
cv_scores = cross_val_score(knn, X, y, cv=5)

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f}")

print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# %%  Abirham
# Do people with Mental health history receive treatment or not?
# how well "Mental Health History" predicts whether a person will seek treatment

# %%
import pandas as pd
from scipy.stats import chi2_contingency

# Let's first examine the unique values for 'Mental_Health_History' and 'Treatment'
print(encoded_final_df['Mental_Health_History'].unique())
print(encoded_final_df['treatment'].unique())

# Performing a Chi-Square Test for Independence to check if there's a relationship between the two variables
crosstab = pd.crosstab(encoded_final_df['Mental_Health_History'], encoded_final_df['treatment'])
stat, p, dof, expected = chi2_contingency(crosstab)

print(f"Chi-Square Test: p-value = {p}")
if p < 0.05:
    print("There is a significant association between Mental Health History and Treatment.")
else:
    print("There is no significant association between Mental Health History and Treatment.")

# %%
# the small p  value indicates that there is a strong association between mental_health_history and treatment
# %%
# visualizations to better understand the relationship between "Mental Health History" and "Treatment".
import pandas as pd
import matplotlib.pyplot as plt
import mplcursors

# Group data for the plot
grouped_data = encoded_final_df.groupby(['Mental_Health_History', 'treatment']).size().unstack()

# Create a stacked bar chart
ax = grouped_data.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='Set2')
plt.title('Mental Health History vs Treatment')
plt.xlabel('Mental Health History')
plt.ylabel('Count')
plt.legend(title='Treatment')

# Add interactivity
mplcursors.cursor(ax, hover=True)

# Show the plot
plt.tight_layout()
plt.show()


# %% modelling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Features and target
X = encoded_final_df.drop(['treatment','Occupation_Student','Occupation_Housewife','Occupation_Others','Occupation_Corporate'],axis=1)  # Using only 'Mental_Health_History' for prediction
y = encoded_final_df['treatment']

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Predict on the test set
y_pred = logreg.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Logistic Regression Model: {accuracy:.2f}")

# %% Feature importance
# Getting the coefficient for 'Mental_Health_History' from the logistic regression model
coeff = logreg.coef_[0][0]
print(f"Coefficient for Mental Health History: {coeff:.4f}")
if coeff > 0:
    print("A positive coefficient means that having a mental health history increases the likelihood of seeking treatment.")
else:
    print("A negative coefficient means that having a mental health history decreases the likelihood of seeking treatment.")

#%%
# Extract feature importance
feature_importance = logreg.coef_[0]  # Coefficients for each feature
feature_names = X.columns  # Names of the features

# Combine them into a DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values(by='Importance', key=abs, ascending=False)  # Sort by absolute importance

# Display the feature importance
print("Feature Importance:")
print(importance_df)

# Optional: Visualize the feature importance
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Coefficient Value')
plt.ylabel('Features')
plt.title('Feature Importance for Logistic Regression')
plt.gca().invert_yaxis()  # Reverse the order for better readability
plt.show()

# %%
# we got a positive coeficient and  it means having a mental health history  increases the likely hood of getting treatment
# %%
from sklearn.metrics import roc_auc_score, roc_curve

# AUC-ROC
roc_auc = roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])
print(f"AUC-ROC: {roc_auc:.2f}")

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Initialize and train the KNN model
knn = KNeighborsClassifier(n_neighbors=10)  # Start with 5 neighbors
knn.fit(X_train, y_train)

# Predict on the test set
y_pred_knn = knn.predict(X_test)
y_pred_proba_knn = knn.predict_proba(X_test)[:, 1]

# Evaluate KNN
print("KNN Classification Report:")
print(classification_report(y_test, y_pred_knn))

print("KNN Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))

# AUC-ROC
roc_auc_knn = roc_auc_score(y_test, y_pred_proba_knn)
print(f"KNN AUC-ROC: {roc_auc_knn:.2f}")


#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt

# Initialize and train the Random Forest model
rf = RandomForestClassifier(n_estimators= 20, max_depth= 5, random_state=42)  # You can tune hyperparameters
rf.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf.predict(X_test)
y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]

# Evaluate the model
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# AUC-ROC
roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
print(f"Random Forest AUC-ROC: {roc_auc_rf:.2f}")

# Feature Importance
feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("Feature Importances:")
print(feature_importances)

# Visualize Feature Importances
plt.figure(figsize=(10, 6))
plt.bar(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#From the above models it could be seen that the treatment with different models has the highest coefficent for
#mental health history




# After this need to maybe REMOVE CODE !!!!










# %%
from scipy.stats import chi2_contingency

# Contingency table
contingency_table = pd.crosstab(encoded_final_df['family_history'], encoded_final_df['treatment'])

# Perform Chi-Square Test
chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-Square Test Results:")
print(f"Chi-Square Statistic: {chi2_stat:.2f}")
print(f"P-value: {p_value:.4f}")
if p_value < 0.05:
    print("Family History is significantly associated with Treatment.")
else:
    print("Family History is NOT significantly associated with Treatment.")


# %%
# EDA between Family History and Treatment
plt.figure(figsize=(8, 6))
sns.countplot(x='family_history', hue='treatment', data=encoded_final_df, palette='Set2')
plt.title('Family History vs Treatment')
plt.xlabel('Family History')
plt.ylabel('Count')
plt.legend(title='Treatment', loc='upper right')
plt.show()

# EDA between Mental Health History, Family History, and Treatment
plt.figure(figsize=(10, 8))
sns.heatmap(encoded_final_df.groupby(['Mental_Health_History', 'family_history'])['treatment']
            .mean().unstack(), annot=True, fmt='.2f', cmap='coolwarm', cbar_kws={'label': 'Treatment Rate'})
plt.title('Mental Health History and Family History vs Treatment Rate')
plt.xlabel('Family History')
plt.ylabel('Mental Health History')
plt.show()

# %%[markdown]
# Without Family History (family_history=0):
# A larger proportion of people do not seek treatment (green bar) compared to those who do (orange bar).
# This suggests that individuals without a family history of mental health issues are less likely to seek treatment overall.
# With Family History (family_history=1):
# A significantly larger proportion of people with a family history of mental health issues seek treatment (orange bar) compared to those who do not (green bar).
# This suggests that having a family history of mental health issues is positively associated with seeking treatment.
# %%  KNN model for family Histor  mental Health History and treatment

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Define features and target
X = encoded_final_df[['Mental_Health_History', 'family_history']]  # Two variables
y = encoded_final_df['treatment']  # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predictions
y_pred = knn.predict(X_test)
y_pred_proba = knn.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba)

print("Performance Metrics for KNN:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"AUC-ROC: {auc_roc:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Confusion Matrix Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Treatment', 'Treatment'], 
            yticklabels=['No Treatment', 'Treatment'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_roc:.2f})', color='darkorange', linewidth=2)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--', linewidth=1.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(alpha=0.4)
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Bar Chart of Performance Metrics
metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1, 'AUC-ROC': auc_roc}
plt.figure(figsize=(10, 6))
plt.bar(metrics.keys(), metrics.values(), color=['skyblue', 'orange', 'green', 'purple', 'red'])
plt.ylim(0, 1)
plt.title('Performance Metrics')
plt.ylabel('Score')
plt.xlabel('Metric')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



# %%



# %% 

# %%
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Perform Grid Search
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='roc_auc')
grid_knn.fit(X_train, y_train)

# Best model
best_knn = grid_knn.best_estimator_
print(f"Best parameters: {grid_knn.best_params_}")

# Evaluate best KNN
y_pred_best = best_knn.predict(X_test)
y_pred_proba_best = best_knn.predict_proba(X_test)[:, 1]
print(f"Optimized KNN AUC-ROC: {roc_auc_score(y_test, y_pred_proba_best):.2f}")


# %%









































#%%[markdown]
# EDA specifically for the Student occupation

# Data preparation
# Filter data for 'Student' occupation
health_data_student = health_data_backup[health_data_backup['Occupation'] == 'Student']

# Drop unnecessary columns same as previous
health_data_student = health_data_student.drop(columns=['Country'])

# Check the first few rows of the filtered data
health_data_student.head()

# Check student data info
health_data_student.info()


#%%[markdown]
# EDA for student start here

# Initialize a list to hold figures_st
figures_st = []

# Loop through each feature in student_data
for feature in health_data_student.columns:
    if feature not in ['Growing_Stress', 'Occupation']:  # Skip the Growing_Stress, Occupation column
        # Calculate counts of Growing Stress for each category
        counts = health_data_student.groupby([feature, 'Growing_Stress']).size().unstack(fill_value=0)
        
        # Calculate percentages
        percentages = counts.div(counts.sum(axis=1), axis=0) * 100
        
        # Create a stacked bar chart
        fig = go.Figure()
        
        # Add traces for each Growing Stress value
        for stress_value in percentages.columns:
            fig.add_trace(go.Bar(
                x=percentages.index.map(lambda x: label_mappings[feature].get(x, x) if feature in label_mappings else x),  # Apply label mapping only if feature is in label_mappings
                y=percentages[stress_value],
                name=f'Growing Stress: {("No" if stress_value == 0 else "Yes")}',  # Change legend labels
                marker_color=color_palette[stress_value],
                text=[f"{val:.1f}%" for val in percentages[stress_value]],  # Percentage text
                textposition='inside'  # Center the text inside the bars
            ))
        
        # Update layout for each figure
        fig.update_layout(
            title=f'Growing Stress Distribution by {feature}',
            xaxis_title=feature,
            yaxis_title='Percentage',
            barmode='stack',
            legend_title='Growing Stress',
            template='plotly_white',
            height=400,  # Adjust height for better visualization
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        # Add grid lines for better readability
        fig.update_xaxes(showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridcolor='lightgray')
        
        # Append the figure to the list
        figures_st.append(fig)

# Show all figures
for fig in figures_st:
    fig.show()

#%%[markdown]
# Insights from ditributions
#
# Growing Stress vs Timestamp: 
# The distribution differences across the years 2014, 2015, and 2016 are not significant. In all three years, the proportion of respondents answering "Yes" is higher.
#
# Growing Stress vs Gender:
# Within the same gender, for male, the proportion of those who answered "Yes" to growing stress is higher than that of female.
#
# Growing Stress vs Self-employed/Family history/treatment :
# The proportion looks similar between No and Yes groups.

# Growing Stress vs Days_Indoors: 
# Students who spend more than 30 days away from home are more likely to experience growing stress.

# Growing Stress vs Changes Habits

# Growing Stress vs Mental Health History

# Growing Stress vs Mood Swings

# Growing Stress vs Coping Struggles

# Growing Stress vs Work Interest

# Growing Stress vs Social Weakness

# Growing Stress vs Mental Health Interview

# Growing Stress vs Care options

#%%[markdown]
# Correlation heatmap for variables correlated with Growing Stress for students
student_corr = health_data_student[['Days_Indoors', 'Growing_Stress', 'Changes_Habits', 'Coping_Struggles', 
                             'Mood_Swings', 'Social_Weakness', 'treatment']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(student_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap for Growing Stress and Related Variables for Students')
plt.show()

#%%[markdown]
# Statistical tests for student
from scipy.stats import chi2_contingency, chi2

cols= ['Timestamp','Gender','self_employed','family_history', 'treatment', 'Days_Indoors',
                    'Changes_Habits', 'Mental_Health_History', 'Mood_Swings','Coping_Struggles', 'Work_Interest', 'Social_Weakness','mental_health_interview','care_options']
def calculate_chi_square(column1,column2 = 'Growing_Stress'):
    print(f"Correlation between **{column1}** and **{column2}**\n")
    crosstab = pd.crosstab(health_data_student[column1],health_data_student[column2])
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

#%%[markdown]
# Modeling For Student - yonathan


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix


#Make a copy of the data
health_data_yon = health_data.copy()


#Break student apart from the other occupations
health_data_student_yon = health_data_yon[health_data_yon['Occupation'] == 'Student'].drop(['Timestamp', 'Country', 'Occupation',], axis = 1)
health_data_student_yon = health_data_student_yon[health_data_student_yon['Growing_Stress'] != 2]
health_data_student_yon.head()
#With the value 2 taken out for growing stress and the data filtered for only students, we have 37772 observations.


#Features and target
X = health_data_student_yon.drop('Growing_Stress', axis = 1)
y = health_data_student_yon ['Growing_Stress']


#Training and Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


#Scaling work
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)


# Predictions
y_pred = log_reg.predict(X_test_scaled)
y_pred_prob = log_reg.predict_proba(X_test_scaled)[:, 1]


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)


print(f"Accuracy: {accuracy}")
print(f"ROC-AUC: {auc:.2f}")


# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


from sklearn.model_selection import cross_val_score
cross_val_score(log_reg, X_train_scaled, y_train, cv=5)
#Initial AUC of 0.62 and accuracy of 0.61. We aren't satisfied so we have to try to do feature engineering.


# Evaluate feature importance
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': log_reg.coef_[0]
}).sort_values(by='Coefficient', ascending=False)


print(coefficients)
#Given the model we just built, the strongest coefficients seem to be Days_Indoors, Mood_Swings, and social_weakness,
#Let's try to remove some of the unimportant variables to test if model accuracy will improve




# %% What is this for? ####################################
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
health_data_student = health_data_copy[health_data_copy['Occupation'] == 'Student']

# Preprocessing the dataset
# Drop 'Timestamp' and other irrelevant columns for prediction
X = health_data_student.drop(['Growing_Stress', 'Timestamp', 'Country', 'Occupation'], axis=1)  # Drop target and irrelevant columns
y = health_data_student['Growing_Stress']  # Target column

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

#%%
# Calculate the combined new feature as a weighted sum of the conditional probabilities for both 'Yes' and 'No'
df['New_Feature'] = sum(df[f'{feature}_prob_1'] * weights[1][feature] + df[f'{feature}_prob_0'] * weights[0][feature] for feature in selected_features)
df['unknown_factor'] = 1 - df['New_Feature']
pd.set_option('display.max_columns', None)
df.head()
#%%
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



# %%[markdown] - how to explain unknown factor by utilizing some visualization ---- start from haeyeon

# 1. Distribution of Conditional Probabilities per Feature
# Visualize how the conditional probability of each feature varies across the different values of Growing_Stress
# Display the distribution of the conditional probabilities for each feature, so you can see how the features behave for each category of Growing_Stress (e.g., Yes, No).

def plot_conditional_probability(feature, conditional_probs):
    """
    Plot the distribution of conditional probabilities for a specific feature across Growing_Stress classes.
    """
    plt.figure(figsize=(10, 6))
    for target in conditional_probs[feature]:
        values = list(conditional_probs[feature][target].values())
        sns.histplot(values, kde=True, label=f'Growing_Stress {target}', color='blue' if target == 1 else 'orange')
    
    plt.title(f"Conditional Probability Distribution for {feature} by Growing_Stress")
    plt.xlabel('Conditional Probability')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Example: Visualizing conditional probabilities for the 'Occupation' feature
plot_conditional_probability('Occupation', conditional_probs)


# %%[markdown]
# 2. Feature Weights Visualization
# how each feature contributes to explaining the variance in Growing_Stress by showing the weights for each feature.
# Show the weights calculated for each feature, indicating how much each feature contributes to the overall model

def plot_feature_weights(weights):
    """
    Plot feature weights, which represent how much each feature contributes to the model.
    """
    # Flatten the feature weights for plotting
    feature_names = list(weights[1].keys())
    feature_weights = [weights[1][feature] + weights[0][feature] for feature in feature_names]  # Sum of weights for both Yes and No

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.barplot(x=feature_names, y=feature_weights, palette='viridis')
    plt.title('Feature Weights for Growing_Stress')
    plt.xlabel('Feature')
    plt.ylabel('Weight')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# Visualize feature weights for Growing_Stress
plot_feature_weights(weights)
print(final_df.columns)
# %%[markdown]
# 3. Unknown Factor Distribution
# visualize its distribution to see how much it contributes to each data point.
# Display the distribution of the unknown factor and compare it with the classification of Growing_Stress to see how the factor influences the classification

def plot_unknown_factor_distribution(df):
    """
    Plot the distribution of the 'unknown_factor' and how it relates to 'Growing_Stress' classifications.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df['unknown_factor'], kde=True, color='green')
    
    plt.title('Distribution of Unknown Factor')
    plt.xlabel('Unknown Factor')
    plt.ylabel('Frequency')
    
    # Show how unknown_factor is related to Growing_Stress
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Growing_Stress', y='unknown_factor', data=df, palette='Set2')
    plt.title('Unknown Factor by Growing_Stress Class')
    plt.xlabel('Growing_Stress')
    plt.ylabel('Unknown Factor')
    plt.show()

# Visualize the unknown factor distribution and how it relates to Growing_Stress
plot_unknown_factor_distribution(final_df)

# %%[markdown]
# 4. Impact of Threshold on Classification
# visualize how the unknown_factor affects the binary classification of Growing_Stress
# plot a histogram of the unknown_factor values and use a threshold to split the data into 0 (No Stress) and 1 (Stress)
# Display how the threshold (set to the median) splits the data into two categories, allowing you to visualize how the unknown_factor affects classification.

def plot_threshold_impact(df, threshold):
    """
    Plot the impact of the threshold on Growing_Stress classification.
    """
    plt.figure(figsize=(10, 6))

    # Plot the histogram of unknown_factor
    sns.histplot(df['unknown_factor'], kde=True, color='purple', label='Unknown Factor', bins=30)

    # Add threshold line
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.2f}')
    
    # Display
    plt.title('Impact of Threshold on Growing_Stress Classification')
    plt.xlabel('Unknown Factor')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Plot the impact of the threshold on the classification of Growing_Stress
plot_threshold_impact(final_df, threshold)

#%% [markdown]
# 5. Final Classification Overview
# After the transformation of Growing_Stress based on the unknown factor, we can visualize the final class distribution of Growing_Stress (either 0 or 1), comparing it to the original categories
# Show the final distribution of Growing_Stress after applying the threshold to classify it into two categories (1 for stress, 0 for no stress)

def plot_final_classification(final_df):
    """
    Visualize the final classification of Growing_Stress (0 or 1).
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Growing_Stress', data=final_df, palette='pastel')
    plt.title('Final Classification of Growing_Stress')
    plt.xlabel('Growing_Stress')
    plt.ylabel('Count')
    plt.show()

# Plot the final classification of Growing_Stress
plot_final_classification(final_df)

#%% [markdown] -- haeyeon end of unknown factor visualization

#%%
# Set threshold based on the mean or a slightly higher value
  # or use another positive threshold based on analysis

# Convert Growing_Stress from 2 to 0 and 1 based on unknown_factor
final_df['Growing_Stress'] = final_df.apply(
    lambda row: 1 if row['unknown_factor'] > threshold else 0 
    if row['Growing_Stress'] == 2 else row['Growing_Stress'], axis=1
)
# Check the updated values
print(final_df['Growing_Stress'].value_counts())

#%%
final_df = final_df.drop('unknown_factor',axis = 1)

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

#%%
#from sklearn.model_selection import train_test_split
#Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

#scaler = StandardScaler()
#Xtrain_scaled = scaler.fit_transform(Xtrain)
#Xtest_scaled = scaler.transform(Xtest)

#svm_model = SVC(kernel='linear', C=1.0, gamma='scale', probability=True)
#svm_model.fit(Xtrain_scaled, Ytrain)

# Predictions
#y_pred_class = svm_model.predict(Xtest_scaled)
#y_pred_prob = svm_model.predict_proba(Xtest_scaled)[:, 1]

# %%[markdown]
# Is SVM model ideal for the dataset? SVM is more ideal for smaller datasets
# Should I take a percentage of the data and test again to see if computation time lowers?

'''New code to take random sample(10%) of the data'''
X_clean_sampled = X_clean.sample(frac=0.1, random_state=42)
y_clean_sampled = y_clean[X_clean_sampled.index]


from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_clean_sampled, y_clean_sampled, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Xtrain_scaled = scaler.fit_transform(Xtrain)
Xtest_scaled = scaler.transform(Xtest)

from sklearn.svm import SVC
svm_model = SVC(kernel='rbf', C=1.0, probability=True)
svm_model.fit(Xtrain_scaled, Ytrain)


# Predictions
y_pred_class = svm_model.predict(Xtest_scaled)
y_pred_prob = svm_model.predict_proba(Xtest_scaled)[:, 1]

#%%[markdown]
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
accuracy = accuracy_score(Ytest, y_pred_class)
print(f"Accuracy: {accuracy:.2f}")


print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(Ytest, y_pred_class)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

auc_score = roc_auc_score(Ytest, y_pred_prob)
print(f"ROC-AUC: {auc_score:.2f}")

fpr, tpr, _ = roc_curve(Ytest, y_pred_prob)
plt.plot(fpr, tpr, label=f"SVM (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

#Model has an accuracy of 0.75 and an AUC of 0.83. Let's try to see what parameters we can optimize to improve predictive power.

from sklearn.model_selection import GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Grid search
grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(Xtrain_scaled, Ytrain)

print("Best Parameters:", grid_search.best_params_)

#Analysis and conclusion for SVM:
#The target variable was Growing Stress similar to other binary models in this document. 
#We've evaluated the performance of the model through metrics like accuracy (0.68) and AUC(0.71)
#SVM has presented us with computational challanges likely due to the number of features, one-hot encoding, and data size
#To mitigate that, we tried to tune our parameters (TBD) and did our SVM on a sample of the dataset(10%)
#The difficulty and output leaves us wondering if SVM is the ideal model to move forward with
# %%[]
