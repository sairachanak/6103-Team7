# %%[markdown]
# Introduction  : How can we develop models to assess mental stress,
# and which factors are most influential in predicting 
# mental health outcomes
#
# SMART questions : 
#
# 1. What are the top 5 factors that influence the growing stress?
#
# 2. Do people with family history receive treatment or not?
#
# 3. Q3. What are the factors that impact the growing stress for students?




# %%[markdown]
# Importing Libraries
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
from sklearn.model_selection import KFold
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
from sklearn.metrics import ConfusionMatrixDisplay
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

# %%[markdown]
# Importing the Dataset
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

# %%[markdown]
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

# %%
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


# %%[markdown] - Haeyeon
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

# %%[markdown]
# Statistical Testing for Treatment

# %%[markdown]
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


# %%[markdown]
# As the Timestamp, Country, self-employed are independednt of Growing Stress and also have a highly unbalanced data
# it is better to remove those columns for our Modelling.

# %%[markdown]
# Modelling

encoded_final_df = encoded_final_df.drop(columns=['Timestamp', 'self_employed'])
data = encoded_final_df.drop('Growing_Stress', axis=1)
f_data = encoded_final_df['Growing_Stress']
X = data
y = f_data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, f_data, test_size=0.2, random_state=42)
methodDict = {}

# %%
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
# %%[markdown]
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

    coefficients = logreg.coef_[0]
    features = Xtrain.columns

    # Create a DataFrame for feature importances
    feature_importances = pd.DataFrame({
        'Feature': features,
        'Importance': coefficients  # Direct coefficients without abs()
    }).sort_values(by='Importance', ascending=False)

    # Display the top 5 features
    top_5_features = feature_importances.head(5)
    print("Top 5 Features by Importance:")
    print(top_5_features)

    # Plot the top 5 features
    plt.figure(figsize=(10, 6))
    plt.bar(top_5_features['Feature'], top_5_features['Importance'], color='skyblue')
    plt.title('Top 5 Feature Importances (Logistic Regression)')
    plt.xlabel('Feature')
    plt.ylabel('Importance (Coefficients)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    
logisticRegression()

# %%
model = LogisticRegression(max_iter=1000)

# Create the feature importances visualizer
visualizer = FeatureImportances(model)

# Fit the visualizer and the model
visualizer.fit(Xtrain, Ytrain)

# Show the visualization
visualizer.show()
# %%
y_pred = model.predict(Xtest)

# Generate classification report
print("Classification Report:")
print(classification_report(Ytest, y_pred))

# %%[markdown]

# Desciion Tree

target = encoded_final_df['Growing_Stress']

# Drop the target variable from the features
features = encoded_final_df.drop(['Growing_Stress'], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

# %%

# Base model with default parameters
base_params = {
    'max_depth': 10,         # No limit by default
    'min_samples_split': 10,    # Default value
    'min_samples_leaf': 20,      # Default value
    'max_features': 'sqrt',      # Consider all features
}

# Define K-Fold cross-validator
kfold = KFold(n_splits=10, shuffle=True, random_state=0)

# %%[markdown]

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

# %%
# Tune max_depth
tune_and_plot('max_depth', [None, 5, 6, 7, 8, 9, 10], base_params, X_train, y_train)

# At 9 test accuarcy seems to be the highest hence better to set max_depth as 9
# If the tree becomes complex, we can decrease it later to 6 or 7 as test accuracy is still high for them
# %%
# Tune min_samples_split
tune_and_plot('min_samples_split', [2, 5, 10, 15, 20], base_params, X_train, y_train)
# %%
# Tune min_samples_leaf
tune_and_plot('min_samples_leaf', [1, 2, 5, 10], base_params, X_train, y_train)
# %%
# Tune max_features
tune_and_plot('max_features', ['sqrt', 'log2'], base_params, X_train, y_train)
# %%
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

# %%
y_pred = final_model.predict(X_test)

# Generate classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))
# %%
# For depth = 9 it seems that the model has slightly greater test accuracy than train accuracy but the 
# cross validation accuracy is high hence, this seems to be a good fit

# %%
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
# %%
# Feature Importance

feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': final_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Print feature importances
print("Feature Importances (Descending Order):")
print(feature_importances)

# Visualize feature importances
plt.figure(figsize=(10, 6))
plt.bar(feature_importances['Feature'], feature_importances['Importance'], color='teal')
plt.title('Feature Importance (Decision Tree)')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()




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


# %%[markdown]
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
# %%[markdown]

tune_and_plot('n_estimators', [10, 20, 30, 50], base_params, X_train, y_train)

# After estimators of 10 it seems that the train error rate increases compared to test error rate
# hence best to set n_estimators as 10
# %%

tune_and_plot('max_depth', [5, 6, 7, 8,9, 10], base_params, X_train, y_train)
# It seems that at max_depth = 7 the test accuracy is highest comapred to train hence 
# better to set the max_depth to 7

# %%

tune_and_plot('min_samples_split', [5, 10, 15, 20], base_params, X_train, y_train)

# It has same accuracy for all values hence better to set min_samples_split to 20
# as it will reduce the complexity of the model
# %%
tune_and_plot('min_samples_leaf', [1, 5, 10, 20], base_params, X_train, y_train)

# According to the graph it seems that min_samples_leaf has highest accuarcy hence better to set
# min_samples_leaf to 20
# %%

tune_and_plot('max_features', ['sqrt', 'log2'], base_params, X_train, y_train)

# %%
tune_and_plot('bootstrap', [True, False], base_params, X_train, y_train)

# According to the graph it is better to choose booststrap = TRUE
# %%
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

# %%
# As seen the training testing accuracies along with highest cross validation accuracies seems to be 
# almost same, hence we can choose this fit, if we want to reduce complexity and compromise accuracy it is 
# good to choose depth 6 as well

# %%
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
# %%
y_pred = final_model.predict(X_test)

# Generate classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# %%
# Train the Random Forest model on training data
final_model.fit(Xtrain, Ytrain)

# Extract feature importances
feature_importances = final_model.feature_importances_

# Create a DataFrame to sort and visualize feature importances
importance_df = pd.DataFrame({
    'Feature': Xtrain.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Display the top 5 important features
top_5_features = importance_df.head(5)
print("Top 5 Features by Importance:")
print(top_5_features)

# Plot the top 5 features
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(top_5_features['Feature'], top_5_features['Importance'], color='skyblue')
plt.title('Top 5 Feature Importances (Random Forest)')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# %% KNN

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
# As we can see the model is overfitting and the cross validation results suggest the model has high variance 
# and changes for different data.

# %%  Abirham - Q2
# Do people with Family history receive treatment or not?
# how well "Family History" predicts whether a person will seek treatment
# %%[markdown]
# EDA between Family History and Treatment
plt.figure(figsize=(8, 6))
sns.countplot(x='family_history', hue='treatment', data=encoded_final_df, palette='Set2')
plt.title('Family History vs Treatment')
plt.xlabel('Family History')
plt.ylabel('Count')
plt.legend(title='Treatment', loc='upper right')
plt.show()

# %%[markdown]
# Statistcial Testing
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

# %%[markdown]
# Without Family History (family_history=0):
# A larger proportion of people do not seek treatment (green bar) compared to those who do (orange bar).
# This suggests that individuals without a family history of mental health issues are less likely to seek treatment overall.
# With Family History (family_history=1):
# A significantly larger proportion of people with a family history of mental health issues seek treatment (orange bar) compared to those who do not (green bar).
# This suggests that having a family history of mental health issues is positively associated with seeking treatment.


# %%[markdown]
#  Modelling - Logistic Regression
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


# %%[markdown]
# Feature importance for Logistic Regression
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
# we got a positive coeficient and  it means having a family history  increases the likely hood of getting treatment
# %%[markdown]
# Evaluation for the model
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

# %%[markdown]
# Modelling - KNN
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
# %%[markdown]
# The KNN doesnot perform well with this data.

# %%[markdown]
# Modelling - Random Forest

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

# From the above models it could be seen that the treatment with different models has the highest coefficent for
# family history





# %%[markdown]
# Q3 - What are the top 5 factors thats impact the growing stress in students  
# Data preparation
# Filter data for 'Student' occupation

health_data_student = health_data_backup[health_data_backup['Occupation'] == 'Student']

# Drop unnecessary columns same as previous
health_data_student = health_data_student.drop(columns=['Country'])

# Check the first few rows of the filtered data
health_data_student.head()

# Check student data info
health_data_student.info()


# %%[markdown]
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

# %%[markdown]
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

# %%[markdown]

# Correlation heatmap for variables correlated with Growing Stress for students
student_corr = health_data_student[['Days_Indoors', 'Growing_Stress', 'Changes_Habits', 'Coping_Struggles', 
                             'Mood_Swings', 'Social_Weakness', 'treatment']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(student_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap for Growing Stress and Related Variables for Students')
plt.show()

# %%[markdown]

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

# %%[markdown]

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
