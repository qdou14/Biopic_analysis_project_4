import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class DataProcess:
    """
    A class designed to process and analyze biopics data from a CSV file.

    This class includes methods for data loading, conversion, enhancement,
    and exploratory data analysis.

    Attributes:
        data (DataFrame): A pandas DataFrame that stores the biopics data.
    """

    def __init__(self, url):
        """
        Initializes the DataProcess object by loading data from a specified CSV file URL.

        Parameters:
            url (str): The URL to the CSV file containing the biopics data.
        """
        self.data = pd.read_csv(url, encoding='ISO-8859-1')

    def get_data(self):
        """
        Retrieves the currently loaded biopics data.

        Returns:
            DataFrame: The biopics data.
        """
        return self.data

    def merge_movie_data(self, movie_data_url):
        """
        Merges movie data into the biopics data based on the movie title.

        Parameters:
            movie_data_url (str): The URL to the CSV file containing the movie data.
        """
        movies_df = pd.read_csv(movie_data_url)

        # Rename 'title' column to 'movie_title' to match movies_df
        self.data = self.data.rename(columns={'title': 'movie_title'})

        # Clean the 'movie_title' column in movies_df
        movies_df['movie_title'] = movies_df['movie_title'].str.replace("Â", "").str.rstrip()

        # Merge the datasets
        self.data = self.data.merge(movies_df[['movie_title', 'budget', 'imdb_score']], on='movie_title', how='left')

        # Return the merged DataFrame
        return self.data
    
    def convert_box_office(self):
        """
        Converts box office values in the DataFrame from string format to a numerical format.
        This method specifically handles values represented in millions (with 'M') and 
        thousands (with 'K'), converting them to their numerical equivalents.

        This method modifies the 'box_office' column of the DataFrame, replacing string values 
        with their corresponding float representations. Values with '-' are replaced with NaN.

        Returns:
        DataFrame: The DataFrame with the 'box_office' column values converted to floats.
        """
        def convert_value(value):
            """
            Converts a string representation of a monetary value to a float. This function handles
            the conversion of values represented with 'M' (millions) and 'K' (thousands) by removing
            these characters and scaling the value accordingly.

            If the input is NaN or a non-convertible string, it returns NaN. Otherwise, it strips 
            away any dollar signs ('$') and commas (',') and converts the string to a float.

            Args:
                value (str): A string representing a monetary value, possibly containing 'M', 'K', 
                            a dollar sign, or commas.

            Returns:
                float: The numerical representation of the input string. Returns NaN for non-numeric 
                    or unconvertible inputs.
            """
            if pd.isnull(value):
                return np.nan
            value = value.replace('$', '').replace(',', '')
            if 'M' in value:
                return float(value.replace('M', '')) * 1e6
            if 'K' in value:
                return float(value.replace('K', '')) * 1e3
            return float(value)

        self.data['box_office'] = self.data['box_office'].replace('-', np.nan)
        self.data['box_office'] = self.data['box_office'].apply(convert_value)
        self.data['box_office'] = self.data['box_office'].astype('float64')
        return self.data
    
    def add_category(self):
        """
        Adds a new 'category' column to the DataFrame based on 'subject_race' and 'subject_sex' columns.

        This method applies a categorization function to each row of the DataFrame. The category is determined 
        by the race and sex of the subject. The categories include 'White Male', 'White Female', 
        'Non-White Male', 'Non-White Female', or 'Unknown'. If the race is missing, the category is set as pd.NA.

        The new column 'category' is appended to the existing DataFrame.

        Parameters:
            self: Instance of the class containing the DataFrame.

        Returns:
            None: The method directly modifies the instance's DataFrame by adding a new column.
        """
        def categorize(row):
            # Internal function to determine the category of a given row
            if pd.isna(row['subject_race']):
                return pd.NA  
            elif row['subject_race'] == 'White':
                if row['subject_sex'] == 'Male':
                    return 'White Male'
                elif row['subject_sex'] == 'Female':
                    return 'White Female'
            else:
                if row['subject_sex'] == 'Male':
                    return 'Non-White Male'
                elif row['subject_sex'] == 'Female':
                    return 'Non-White Female'
            return 'Unknown' 

        # Apply the categorize function to each row and create a new column 'category'
        self.data['category'] = self.data.apply(categorize, axis=1)

    def check_data(self):
        """
        Prints information about the DataFrame including data types, non-null values, and
        percentage of missing values in each column.
        """
        print(self.data.info())
        print(self.data.isna().sum())

    def describe_categorical(self):
        """
        Provides descriptive statistics for all categorical variables in the DataFrame.

        Returns:
            DataFrame: Summary statistics for categorical data.
        """
        return self.data.describe(include=['object'])
    
    
    
    




    
    

    

        

