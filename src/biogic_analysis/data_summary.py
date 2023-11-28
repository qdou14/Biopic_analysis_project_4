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

    Methods:
        get_data(): Retrieves the loaded data.
        convert_box_office(): Converts box office values to numerical format.
        fetch_bechdel_data(title): Fetches Bechdel rating for a given movie.
        update_dataframe_with_bechdel(): Adds Bechdel ratings to the DataFrame.
        get_shape(): Returns the shape (dimensions) of the DataFrame.
        missing_data(): Calculates the percentage of missing data in each column.
        check_data(): Prints data overview and missing value statistics.
        describe_categorical(): Provides summary statistics for categorical data.
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
    
    def convert_box_office(self):
        """
        Converts box office values in the data from string to numerical format. Handles
        values in millions (M) and thousands (K).

        Returns:
            DataFrame: The updated DataFrame with converted box office values.
        """
        def convert_value(value):
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

    def fetch_bechdel_data(self, title):
        """
        Fetches the Bechdel rating for a given movie title from the Bechdel Test API.

        Parameters:
            title (str): The title of the movie for which the Bechdel rating is required.

        Returns:
            dict: A dictionary containing the Bechdel rating. None if not found.
        """
        url = "http://bechdeltest.com/api/v1/getMoviesByTitle"
        params = {'title': title}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            movies = response.json()
            for movie in movies:
                if movie['title'].lower() == title.lower():
                    return {'rating': movie.get('rating')}
        return {'rating': None}

    def update_dataframe_with_bechdel(self):
        """
        Updates the DataFrame with Bechdel ratings for each movie.

        Returns:
            DataFrame: The updated DataFrame including Bechdel ratings.
        """
        self.data['bechdel_rating'] = None
        for index, row in self.data.iterrows():
            bechdel_data = self.fetch_bechdel_data(row['title'])
            self.data.at[index, 'bechdel_rating'] = bechdel_data['rating']
        return self.data
    
    def add_category(self):
        def categorize(row):
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

        self.data['category'] = self.data.apply(categorize, axis=1)

    def get_shape(self):
        """
        Returns the shape (dimensions) of the DataFrame.

        Returns:
            tuple: A tuple representing the dimensions of the DataFrame.
        """
        return self.data.shape

    def missing_data(self):
        """
        Calculates and returns the percentage of missing data in each column of the DataFrame.

        Returns:
            Series: A pandas Series representing the percentage of missing data per column.
        """
        return self.data.isna().sum() / len(self.data) * 100

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
    
    
    
    




    
    

    

        

