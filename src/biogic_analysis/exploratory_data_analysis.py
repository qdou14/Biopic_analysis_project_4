import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EDA:
    """
    A class for performing exploratory data analysis on a pandas DataFrame.

    Attributes:
        df (pd.DataFrame): A pandas DataFrame containing the data for analysis.

    Methods:
        - __init__(self, df): Initialize the EDA class with a DataFrame.
        - unique_value_counts(self, column_name): Return counts of unique values in a specified column.
        - combined_category_plot(self, type_col, category_col): Create stacked bar plots using Matplotlib and Seaborn.
        - describe_stats(self, column_name): Provide descriptive statistics for a specified column.
        - column_boxplot(self, x_column, y_column, hue): Create a boxplot for specified columns.
    """
    def __init__(self, df):
        """
        Initializes the EDA class with the given DataFrame.

        Parameters:
            df (pd.DataFrame): A DataFrame containing the data for exploratory analysis.
        """
        self.df = df
    
    def unique_value_counts(self, column_name):
        """
        Return the counts of unique values in a specified column in the DataFrame.

        This method returns a Series with the counts of unique values, which can be useful for
        understanding the distribution of categorical data. NaN values are dropped by default.

        Parameters:
            column_name (str): The name of the column from which to retrieve value counts.

        Returns:
            pandas.Series: A Series containing counts of unique values in the specified column.
        """
        return self.df[column_name].value_counts(dropna=True)

    def combined_category_plot(self, type_col='type_of_subject', category_col='category'):
        """
        Create stacked bar plots for specified columns using both Matplotlib and Seaborn.

        Parameters:
            type_col (str, optional): The name of the DataFrame column for the x-axis.
                                      Defaults to 'type_of_subject'.
            category_col (str, optional): The name of the DataFrame column for the hue in the plot.
                                          Defaults to 'category'.
        Returns: None
        """
        df_cleaned = self.df.dropna(subset=[category_col])
        # Set up the matplotlib figure
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(24, 6))
        
        # Matplotlib plot - stacked bar plot
        grouped_data = df_cleaned.groupby([type_col, category_col]).size().unstack()
        grouped_data.plot(kind="bar", stacked=True, colormap="viridis", ax=ax[0])
        ax[0].set_title(f"Stacked Barplot with Matplotlib\n{type_col} by {category_col}")
        ax[0].set_xlabel(type_col)
        ax[0].set_ylabel('Count')
        ax[0].tick_params(axis='x', rotation=45)

        # Seaborn plot - stacked bar plot
        sns.histplot(data=df_cleaned, x=type_col, hue=category_col,
                     multiple='stack', palette='viridis', shrink=0.8, ax=ax[1])
        ax[1].set_title(f"Stacked Barplot with Seaborn\n{type_col} by {category_col}")
        ax[1].set_xlabel(type_col)
        ax[1].set_ylabel('Count')
        ax[1].tick_params(axis='x', rotation=45)

        # Adjust the layout and display the plot
        plt.tight_layout()
        plt.show()

    def describe_stats(self, column_name):
        """
        Return descriptive statistics for a specified column in the DataFrame.

        This method returns a Series that describes the central tendency, dispersion, and shape
        of the dataset's distribution, excluding NaN values. It includes measures such as mean,
        median, mode, standard deviation, minimum and maximum values, and quartiles.

        Parameters:
            column_name (str): The name of the column for which to retrieve statistics.

        Returns:
            pandas.Series: A Series containing descriptive statistics of the specified column.
        """
        return self.df[column_name].describe()

    def column_boxplot(self, x_column, y_column, hue=None):
        """
        Create a boxplot for the distribution of specified columns using Seaborn and Matplotlib.

        This method uses Seaborn to create a boxplot, which is a standardized way of displaying the distribution
        of data based on a five-number summary: minimum, first quartile (Q1), median, third quartile (Q3), and maximum.
        It can also show outliers beyond the whiskers.
        
        Parameters:
            x_column (str): The name of the DataFrame column for the x-axis.
            y_column (str): The name of the DataFrame column for the y-axis.
            hue (str, optional): The name of the DataFrame column for hue categorization.

        Returns:
            None
        """
        # Handle missing data
        self.df = self.df.dropna(subset=[x_column, y_column])  # or use other methods to handle NA values
        
        plt.figure(figsize=(12, 8))
        sns.boxplot(
            x=x_column,
            y=y_column,
            data=self.df,
            palette="Set3",
            hue=hue
        )
        plt.title(f"Distribution of {y_column} in different {x_column}")
        if hue:
            plt.legend(title=hue)
        plt.show()

    
    


            