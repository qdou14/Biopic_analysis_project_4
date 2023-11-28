import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EDA:
    """
    A class for performing exploratory data analysis on a pandas DataFrame.

    Attributes:
        df (pd.DataFrame): A pandas DataFrame containing the data for analysis.

    Methods:
        unique_values(column_name): Returns unique values from a specified column.
        unique_value_counts(column_name): Returns counts of unique values in a specified column.
        category_plt(type_col, category_col): Creates a stacked barplot using Matplotlib.
        category_plt_seaborn(type_col, category_col): Creates a stacked barplot using Seaborn.
        race_sex_stacked_plot(): Stacked bar chart of subject_race and subject_sex.
        column_plt(column): Horizontal boxplot for a specified column using Matplotlib.
        column_sns(column): Horizontal boxplot for a specified column using Seaborn.
        column_boxplot(x_column, y_column, hue): Boxplot for distribution of specified columns using Seaborn.
        describe_stats(column_name): Descriptive statistics for a specified column.
        bar_plot_matplotlib(column): Bar chart for a specified column using Matplotlib.
        bar_plot_seaborn(column): Bar chart for a specified column using Seaborn.
    """
    def __init__(self, df):
        self.df = df

    def unique_values(self, column_name):
        """
        Return unique values from a specified column in the DataFrame.

        Parameters:
            column_name (str): The name of the column from which to retrieve unique values.

        Returns:
            numpy.ndarray: An array of unique values from the specified column.
        """
        return self.df[column_name].dropna().unique()
    
    def unique_value_counts(self, column_name):
        """
        Return the counts of unique values in a specified column in the DataFrame.

        Parameters:
            column_name (str): The name of the column from which to retrieve value counts.

        Returns:
            pandas.Series: A Series containing counts of unique values in the specified column.
        """
        return self.df[column_name].value_counts(dropna=True)
    
    def category_plt(self, type_col='type_of_subject', category_col='category'):
        """
        Create a stacked barplot for specified columns using Matplotlib.

        Parameters:
            type_col (str, optional): The name of the DataFrame column for the x-axis. Defaults to 'type_of_subject'.
            category_col (str, optional): The name of the DataFrame column for stacked categories. Defaults to 'category'.
        """
        plt.figure(figsize=(12, 6))
        grouped_data = self.df.groupby([type_col, category_col]).size().unstack()
        grouped_data.plot(kind="bar", stacked=True, colormap="viridis", figsize=(12, 6))
        plt.xlabel(type_col)
        plt.ylabel("Count")
        plt.title(f"Stacked Barplot of {type_col} by {category_col}")

        plt.show()

    def category_plt_seaborn(self, type_col='type_of_subject', category_col='category'):
        """
        Create a stacked barplot for specified columns using Seaborn.

        Parameters:
            type_col (str, optional): The name of the DataFrame column for the x-axis. Defaults to 'type_of_subject'.
            category_col (str, optional): The name of the DataFrame column for stacked categories. Defaults to 'category'.
        """
        plt.figure(figsize=(12, 6))
        sns.histplot(
            data=self.df,
            x=type_col, 
            hue=category_col, 
            multiple="stack", 
            palette="viridis", 
            shrink=0.8  
        )
        plt.xlabel(type_col)
        plt.ylabel("Count")
        plt.title(f"Stacked Barplot of {type_col} by {category_col}")

        plt.show()

    def race_sex_stacked_plot(self):
        """
        Create a stacked bar chart of subject_race and subject_sex.
        
        This method generates a bar chart using the DataFrame's 'subject_race' and 'subject_sex' columns. 
        It groups the data by 'subject_race' and stacks each race's count by 'subject_sex'. 
        
        Returns:
        - None
        """
        race_sex_counts = self.df.groupby(['subject_race', 'subject_sex']).size().unstack()
        
        race_sex_counts.plot(kind='bar', stacked=True, colormap='coolwarm', figsize=(10, 6))
        
        plt.title('Stacked Bar Chart of Subject Race by Sex')
        plt.xlabel('Subject Race')
        plt.ylabel('Count')
        
        plt.show()

    def column_plt(self, column):
        """
        Create a horizontal boxplot for a specified column using Matplotlib.

        Parameters:
            column (str): The name of the DataFrame column to be plotted.
        """
        plt.figure(figsize=(15, 10))
        plt.boxplot(self.df[column].dropna(), vert=False) 
        plt.title(f"Boxplot of {column}")
        plt.xlabel("Values")
        plt.show()
    
    def column_sns(self, column):
        """
        Create a horizontal boxplot for a specified column using Seaborn.

        Parameters:
            column (str): The name of the DataFrame column to be plotted.
        """
        plt.figure(figsize=(15, 10))
        sns.boxplot(data=self.df[column], orient="h", palette="coolwarm")
        plt.title(f"Boxplot of {column}")
        plt.xlabel("Values")
        plt.show()

    def column_boxplot(self, x_column, y_column, hue=None):
        """
        Create a boxplot for the distribution of specified columns using Seaborn.

        Parameters:
            x_column (str): The name of the DataFrame column for the x-axis.
            y_column (str): The name of the DataFrame column for the y-axis.
            hue (str, optional): The name of the DataFrame column for hue categorization.
        """
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

    def describe_stats(self, column_name):
        """
        Return descriptive statistics for a specified column in the DataFrame.

        Parameters:
            column_name (str): The name of the column for which to retrieve statistics.

        Returns:
            pandas.Series: A Series containing descriptive statistics of the specified column.
        """
        return self.df[column_name].describe()
    
    def bar_plot_matplotlib(self, column):
        """
        Create a bar chart for a specified column using Matplotlib.

        Parameters:
            column (str): The name of the DataFrame column to be plotted.
        """
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            print(f"Column {column} is not numeric and cannot be plotted as a bar chart.")
            return

        plt.figure(figsize=(12, 8))
        plt.bar(self.df[column].value_counts().index, self.df[column].value_counts().values, color='cadetblue')
        plt.title(f'Bar Chart of {column} - Matplotlib')
        plt.xlabel(column.capitalize())
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def bar_plot_seaborn(self, column):
        """
        Create a bar chart for a specified column using Seaborn.

        Parameters:
            column (str): The name of the DataFrame column to be plotted.
        """
        # Ensure the column is suitable for a bar plot
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            print(f"Column {column} is not numeric and cannot be plotted as a bar chart.")
            return

        # Use Seaborn to create a bar chart
        plt.figure(figsize=(12, 8))
        sns.barplot(x=self.df[column].value_counts().index, y=self.df[column].value_counts().values, palette='Spectral')
        plt.title(f'Bar Chart of {column} - Seaborn')
        plt.xlabel(column.capitalize())
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    


            