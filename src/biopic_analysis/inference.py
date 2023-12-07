import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from adjustText import adjust_text

class Inference:
    """
    A class to perform various data visualization inferences on a movie dataset.

    Attributes:
    df (DataFrame): A pandas DataFrame containing movie data with attributes such as box office, 
                    diversity categories, year of release, etc.

    Methods:
    bechdel_test_plot(subject_col, bechdel_col): Generates a horizontal bar plot showing the percentage of movies that pass or fail the Bechdel test by specified subject categories.
    proportion_in_top_box_office(column, hue): Displays a side-by-side bar chart showing the proportion of movies in top box office categorized by specified column and hue.
    box_office_diversity_correlation(): Visualizes the correlation between diversity categories and box office performance using scatter plots.
    diversity_trends_over_time(): Analyzes and visualizes the trends in movie diversity over time with line plots.
    """
    
    def __init__(self, df):
        """
        Initialize the Inference class with a DataFrame containing movie data.

        Parameters:
        - df (DataFrame): A pandas DataFrame containing movie data with attributes like box office, 
                          diversity categories, year of release, etc.
        """
        self.df = df

    
    def proportion_in_top_box_office(self, column, hue):
        """
        Displays a side-by-side bar chart showing the proportion of movies in top box office categorized by specified column and hue.

        Parameters:
        column (str): The name of the main DataFrame column to categorize data on the x-axis.
        hue (str): The name of the DataFrame column to create subcategories within the main category.

        Returns:
        None: This method displays the plot and does not return any value.
        """
        top_100 = self.df.nlargest(100, 'box_office')
        
        fig, axes = plt.subplots(1, 2, figsize=(24, 8)) 

        grouped_data = top_100.groupby([column, hue]).size().unstack().fillna(0)
        grouped_data.plot(kind='bar', stacked=False, color=['skyblue', 'lightgreen'], ax=axes[0])
        axes[0].set_title(f'Proportion of {column.capitalize()} and {hue.capitalize()} in Top 100 Box Office Movies - Matplotlib')
        axes[0].set_xlabel(column.capitalize())
        axes[0].set_ylabel('Number of Movies')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].legend(title=hue.capitalize())

        sns.countplot(x=column, hue=hue, data=top_100, palette='Set2', ax=axes[1])
        axes[1].set_title(f'Proportion of {column.capitalize()} and {hue.capitalize()} in Top 100 Box Office Movies - Seaborn')
        axes[1].set_xlabel(column.capitalize())
        axes[1].set_ylabel('Count')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].legend(title=hue.capitalize())

        plt.tight_layout()
        plt.show()


    def box_office_diversity_correlation(self):
        """
        Analyze and visualize the correlation between movie diversity (combined race and sex in 'category') 
        and box office performance using scatter plots with Matplotlib and Seaborn.

        Returns:
        - None: This method displays the plot and does not return any value.
        """
        valid_data = self.df.dropna(subset=['box_office', 'category'])
        
        plt.figure(figsize=(12, 8))
        categories = valid_data['category'].unique()
        for category in categories:
            category_data = valid_data[valid_data['category'] == category]
            plt.scatter(category_data['category'], category_data['box_office'], alpha=0.5, label=category)
        
        plt.title('Box Office Performance by Diversity Category - Matplotlib')
        plt.xlabel('Diversity Category')
        plt.ylabel('Box Office')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='category', y='box_office', hue='category', data=valid_data, palette='Set2')
        plt.title('Box Office Performance by Diversity Category - Seaborn')
        plt.xlabel('Diversity Category')
        plt.ylabel('Box Office')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    
    


