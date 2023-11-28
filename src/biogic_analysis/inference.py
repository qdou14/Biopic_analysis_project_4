import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

class Inference:
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
        Generate grouped bar charts showing the proportion of movies within specified 
        categories and subcategories in the top 100 box office movies using Matplotlib and Seaborn.

        Parameters:
        - column (str): The name of the main DataFrame column to categorize data on the x-axis.
        - hue (str): The name of the DataFrame column to create subcategories within the main category.

        Returns:
        - None: This method displays the plot and does not return any value.
        """
        # Select the top 100 movies by box office
        top_100 = self.df.nlargest(100, 'box_office')
        
        # Use Matplotlib to create a grouped bar chart
        plt.figure(figsize=(12, 8))
        grouped_data = top_100.groupby([column, hue]).size().unstack().fillna(0)
        grouped_data.plot(kind='bar', stacked=False, color=['skyblue', 'lightgreen'])
        plt.title(f'Proportion of {column.capitalize()} and {hue.capitalize()} in Top 100 Box Office Movies - Matplotlib')
        plt.xlabel(column.capitalize())
        plt.ylabel('Number of Movies')
        plt.xticks(rotation=45)
        plt.legend(title=hue.capitalize())
        plt.tight_layout()
        plt.show()
        
        # Use Seaborn to create a grouped bar chart
        plt.figure(figsize=(12, 8))
        sns.countplot(x=column, hue=hue, data=top_100, palette='Set2')
        plt.title(f'Proportion of {column.capitalize()} and {hue.capitalize()} in Top 100 Box Office Movies - Seaborn')
        plt.xlabel(column.capitalize())
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(title=hue.capitalize())
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
    
    def diversity_trends_over_time(self):
        """
        Analyze and visualize the trends in movie diversity over time. It computes the yearly proportion 
        of movies for different diversity categories and plots this data using line plots with Matplotlib and Seaborn.

        Returns:
        - None: This method displays the plot and does not return any value.
        """
        # Ensure 'year_release' and 'category' columns do not have NaN values
        valid_data = self.df.dropna(subset=['year_release', 'category'])

        # Group by year of release and diversity category, then count the number of movies in each group
        yearly_diversity = valid_data.groupby(['year_release', 'category']).size().unstack().fillna(0)

        # Calculate the annual proportion for each category
        yearly_proportions = yearly_diversity.div(yearly_diversity.sum(axis=1), axis=0) * 100
        
        plt.figure(figsize=(14, 8))
        for category in yearly_proportions.columns:
            plt.plot(yearly_proportions.index, yearly_proportions[category], marker='o', label=category)
        
        plt.title('Diversity Trends Over Time - Matplotlib')
        plt.xlabel('Year of Release')
        plt.ylabel('Proportion of Movies (%)')
        plt.legend(title='Diversity Category')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Use Matplotlib to plot a line chart
        plt.figure(figsize=(14, 8))
        sns.lineplot(data=yearly_proportions)
        plt.title('Diversity Trends Over Time - Seaborn')
        plt.xlabel('Year of Release')
        plt.ylabel('Proportion of Movies (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    


