import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

class Inference:
    """
    A class to perform various data visualization inferences on a movie dataset.

    Attributes:
    df (DataFrame): A pandas DataFrame containing movie data with attributes such as box office, 
                    diversity categories, year of release, etc.

    Methods:
    bechdel_test_plot(subject_col, bechdel_col): Generates a horizontal bar plot showing the percentage of movies that pass or fail the Bechdel test by specified subject categories.
    bechdel_category_boxplot(category_col, bechdel_col): Creates a box plot to analyze the distribution of Bechdel test ratings across different categories.
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

    def bechdel_test_plot(self, subject_col='type_of_subject', bechdel_col='bechdel_rating'):
        """
        Generates a horizontal bar plot showing the percentage of movies that pass or fail the Bechdel test by specified subject categories.

        Parameters:
        subject_col (str): The name of the DataFrame column to categorize data on the y-axis.
        bechdel_col (str): The name of the DataFrame column containing Bechdel test ratings.

        Returns:
        None
        """
        # Convert numerical ratings to 'Pass' or 'Fail'
        self.df[bechdel_col] = self.df[bechdel_col].apply(lambda x: 'Pass' if x == 3 else 'Fail' if x in [1, 2] else None)

        # Drop rows where bechdel_rating is None
        df_cleaned = self.df.dropna(subset=[bechdel_col])

        # Group by subject_col and bechdel_col, then unstack to prepare for the percentage calculation
        bechdel_summary = df_cleaned.groupby([subject_col, bechdel_col]).size().unstack().fillna(0)
        bechdel_summary['Total'] = bechdel_summary.sum(axis=1)
        bechdel_summary['Passed'] = (bechdel_summary.get('Pass', 0) / bechdel_summary['Total']) * 100
        bechdel_summary['Failed'] = (bechdel_summary.get('Fail', 0) / bechdel_summary['Total']) * 100
        
        # Sorting by the total number for better visibility
        bechdel_summary = bechdel_summary.sort_values(by='Total', ascending=False)
        
        # Plotting
        plt.figure(figsize=(10, 8))
        bechdel_summary[['Failed', 'Passed']].plot(kind='barh', stacked=True, color=['grey', 'lightblue'], figsize=(10, 8))
        plt.xlabel('Percentage')
        plt.ylabel(subject_col)
        plt.title('Bechdel Test Performance by ' + subject_col)
        plt.legend(title='Bechdel test', loc='lower right')
        plt.gca().invert_yaxis()  # To match the example's layout
        plt.show()

    def bechdel_category_boxplot(self, category_col='category', bechdel_col='type_of_subject'):
        """
        Create a box plot to analyze the distribution of Bechdel test ratings across different categories.
        """
        # Clean the data by replacing 'None' and 'NaN' in bechdel_rating with a default value or category
        self.df[bechdel_col] = self.df[bechdel_col].replace({None: 'Unknown', np.nan: 'Unknown'})
            
        # Replace 'NaN' and '<NA>' in category with 'Unknown'
        self.df[category_col] = self.df[category_col].replace({np.nan: 'Unknown', '<NA>': 'Unknown'})

        # Create a box plot
        plt.figure(figsize=(12, 8))
        sns.boxplot(x=category_col, y=bechdel_col, data=self.df, palette="Set3")

        # Label the plot
        plt.title('Bechdel Rating Distribution by Category')
        plt.xlabel('Category')
        plt.ylabel('Bechdel Rating')

        # Show the plot
        plt.show()


    
    def proportion_in_top_box_office(self, column, hue):
        # 选择票房前100的电影
        top_100 = self.df.nlargest(100, 'box_office')
        
        # 创建一个带有两个子图的图表布局
        fig, axes = plt.subplots(1, 2, figsize=(24, 8))  # 两个子图并排

        # 使用 Matplotlib 创建分组条形图
        grouped_data = top_100.groupby([column, hue]).size().unstack().fillna(0)
        grouped_data.plot(kind='bar', stacked=False, color=['skyblue', 'lightgreen'], ax=axes[0])
        axes[0].set_title(f'Proportion of {column.capitalize()} and {hue.capitalize()} in Top 100 Box Office Movies - Matplotlib')
        axes[0].set_xlabel(column.capitalize())
        axes[0].set_ylabel('Number of Movies')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].legend(title=hue.capitalize())

        # 使用 Seaborn 创建分组条形图
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
    


