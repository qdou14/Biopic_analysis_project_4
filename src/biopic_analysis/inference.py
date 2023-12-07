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

    def movie_gross_polynomial(self):
        """
        Perform Polynomial Regression to predict movie gross based on IMDB ratings.

        Returns:
        Displays a scatter plot with the original data and the polynomial regression line.
        """
        predictors = ["imdb_score"]
        target = ["box_office"]

        X = self.df[predictors]
        y = self.df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=0
        )
        self.y_test = y_test.values
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_poly, y_train)

        y_predict = model.predict(X_test_poly)

        mae = mean_absolute_error(y_test, y_predict)
        r2 = r2_score(y_test, y_predict)
        print("Polynomial Regression MAE:", mae)
        print("Polynomial Regression R-squared:", r2)

        X_values = X.values
        X_values_sorted = np.sort(X_values, axis=0)
        y_pred_sorted = model.predict(poly.transform(X_values_sorted))

        plt.scatter(X_values, y, label="Original data")

        plt.plot(
            X_values_sorted,
            y_pred_sorted,
            color="red",
            label="Polynomial regression line (degree=2)",
        )

        plt.xlabel("imdb_rating")
        plt.ylabel("imdb_gross")
        plt.title("Polynomial Regression (degree=2)")
        plt.legend()

        plt.gca().yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: format(int(x), ","))
        )
        self.y_predict = y_predict

        # Show plot
        plt.show()

    def plot_top_profitable_movies(df):
        df['profit'] = df['box_office'] - df['budget']
        top_20_movies = df.sort_values('profit', ascending=False).head(20)
        plt.figure(figsize=(12, 8))
        ax = sns.scatterplot(x='budget', y='profit', data=top_20_movies, s=100)
        sns.regplot(x='budget', y='profit', data=top_20_movies, scatter=False, ax=ax)
        texts = []
        for i, point in top_20_movies.iterrows():
            texts.append(plt.text(point['budget'], point['profit'], str(point['movie_title'])))
        adjust_text(texts)
        plt.title('Top 20 Profitable Movies', fontsize=20)
        plt.xlabel('Budget ($)', fontsize=14)
        plt.ylabel('Profit ($)', fontsize=14)
        # 检查是否存在图例，如果有则移除
        if ax.legend_:
            ax.legend_.remove()
        ax.set_facecolor('lightgrey')
        plt.show()

    


