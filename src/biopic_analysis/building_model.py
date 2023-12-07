from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ModelBuilding:
    """
    A class designed for building and evaluating predictive models focused on the relationship 
    between various film attributes and their success, measured by box office revenue.

    This class offers methods for preprocessing film data, generating visualizations like 
    pairplots and heatmaps for exploratory data analysis, and building classification and 
    regression models to predict film success.

    Methods:
    preprocess(df): Preprocesses the given DataFrame by handling missing values and selecting relevant columns.
    pairplot(df): Generates and saves a pairplot for visual exploratory analysis.
    heatmap(df): Creates a heatmap to visualize correlations among numerical features.
    build_model(df): Constructs and evaluates a RandomForestClassifier model for predicting film success.
    fit_and_plot(df): Fits Linear and Polynomial Regression models and plots the results to analyze the relationship 
                      between film budget and box office earnings.
    """
    def preprocess(self, df):
        """
        Preprocesses the given DataFrame by dropping rows with missing values in 'budget' and 'box_office',
        selecting relevant columns, converting the 'imdb_score' to integer, and creating a 'success' column.

        Parameters:
        df (pd.DataFrame): The DataFrame to preprocess.

        Returns:
        pd.DataFrame: The preprocessed DataFrame.
        """
        df = df.dropna(subset=['budget', 'box_office'])
        columns_to_keep = ['movie_title', 'country', 'year_release', 'box_office', 
                           'type_of_subject', 'person_of_color', 'subject_sex', 
                           'budget', 'imdb_score']
        df = df[columns_to_keep]
        df['imdb_score'] = df['imdb_score'].round().astype(int)
        df['success'] = df['box_office'].apply(lambda x: 1 if x > 10000000 else 0)
        return df

    def pairplot(self, df):
        """
        Generates a pairplot of the given DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame to visualize.

        This method saves the pairplot as 'pairplot.png' and does not return anything.
        """
        sns.pairplot(df, height=1.2, aspect=1.25)
        plt.savefig('pairplot.png')

    def heatmap(self, df):
        """
        Generates a heatmap of the correlation matrix for numeric columns in the DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame to visualize.

        This method displays the heatmap and does not return anything.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
        plt.show()

    def build_model(self, df):
        """
        Builds and evaluates a RandomForestClassifier model. The method splits the data into training and 
        testing sets, preprocesses it, and fits the model.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data to model.

        Returns:
        sklearn.pipeline.Pipeline: The fitted model pipeline.
        """
        X = df[['person_of_color', 'subject_sex', 'imdb_score', 'budget']]
        y = df['success']

        categorical_features = ['subject_sex']
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        numeric_features = ['imdb_score', 'budget']
        numeric_transformer = SimpleImputer(strategy='median')

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])

        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', RandomForestClassifier(random_state=42))])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)

        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        print('CV accuracy scores:', cv_scores)
        print('CV accuracy mean:', cv_scores.mean())

        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))

        return model

    def fit_and_plot(self, df):
        """
        Fits a Linear Regression and a Polynomial Regression model to the 'budget' and 'box_office' columns
        of the DataFrame. It plots the results and calculates R-squared values for each model.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data for regression analysis.

        Returns:
        dict: A dictionary containing R-squared values for Linear and Polynomial Regression.
        """
        budget = df['budget'].values.reshape(-1, 1)  # predictor
        box_office = df['box_office'].values  # target

        X_train, X_test, y_train, y_test = train_test_split(budget, box_office, test_size=0.2, random_state=42)

        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)

        poly_features = PolynomialFeatures(degree=4)
        X_train_poly = poly_features.fit_transform(X_train)
        X_test_poly = poly_features.transform(X_test)

        poly_model = LinearRegression()
        poly_model.fit(X_train_poly, y_train)

        plt.scatter(budget, box_office, color='blue', label='Original data')
        plt.plot(X_test, linear_model.predict(X_test), color='green', label='Linear regression line')

        X_fit = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)
        X_fit_poly = poly_features.transform(X_fit)
        plt.plot(X_fit, poly_model.predict(X_fit_poly), color='red', label='Polynomial regression line')

        plt.title('Budget vs Box Office')
        plt.xlabel('Budget')
        plt.ylabel('Box Office')
        plt.legend()
        plt.show()

        r_squared_linear = r2_score(y_test, linear_model.predict(X_test))
        r_squared_poly = r2_score(y_test, poly_model.predict(X_test_poly))

        return {'Linear Regression R-squared': r_squared_linear, 'Polynomial Regression R-squared': r_squared_poly}

    
    




