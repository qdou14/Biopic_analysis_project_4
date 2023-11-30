from src.biogic_analysis.data_summary import DataProcess
from src.biogic_analysis.exploratory_data_analysis import EDA
from src.biogic_analysis.inference import Inference


def main():
    """
    Run biogic analysis as a script.
    """
    print("------------------------------------------------")
    print("Biogic_Analysis")
    print("------------------------------------------------")

    data_summary()
    print("------------------------------------------------")
    exploratory_data_analysis()
    print("------------------------------------------------")
    inference()
    print("------------------------------------------------")


def data_summary():
    data_url = 'https://raw.githubusercontent.com/qdou14/biopic_analysis/main/dataset/biopics_dataset.csv'
    data_processor = DataProcess(data_url)
    df = data_processor.get_data()
    print(df.head())

    column_types = df.dtypes
    print(column_types)

    data_processor.convert_box_office()
    
    data_processor.update_dataframe_with_bechdel()
    
    data_processor.add_category()
    
    data_processor.get_data().head()

def exploratory_data_analysis():
    data_url = 'https://raw.githubusercontent.com/qdou14/biopic_analysis/main/dataset/biopics_dataset.csv'
    data_processor = DataProcess(data_url)
    data_processor.convert_box_office()
    data_processor.update_dataframe_with_bechdel()
    data_processor.add_category()
    df = data_processor.get_data()
    eda = EDA(df)
    while True:
            eda.unique_value_counts('subject_race')

            eda.combined_category_plot('subject_race', 'subject_sex')

            country_types_of_subject = eda.unique_value_counts('type_of_subject')

            eda.combined_category_plot('type_of_subject', 'category')

            country_bechdel_rating = eda.unique_value_counts('bechdel_rating')

            eda.combined_category_plot('bechdel_rating', 'category')

            eda.combined_category_plot('year_release', 'category')

            eda.column_boxplot('year_release', 'category')

            box_office_stats = eda.describe_stats('box_office')

            eda.column_boxplot("box_office",'category')


    def inferences():
        data_url = 'https://raw.githubusercontent.com/qdou14/biopic_analysis/main/dataset/biopics_dataset.csv'
        data_processor = DataProcess(data_url)
        data_processor.convert_box_office()
        data_processor.update_dataframe_with_bechdel()
        data_processor.add_category()
        df = data_processor.get_data()
        inference_analysis = Inference(df)
        while True:
            inference_analysis.bechdel_test_plot('type_of_subject', 'bechdel_rating')

            inference_analysis.proportion_in_top_box_office('subject_race', 'subject_sex')

            inference_analysis.box_office_diversity_correlation()

            inference_analysis.diversity_trends_over_time()


    main()