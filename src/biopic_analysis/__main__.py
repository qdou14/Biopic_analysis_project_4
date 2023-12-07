from .data_summary import DataProcess
from .exploratory_data_analysis import EDA
from .inference import Inference
from .building_model import ModelBuildin


def main():
    """
    Run biopic analysis as a script.
    """
    print("------------------------------------------------")
    print("Biopic_Analysis")
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
    df = data_processor.merge_movie_data('https://raw.githubusercontent.com/qdou14/Biopic_analysis_project_4/main/movie_data.csv')
    df.head()

    column_types = df.dtypes
    column_types

    data_processor.convert_box_office()

    # Add a 'category' column.
    data_processor.add_category()
    data_processor.get_data().head()

    data_processor.check_data()

def exploratory_data_analysis():
    data_url = 'https://raw.githubusercontent.com/qdou14/biopic_analysis/main/dataset/biopics_dataset.csv'
    data_processor = DataProcess(data_url)
    df = data_processor.merge_movie_data('https://raw.githubusercontent.com/qdou14/Biopic_analysis_project_4/main/movie_data.csv')
    data_processor.convert_box_office()
    df=data_processor.add_category()
    eda = EDA(df)
    
    data_processor.describe_categorical()
    while True:
            eda.unique_value_counts('subject_race')

            eda.combined_category_plot('subject_race', 'subject_sex')

            country_types_of_subject = eda.unique_value_counts('type_of_subject')

            eda.combined_category_plot('type_of_subject', 'category')

            eda.combined_category_plot('year_release', 'category')

            box_office_stats = eda.describe_stats('box_office')

            eda.column_boxplot("box_office",'category')


    def inferences():
        data_url = 'https://raw.githubusercontent.com/qdou14/biopic_analysis/main/dataset/biopics_dataset.csv'
        data_processor = DataProcess(data_url)
        df = data_processor.merge_movie_data('https://raw.githubusercontent.com/qdou14/Biopic_analysis_project_4/main/movie_data.csv')
        data_processor.convert_box_office()
        df=data_processor.add_category()
        inference_analysis = Inference(df)
        while True:

            inference_analysis.proportion_in_top_box_office('subject_race', 'subject_sex')

            inference_analysis.box_office_diversity_correlation()

    main()
