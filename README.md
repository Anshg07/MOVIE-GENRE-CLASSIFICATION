# Movie Genre Classification

This project aims to classify movie genres based on their plot summaries using machine learning techniques. The dataset used for this project is the IMDb Movie Dataset, which contains information about movies including their titles, genres, and plot summaries.

## Dataset

The IMDb Movie Dataset used in this project contains the following features:

- Title: The title of the movie.
- Genre: The genre(s) of the movie.
- Plot Summary: A brief summary of the plot of the movie.

## Methodology

1. **Data Preprocessing**: The dataset is cleaned and preprocessed to remove any irrelevant information and handle missing values.

2. **Text Processing**: The plot summaries are tokenized, normalized, and vectorized using techniques such as TF-IDF (Term Frequency-Inverse Document Frequency).

3. **Model Building**: Various machine learning algorithms such as Naive Bayes, Logistic Regression, and Random Forest are trained on the vectorized plot summaries to classify the movies into different genres.

4. **Model Evaluation**: The performance of each model is evaluated using metrics such as accuracy, precision, recall, and F1-score.

5. **Hyperparameter Tuning**: The hyperparameters of the best-performing model are fine-tuned to improve its performance further.

## Usage

To run the code:

1. Clone the repository:
   ```
   git clone https://github.com/Anshg07/MOVIE-GENRE-CLASSIFICATION.git
   ```

2. Navigate to the project directory:
   ```
   cd MOVIE-GENRE-CLASSIFICATION
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the Jupyter Notebook or Python script to train and evaluate the models.

## Results

The best-performing model achieves an accuracy of X% on the test dataset.

## Future Work

- Experiment with different text processing techniques such as word embeddings and deep learning models.
- Explore ensemble methods to further improve the classification performance.
- Enhance the dataset by incorporating additional features or external data sources.

## Contributors

- [Ansh Gupta](https://github.com/Anshg07)