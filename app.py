import streamlit as st
import joblib

# Load the trained SVM model and TF-IDF vectorizer
svm_classifier = joblib.load('svm_model.joblib')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Streamlit app code
def predict_genre(movie_plot_summary):
    # Convert the preprocessed movie data to TF-IDF vectors
    movie_tfidf = tfidf_vectorizer.transform([movie_plot_summary])

    # Make predictions using the SVM model
    predicted_genre = svm_classifier.predict(movie_tfidf)[0]

    return predicted_genre

def main():
    st.title('Movie Genre Classification')

    # Input: Text area for user to enter movie plot summary
    movie_plot_summary = st.text_area('Enter the movie plot summary here:', '')

    if st.button('Predict Genre'):
        if movie_plot_summary:
            # Predict the genre based on the input plot summary
            predicted_genre = predict_genre(movie_plot_summary)
            st.header(f'Predicted Genre: {predicted_genre}')
        else:
            st.warning('Please enter a movie plot summary.')

if __name__ == '__main__':
    main()
