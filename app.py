from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Load the data
movies_file_path = 'Resources/tmdb_5000_movies.csv'
df_movies = pd.read_csv(movies_file_path)

# Preprocess the data and define the recommendation function
def get_recommendations(title):
    # Preprocess the overview text
    tfidf = TfidfVectorizer(stop_words="english")
    df_movies["overview"] = df_movies["overview"].fillna("")
    tfidf_matrix = tfidf.fit_transform(df_movies["overview"])

    # Compute cosine similarity
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Build a mapping of movie titles to their corresponding indices
    indices = pd.Series(df_movies.index, index=df_movies['original_title']).drop_duplicates()

    # Get the index of the movie title
    idx = indices[title]

    # Compute similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top 10 most similar movies
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]

    # Return the titles of the top 10 most similar movies
    return df_movies['original_title'].iloc[movie_indices].tolist()

# Define route for the homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input from the form
        movie_title = request.form['movie_title']

        # Generate movie recommendations
        recommended_movies = get_recommendations(movie_title)

        # Pass the recommendations to the template
        return render_template('index.html', recommended_movies=recommended_movies)
    else:
        # Render the homepage template
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)










