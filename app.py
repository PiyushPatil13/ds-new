from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load your model data
with open("artifacts/model_data.pkl", "rb") as f:
    data = pickle.load(f)
df = data["df"]
cos_sim = data["cos_sim"]

@app.route('/')
def index():
    # Flask looks for 'index.html' inside the 'templates' folder automatically
    return render_template('index.html')

@app.route('/get_movie_list', methods=['GET'])
def get_movie_list():
    # Return all movie titles as a list for the autocomplete feature
    titles = df['title'].tolist()
    return jsonify(titles)

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.json.get('movie')
    # Use case-insensitive search
    if movie_title.lower() not in df['title'].str.lower().values:
        return jsonify({"error": "Movie not found"}), 404

    idx = df[df['title'].str.lower() == movie_title.lower()].index[0]
    distances = sorted(list(enumerate(cos_sim[idx])), reverse=True, key=lambda x: x[1])
    
    recommendations = [df.iloc[i[0]].title for i in distances[1:6]]
    return jsonify({"recommendations": recommendations})

if __name__ == '__main__':
    app.run(debug=True)