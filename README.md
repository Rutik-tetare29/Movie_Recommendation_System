# 🎬 Movie Recommendation System

A sophisticated movie recommendation engine powered by collaborative filtering algorithms, featuring an intuitive Streamlit web interface with real-time movie poster integration.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-green.svg)

## 🚀 Features

- **User-Based Collaborative Filtering**: Get personalized movie recommendations based on users with similar taste preferences
- **Item-Based Collaborative Filtering**: Discover movies similar to ones you already love
- **Interactive Web Interface**: Clean, responsive Streamlit UI with tabbed navigation
- **Real-Time Movie Posters**: Automatic poster fetching via OMDB API integration
- **Scalable Architecture**: Modular codebase with separated concerns for easy maintenance and extension
- **Cosine Similarity**: Advanced similarity computation using scikit-learn for accurate recommendations

## 🎯 Demo

The application provides two main recommendation approaches:

1. **👤 User-Based Recommendations**: 
   - Select a user ID from the dropdown
   - Adjust the number of recommendations (1-20)
   - Get personalized movie suggestions based on similar users' preferences

2. **🎞 Item-Based Recommendations**:
   - Choose a movie you enjoyed
   - Set the number of similar movies to discover
   - Find movies with similar characteristics and ratings patterns

## 🛠 Tech Stack

- **Backend**: Python 3.8+
- **Web Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: scikit-learn (Cosine Similarity)
- **API Integration**: OMDB API for movie posters
- **Data**: MovieLens dataset (ratings.csv, movies.csv)

## 📁 Project Structure

```
movie_recommender/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── notebook.ipynb       # Jupyter notebook for analysis
├── data/                 # Dataset directory
│   ├── movies.csv        # Movie metadata
│   └── ratings.csv       # User ratings data
└── src/                  # Source code modules
    ├── __init__.py       # Package initialization
    ├── preprocess.py     # Data loading and preprocessing
    ├── similarity.py     # Similarity computation algorithms
    ├── recommend.py      # Recommendation logic
    └── posters.py        # Movie poster fetching utilities
```

## 🚦 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd movie_recommender
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the data**
   - Ensure `movies.csv` and `ratings.csv` are in the `data/` directory
   - The project expects MovieLens dataset format

### Running the Application

```bash
streamlit run app.py
```

The application will launch in your default browser at `http://localhost:8501`

## 📊 Algorithm Details

### User-Based Collaborative Filtering

1. **User-Item Matrix Creation**: Builds a sparse matrix with users as rows and movies as columns
2. **Similarity Computation**: Calculates cosine similarity between user rating vectors
3. **Recommendation Generation**: 
   - Finds users most similar to the target user
   - Weights their ratings by similarity scores
   - Recommends unrated movies with highest weighted scores

### Item-Based Collaborative Filtering

1. **Item Similarity Matrix**: Computes cosine similarity between movie rating patterns
2. **Similar Item Discovery**: Identifies movies with similar user rating distributions
3. **Recommendation Ranking**: Returns top-N most similar movies to the selected title

### Cosine Similarity Formula

```
similarity(A,B) = cos(θ) = (A·B) / (||A|| × ||B||)
```

Where A and B are rating vectors for users or items.

## 🔧 Configuration

### API Keys
- Update `OMDB_API_KEY` in `src/posters.py` with your own OMDB API key
- Get a free key at: [http://www.omdbapi.com/apikey.aspx](http://www.omdbapi.com/apikey.aspx)

### Data Format
The system expects CSV files with the following structure:

**movies.csv**:
```csv
movieId,title,genres
1,"Toy Story (1995)",Adventure|Animation|Children|Comedy|Fantasy
```

**ratings.csv**:
```csv
userId,movieId,rating,timestamp
1,1,4.0,964982703
```

## 🧪 Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Structure

- **`preprocess.py`**: Data loading and user-item matrix creation
- **`similarity.py`**: Cosine similarity computation for users and items
- **`recommend.py`**: Core recommendation algorithms
- **`posters.py`**: OMDB API integration for movie poster retrieval
- **`app.py`**: Streamlit web application interface

## 📈 Performance Considerations

- **Memory Usage**: Large datasets may require sparse matrix implementations
- **Computation Time**: Similarity matrices are computed once at startup
- **API Rate Limits**: OMDB API has usage limits; consider caching poster URLs
- **Scalability**: For production use, consider implementing incremental similarity updates

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📋 Future Enhancements

- [ ] Matrix Factorization techniques (SVD, NMF)
- [ ] Deep Learning approaches (Neural Collaborative Filtering)
- [ ] Hybrid recommendation systems
- [ ] Real-time user feedback integration
- [ ] A/B testing framework for recommendation quality
- [ ] Database integration for persistent storage
- [ ] User authentication and personalized profiles
- [ ] Advanced filtering options (genre, year, rating thresholds)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MovieLens Dataset](https://grouplens.org/datasets/movielens/) by GroupLens Research
- [OMDB API](http://www.omdbapi.com/) for movie poster data
- [Streamlit](https://streamlit.io/) for the amazing web framework
- [scikit-learn](https://scikit-learn.org/) for machine learning utilities

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please reach out:

- **Project Maintainer**: [Rutik Tetare]
- **Email**: [rutiktetare@gmail.com]
- **LinkedIn**: [https://www.linkedin.com/in/rutik-tetare-3154b3281/]

---

**⭐ If you found this project helpful, please consider giving it a star!**