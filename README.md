# 🎬 Movie Recommendation System

A sophisticated movie recommendation engine powered by collaborative filtering algorithms, featuring an intuitive Streamlit web interface with real-time movie poster integration.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-green.svg)

## 🚀 Features

- **User-Based Collaborative Filtering**: Get personalized movie recommendations based on users with similar taste preferences
- **Item-Based Collaborative Filtering**: Discover movies similar to ones you already love
- **Matrix Factorization (SVD & NMF)**: Advanced dimensionality reduction techniques for improved recommendation accuracy
- **Hybrid Recommendation System**: Combines multiple algorithms with customizable weights for superior performance
- **Interactive Web Interface**: Clean, responsive Streamlit UI with tabbed navigation and real-time interactions
- **Database Integration**: Persistent storage with SQLite for user data, ratings, and recommendation caching
- **A/B Testing Framework**: Statistical testing and comparison of different recommendation algorithms
- **Real-Time Movie Posters**: Automatic poster fetching via OMDB API integration
- **User Analytics**: Comprehensive user profiling and behavior tracking
- **Performance Metrics**: Advanced analytics including diversity scores, RMSE, and engagement rates
- **Scalable Architecture**: Modular codebase with separated concerns for easy maintenance and extension
- **Caching System**: Smart recommendation caching for improved performance

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
├── app.py                      # Original Streamlit application
├── enhanced_app.py             # Enhanced app with all new features
├── demo_enhanced_features.py   # Demo script for testing features
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── notebook.ipynb             # Jupyter notebook for analysis
├── data/                       # Dataset directory
│   ├── movies.csv             # Movie metadata
│   ├── ratings.csv            # User ratings data
│   └── movie_recommender.db   # SQLite database (auto-generated)
└── src/                        # Source code modules
    ├── __init__.py             # Package initialization
    ├── preprocess.py           # Data loading and preprocessing
    ├── similarity.py           # Similarity computation algorithms
    ├── recommend.py            # Basic recommendation logic
    ├── posters.py              # Movie poster fetching utilities
    ├── matrix_factorization.py # SVD & NMF implementations
    ├── database.py             # Database management system
    ├── hybrid_recommender.py   # Hybrid recommendation engine
    └── ab_testing.py           # A/B testing framework
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

### Running the Enhanced Application

**Basic Application:**
```bash
streamlit run app.py
```

**Enhanced Application (with all new features):**
```bash
streamlit run enhanced_app.py
```

**Demo Script (test all features):**
```bash
python demo_enhanced_features.py
```

The enhanced application includes:
- 🔢 Matrix Factorization (SVD & NMF)
- 🔀 Hybrid Recommendation System
- 💾 Database Integration
- 🧪 A/B Testing Framework
- 📊 Advanced Analytics & Metrics

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

## 🎉 Recently Implemented Features

- [x] **Matrix Factorization techniques (SVD, NMF)** - Advanced dimensionality reduction for improved recommendations
- [x] **Database integration for persistent storage** - SQLite database with comprehensive data management
- [x] **Hybrid recommendation systems** - Combines multiple algorithms for superior accuracy
- [x] **A/B testing framework for recommendation quality** - Statistical testing and performance comparison
- [x] **Real-time user feedback integration** - Interactive rating and feedback collection
- [x] **Performance metrics and analytics** - Comprehensive system monitoring and reporting

## 📋 Future Enhancements

- [ ] Deep Learning approaches (Neural Collaborative Filtering)
- [ ] User authentication and personalized profiles
- [ ] Advanced filtering options (genre, year, rating thresholds)
- [ ] Real-time recommendation updates
- [ ] Social features and collaborative playlists
- [ ] Mobile app development
- [ ] Cloud deployment and scaling
- [ ] Machine Learning pipeline automation

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