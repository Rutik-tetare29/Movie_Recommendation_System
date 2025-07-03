import requests
import re

OMDB_API_KEY = "dc33d19"


def clean_title(title):
    # Remove text in parentheses (usually the year)
    return re.sub(r"\s*\(\d{4}\)", "", title).strip()

def get_movie_poster(title):
    try:
        cleaned_title = clean_title(title)

        # Fuzzy search
        search_url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&s={cleaned_title}"
        search_response = requests.get(search_url).json()

        if search_response.get("Response") == "True" and "Search" in search_response:
            imdb_id = search_response["Search"][0]["imdbID"]

            # Get full details
            details_url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&i={imdb_id}"
            details_response = requests.get(details_url).json()

            poster_url = details_response.get("Poster")
            if poster_url and poster_url != "N/A":
                return poster_url
    except Exception as e:
        print("Error fetching poster:", e)

    return None
