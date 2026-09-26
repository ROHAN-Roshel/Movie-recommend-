




"""
frontend/streamlit_app.py

Streamlit frontend for the Movie Recommendation System.

Features:
- Search the TMDB movie catalog.
- Select a movie.
- Generate content-based recommendations.
- View recommended movie details.
- Submit likes, dislikes and ratings.
- View a user's recent feedback.

Run:
    streamlit run frontend/streamlit_app.py
"""

import os

import requests
import streamlit as st


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

REQUEST_TIMEOUT = 15


# --------------------------------------------------
# 2. API HELPER
# --------------------------------------------------

def api_request(
    method: str,
    endpoint: str,
    *,
    params: dict | None = None,
    payload: dict | None = None,
):
    """
    Send a request to FastAPI.

    Return:
        (response_data, error_message)
    """

    url = f"{API_BASE_URL}{endpoint}"

    try:
        response = requests.request(
            method=method,
            url=url,
            params=params,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        if response.status_code == 204:
            return None, None

        return response.json(), None

    except requests.exceptions.ConnectionError:
        return None, (
            "Cannot connect to FastAPI. "
            "Start the backend using: "
            "python -m uvicorn app.main:app --reload"
        )

    except requests.exceptions.Timeout:
        return None, (
            "The API request timed out. "
            "Please try again."
        )

    except requests.exceptions.HTTPError:
        try:
            detail = response.json().get(
                "detail",
                response.text,
            )
        except ValueError:
            detail = response.text

        return None, (
            f"API error {response.status_code}: "
            f"{detail}"
        )

    except requests.exceptions.RequestException as exc:
        return None, f"Request failed: {exc}"

    except ValueError:
        return None, "The API returned invalid JSON."


# --------------------------------------------------
# 3. API OPERATIONS
# --------------------------------------------------

def search_movies(keyword: str):
    return api_request(
        "GET",
        "/movies/search",
        params={
            "q": keyword,
            "limit": 30,
        },
    )


def get_movie(movie_id: int):
    return api_request(
        "GET",
        f"/movies/{movie_id}",
    )


def get_recommendations(
    movie_id: int,
    top_n: int,
):
    return api_request(
        "POST",
        "/recommendations",
        payload={
            "movie_id": movie_id,
            "top_n": top_n,
        },
    )


def submit_feedback(
    user_id: int,
    movie_id: int,
    interaction_type: str,
    rating: float | None = None,
):
    return api_request(
        "POST",
        "/feedback",
        payload={
            "user_id": user_id,
            "movie_id": movie_id,
            "interaction_type": interaction_type,
            "rating": rating,
        },
    )


def get_user_feedback(user_id: int):
    return api_request(
        "GET",
        f"/feedback/users/{user_id}",
        params={"limit": 20},
    )


# --------------------------------------------------
# 4. SESSION STATE
# --------------------------------------------------

if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None

if "recommendations" not in st.session_state:
    st.session_state.recommendations = []

if "search_results" not in st.session_state:
    st.session_state.search_results = []

if "last_search" not in st.session_state:
    st.session_state.last_search = ""


def select_movie(movie: dict):
    """
    Select a movie and clear recommendations
    from the previously selected movie.
    """

    st.session_state.selected_movie = movie
    st.session_state.recommendations = []


# --------------------------------------------------
# 5. SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.title("🎬 Movie Recommender")

    st.caption(
        "Content-based recommendations "
        "using TMDB movie data."
    )

    st.divider()

    user_id = st.number_input(
        "Your user ID",
        min_value=1,
        value=1,
        step=1,
        help=(
            "Temporary user ID for recording "
            "feedback. Authentication can "
            "be added later."
        ),
    )

    top_n = st.slider(
        "Number of recommendations",
        min_value=1,
        max_value=20,
        value=5,
    )

    st.divider()

    if st.button(
        "Check API connection",
        use_container_width=True,
    ):
        health, error = api_request(
            "GET",
            "/health",
        )

        if error:
            st.error(error)

        elif health.get("model_loaded"):
            st.success(
                "API and recommendation model "
                "are available."
            )

        else:
            st.warning(
                "API is running, but the "
                "recommendation model is unavailable."
            )

    st.caption(f"Backend: {API_BASE_URL}")


# --------------------------------------------------
# 6. PAGE HEADER
# --------------------------------------------------

st.title("🎬 Movie Recommendation System")

st.write(
    "Search for a movie and discover similar "
    "titles based on genres and descriptions."
)


# --------------------------------------------------
# 7. SEARCH MOVIES
# --------------------------------------------------

st.header("🔎 Find a movie")

with st.form("movie_search_form"):
    keyword = st.text_input(
        "Movie title",
        value=st.session_state.last_search,
        placeholder=(
            "Try Avatar, Interstellar, "
            "Titanic or Batman"
        ),
    )

    search_submitted = st.form_submit_button(
        "Search movies",
        type="primary",
    )


if search_submitted:
    keyword = keyword.strip()

    if not keyword:
        st.warning(
            "Please enter a movie title."
        )

    else:
        with st.spinner("Searching movies..."):
            results, error = search_movies(
                keyword
            )

        if error:
            st.error(error)
            st.session_state.search_results = []

        else:
            st.session_state.last_search = keyword
            st.session_state.search_results = results


search_results = st.session_state.search_results

if search_results:
    st.caption(
        f"Found {len(search_results)} results "
        "(maximum 30)."
    )

    movie_options = {
        (
            f"{movie['title']} "
            f"(TMDB ID: {movie['id']})"
        ): movie
        for movie in search_results
    }

    selected_label = st.selectbox(
        "Choose a movie",
        options=list(movie_options.keys()),
    )

    if st.button(
        "Select this movie",
        type="primary",
    ):
        movie = movie_options[selected_label]

        select_movie(movie)

        st.rerun()

elif st.session_state.last_search:
    st.info(
        "No movies found for the last search."
    )


# --------------------------------------------------
# 8. SELECTED MOVIE
# --------------------------------------------------

selected_movie = (
    st.session_state.selected_movie
)

if selected_movie is not None:
    st.divider()

    st.header("🎥 Selected movie")

    st.subheader(
        selected_movie["title"]
    )

    st.caption(
        f"TMDB ID: {selected_movie['id']}"
    )

    st.write(
        "**Genres:** "
        + (
            selected_movie.get("genres")
            or "Not available"
        )
    )

    st.write(
        selected_movie.get("overview")
        or "No description available."
    )

    if st.button(
        "✨ Get recommendations",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner(
            "Finding similar movies..."
        ):
            result, error = get_recommendations(
                movie_id=selected_movie["id"],
                top_n=top_n,
            )

        if error:
            st.error(error)

        else:
            st.session_state.recommendations = (
                result["recommendations"]
            )

            st.success(
                "Recommendations generated!"
            )


# --------------------------------------------------
# 9. DISPLAY RECOMMENDATIONS
# --------------------------------------------------

recommendations = (
    st.session_state.recommendations
)

if selected_movie is not None and recommendations:
    st.divider()

    st.header("🍿 Recommended movies")

    st.caption(
        "Similarity measures how closely "
        "the movies' text features match. "
        "It is not a user rating."
    )

    for rank, movie in enumerate(
        recommendations,
        start=1,
    ):
        with st.container(border=True):
            left, right = st.columns(
                [4, 1]
            )

            with left:
                st.subheader(
                    f"{rank}. {movie['title']}"
                )

                st.caption(
                    "Genres: "
                    + (
                        movie.get("genres")
                        or "Not available"
                    )
                )

                st.write(
                    movie.get("overview")
                    or "No description available."
                )

            with right:
                similarity = (
                    movie["similarity_score"]
                )

                st.metric(
                    "Similarity",
                    f"{similarity:.1%}",
                )

                st.progress(
                    min(
                        max(similarity, 0.0),
                        1.0,
                    )
                )

            # Each recommended movie gets
            # its own feedback controls.
            with st.expander(
                "👍 Rate or react to this movie"
            ):
                feedback_type = st.selectbox(
                    "Your feedback",
                    options=[
                        "like",
                        "dislike",
                        "rating",
                    ],
                    key=(
                        f"feedback_type_"
                        f"{movie['movie_id']}"
                    ),
                )

                rating = None

                if feedback_type == "rating":
                    rating = st.slider(
                        "Your rating",
                        min_value=1.0,
                        max_value=5.0,
                        value=4.0,
                        step=0.5,
                        key=(
                            f"rating_"
                            f"{movie['movie_id']}"
                        ),
                    )

                if st.button(
                    "Submit feedback",
                    key=(
                        f"submit_feedback_"
                        f"{movie['movie_id']}"
                    ),
                ):
                    result, error = submit_feedback(
                        user_id=int(user_id),
                        movie_id=movie["movie_id"],
                        interaction_type=feedback_type,
                        rating=rating,
                    )

                    if error:
                        st.error(error)

                    else:
                        st.success(
                            "Feedback saved!"
                        )


# --------------------------------------------------
# 10. FEEDBACK FOR SELECTED MOVIE
# --------------------------------------------------

if selected_movie is not None:
    st.divider()

    st.header("⭐ Rate your selected movie")

    with st.form("selected_movie_feedback"):
        interaction_type = st.radio(
            "Choose your feedback",
            options=[
                "like",
                "dislike",
                "rating",
            ],
            horizontal=True,
        )

        selected_rating = st.slider(
            "Rating (used only for rating feedback)",
            min_value=1.0,
            max_value=5.0,
            value=4.0,
            step=0.5,
        )

        feedback_submitted = (
            st.form_submit_button(
                "Save feedback"
            )
        )

    if feedback_submitted:
        rating_value = (
            selected_rating
            if interaction_type == "rating"
            else None
        )

        result, error = submit_feedback(
            user_id=int(user_id),
            movie_id=selected_movie["id"],
            interaction_type=interaction_type,
            rating=rating_value,
        )

        if error:
            st.error(error)

        else:
            st.success(
                "Your feedback was saved."
            )


# --------------------------------------------------
# 11. USER FEEDBACK HISTORY
# --------------------------------------------------

st.divider()

with st.expander(
    "📋 View my feedback history"
):
    if st.button(
        "Load feedback history"
    ):
        history, error = get_user_feedback(
            int(user_id)
        )

        if error:
            st.error(error)

        elif not history:
            st.info(
                "No feedback recorded for "
                "this user."
            )

        else:
            rows = [
                {
                    "Movie ID": item["movie_id"],
                    "Interaction": (
                        item["interaction_type"]
                    ),
                    "Rating": item["rating"],
                    "Created at": (
                        item["created_at"]
                    ),
                }
                for item in history
            ]

            st.dataframe(
                rows,
                use_container_width=True,
                hide_index=True,
            )


# --------------------------------------------------
# 12. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Movie Recommendation System | "
    "TMDB dataset | TF-IDF + Cosine Similarity"
)