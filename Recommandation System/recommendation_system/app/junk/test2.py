from app.recommender.predict import MovieRecommender

recommender = MovieRecommender()
recommender.load_model()

movie_id = recommender.movie_ids[0]

results = recommender.recommend(
    movie_id=movie_id,
    top_n=5,
)

assert len(results) == 5

assert all(
    movie["movie_id"] != movie_id
    for movie in results
)

assert all(
    0 <= movie["similarity_score"] <= 1
    for movie in results
)

assert all(
    results[i]["similarity_score"]
    >= results[i + 1]["similarity_score"]
    for i in range(len(results) - 1)
)

print("All recommendation checks passed!")