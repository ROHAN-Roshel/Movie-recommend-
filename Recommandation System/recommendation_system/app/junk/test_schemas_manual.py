



from pydantic import ValidationError

from app.api.schemas import (
    MovieResponse,
    RecommendationRequest,
    FeedbackRequest,
)


# Test 1: Valid recommendation request
request = RecommendationRequest(
    movie_id=19995,
    top_n=5,
)

print("Recommendation request:")
print(request.model_dump())


# Test 2: Valid movie response
movie = MovieResponse(
    id=19995,
    title="Avatar",
    overview="A marine travels to Pandora.",
    genres="Action,Adventure",
)

print("\nMovie response:")
print(movie.model_dump())


# Test 3: Valid feedback
feedback = FeedbackRequest(
    user_id=1,
    movie_id=19995,
    interaction_type="rating",
    rating=4.5,
)

print("\nFeedback request:")
print(feedback.model_dump())


# Test 4: Invalid feedback
try:
    FeedbackRequest(
        user_id=1,
        movie_id=19995,
        interaction_type="rating",
    )

except ValidationError:
    print(
        "\nInvalid feedback correctly rejected."
    )


print("\nAll schema checks completed.")