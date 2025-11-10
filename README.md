# Aspirational Recommendation System

This project is a Python-based system that uses historical conversations and interaction logs to develop a recommendation engine to suggest posts from an online community. The goal is to deliver content recommendations that are aligned with who a user wants to become, not just what is entertaining or interesting to them right now.

## Architecture

The system is a multi-stage recommendation engine that uses a combination of semantic search and rule-based reranking to provide recommendations.

1.  **Aspiration Modeling**: User aspirations are extracted from conversations using the Gemini API. The system uses a prompt-based approach to classify the user's messages into predefined aspirational categories.

2.  **Candidate Generation**: The system first generates a set of candidate recommendations by searching a FAISS index for posts with the highest cosine similarity to the user's aspirations.

3.  **Reranking and Filtering**: The initial recommendations are then reranked and filtered based on a number of factors, including:
    *   Aspiration alignment
    *   Popularity
    *   Recency
    *   Duplicate penalization

## Key Metrics

The following metrics are calculated and returned with each recommendation request:

*   **Aspirational Alignment**: The percentage of recommended posts that align with the user's aspirations.
*   **Diversity**: The diversity of the recommendations, calculated as the average intra-list cosine distance.

## Setup

To set up the project, you will need to have Docker and Docker Compose installed.

1.  Clone the repository.
2.  Create a `.env` file in the root of the project and add your `GOOGLE_API_KEY`. See the `.env.example` file for an example.
3.  Unzip the `data.zip` file to get the `data` directory.
4.  Run `docker-compose up` to build and start the application.

The API will be available at `http://localhost:8000`.

## Running Tests

To run the tests, you can run the following command:

```bash
docker-compose exec web pytest
```
