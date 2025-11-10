# Architecture

This document outlines the architecture of the Aspirational Recommendation System.

## Overview

The system is a Python-based application that uses a FastAPI server to expose a recommendation API. The recommendations are generated based on user conversations and discussion posts. The application is modularized for better maintainability and readability.

## Components

### 1. FastAPI Application (`src/main.py`)

The core of the system is a FastAPI application. It is responsible for:

*   Exposing the recommendation API.
*   Bringing together the different components of the application.

### 2. Core Components (`src/core/`)

*   **`config.py`**: This file handles configuration management, such as loading the API key.
*   **`models.py`**: This file contains the Pydantic models (`Post`, `UserConv`).
*   **`embed.py`**: This file is responsible for embedding text using the `all-MiniLM-L6-v2` sentence-transformer model.
*   **`aspirations.py`**: This file handles the aspiration extraction logic using the Gemini API. It classifies user messages into a list of aspirations.
*   **`indexing.py`**: This file manages the FAISS index, including building the index from the discussion posts and searching for similar items.
*   **`reranking.py`**: This module contains the logic for reranking and filtering the recommendations. It includes functions to boost posts based on aspiration alignment, penalize duplicates, and incorporate popularity and recency priors from `activity.json`.
*   **`metrics.py`**: This module contains functions to calculate the metrics we've defined: aspirational alignment and diversity.

### 3. Data

The system uses three data sources:

*   **`discussions.json`**: Contains the posts and comments from the online community. This is the content that is recommended to users.
*   **`conversations.json`**: Contains conversations between users and a chatbot. This is used to extract user aspirations.
*   **`activity.json`**: Contains user activity data, such as creating or commenting on posts. This is used for calculating popularity and recency priors for the reranking step.

### 4. Aspiration Extraction

User aspirations are extracted from conversations using the Gemini API. The system uses a prompt-based approach to generate a list of relevant tags that represent the user's key themes and aspirations. These tags are then used for semantic search.

### 5. Recommendation Engine

The recommendation engine is a multi-stage process:

1.  **Candidate Generation**: The system first generates a set of candidate recommendations by searching a FAISS index for posts with the highest cosine similarity to the user's aspirations.
2.  **Reranking and Filtering**: The initial recommendations are then reranked and filtered based on a number of factors, including:
    *   Aspiration alignment
    *   Popularity
    *   Recency
    *   Duplicate penalization

### 6. Docker

The application is containerized using Docker. This allows for easy setup and deployment. The `Dockerfile` defines the application's environment, and the `docker-compose.yml` file is used to build and run the application.

## Metrics

The following metrics are calculated and returned with each recommendation request:

*   **Aspirational Alignment**: The percentage of recommended posts that align with the user's aspirations.
*   **Diversity**: The diversity of the recommendations, calculated as the average intra-list cosine distance.
