# Aspirational Recommendation System

This project is a Python-based system that uses historical conversations and interaction logs to develop a recommendation engine to suggest posts from an online community. The goal is to deliver content recommendations that are aligned with who a user wants to become, not just what is entertaining or interesting to them right now.

## Architecture

For a detailed explanation of the architecture, please see the [architecture.md](architecture.md) file.

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