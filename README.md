# RAG Application for PubMed Abstracts

## Overview

This project is a __Retrieval-Augmented Generation (RAG)__ application designed to retrieve and process scientific data from PubMed. It integrates a Weaviate vector database and a FastAPI backend to enable the search and retrieval of relevant articles. Users can query article abstracts stored in the Weaviate database and generate responses based on those abstracts and extracts, providing a rich, data-driven interaction.


## Installation

1. __Clone the repository:__

    ```
    git clone https://github.com/djgregry/isp-rag-application.git
    cd isp-rag-application
    ```

2. __Run the entire application using Docker:__

    ```
    docker compose up -d
    ```


## Running the Project

The entire application, including the __frontend, backend and Weaviate database__, runs in Docker containers via `compose.yaml`

- __Using the frontend interface:__ Open `http://127.0.0.1:5500/` (or the configured frontend port) in a browser.

- __Backend (FastAPI):__ Interacts with Weaviate instance in Docker for vector loading and search.
    - Accessible on port `8000`
    - Environment variables are stored in `backend/.env`

- __Weaviate Vector Database:__ Stores vectorized data and provides vector search capabilities to the application.
    - Accessible on port `8080`


## Configuration

- Modify `compose.yaml` to update container settings.

- Edit `.env` file by setting environment variables.
