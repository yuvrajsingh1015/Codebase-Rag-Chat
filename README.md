# Codebase-Rag-Chat

🚧 Work in progress — core functionality will be updates soon :)

//imrovements : ai cleint using requests


# local-coderag

A lightweight Retrieval-Augmented Generation (RAG) app for exploring and querying local codebases.

## Overview

This project indexes a local code directory, stores semantic embeddings using FAISS, and allows you to ask questions about the "python code" ??? through a simple Streamlit chat interface.

It is designed to run locally using Ollama(Gemma4:e2) by default, while also supporting OpenAI-compatible APIs through environment configuration.

## Features (Planned)

* Index local source code files
* Store embeddings in a FAISS vector index
* Retrieve relevant code snippets for a query
* Generate contextual answers using a chat model
* Streamlit-based chat UI
* Local-first setup with optional API backend

## Tech Stack

* Python 3.11+
* Streamlit
* FAISS
* OpenAI Python SDK (Ollama-compatible)/ gemma??
* NumPy

## Project Structure

```
.
├── app.py
├── main.py
├── prompt_flow.py
├── coderag/
├── tests/
```

## Setup (Coming Soon)

Instructions for installation and usage will be added as the project develops.

