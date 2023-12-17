# Medical Document Search Engine with TF-IDF
A Django-powered medical document search engine that utilizes the TF-IDF (Term Frequency-Inverse Document Frequency) approach for indexing and retrieval. It leverages the clinicaltrials/2021/trec-ct-2021 dataset from [ir-datasets](http://www.ir-datasets.com), boasting a collection of 376,000 medical documents.

## Website
http://nubengine.annavaws.com
## Features
TF-IDF based indexing: Efficiently index documents based on term frequency and document frequency to identify relevant documents for user queries.
Django backend: Robust and secure backend framework for managing document processing, indexing, and searching.
Tailwind CSS front-end: Modern and responsive web interface built with Tailwind CSS for a sleek and user-friendly experience.
Searchable by keywords and filters: Find relevant documents by entering keywords and refining results with additional filters.
Highlighting of matched keywords: Easily identify relevant terms within retrieved documents for quick scanning.
## Technologies
- Back-end: Python, Django
- Front-end: HTML, CSS, Tailwind
- Indexing: TF-IDF
- Dataset: clinicaltrials/2021/trec-ct-2021 (376,000 medical documents)
## Getting Started
1. Clone the repository: ```git clone https://github.com/daffafaizan/nubengine.git```
2. Set up a virtual environment and install dependencies: ```python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt```
3. Configure database settings in settings.py
4. Run database migrations: ```python manage.py migrate```
5. Run the development server: ```python manage.py runserver```
Visit ```http://localhost:8000``` in your browser and start searching!
## Contributing
We welcome contributions to improve and expand the features of this project. Feel free to fork the repository, make changes, and create pull requests!
