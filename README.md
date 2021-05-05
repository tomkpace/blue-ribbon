# BLUE-RIBBON

# Introduction
This project was a first attempt to create and fuse a knowledge graph representation of a movie data set composed of reviews, plot summaries and tabular information such as the actors involved, who directed the films, etc.  This was originally part of a project called Probabilistic Active Bayesian Search Tool (PABST), hence Blue-Ribbon.  This work was completed with [mkcyoung](https://github.com/mkcyoung).  

## Installation
Clone this repo.
Create a Python 3.7.3 virtual environment.
Install the dependencies.
```bash
pip install -r requirments.txt
```
Install the required models.
```bash
python -m spacy download en_core_web_sm
```
Enjoy!

## Notebooks
This repo contains a few notebooks to illustrate the method.  For example, the relation.ipynb notebook illustrates the knowledge graph extraction method that generates knowledgeable triples from the textual data.  The fusion.ipynb notebook illustrates the knowledge graph fusion method.
