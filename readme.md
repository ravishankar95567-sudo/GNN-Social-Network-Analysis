# Graph Neural Networks for Social Network Analysis

## Project Description

Graph Neural Networks (GNNs) are deep learning models designed to process graph-structured data where entities are interconnected. Unlike traditional machine learning models that treat data points independently, GNNs learn by considering both node features and the relationships between connected nodes. This makes them highly effective for applications such as social network analysis, recommendation systems, fraud detection, and knowledge graphs.

In this project, a Graph Convolutional Network (GCN), which is a type of Graph Neural Network, is implemented for social network analysis using the Cora dataset. The dataset is represented as a graph in which nodes represent entities and edges represent relationships between them. The model learns from the graph structure and predicts the class label of each node by aggregating information from its neighboring nodes.

The primary task implemented in this project is node classification. Given a node in the graph, the model analyzes its features and connections, then predicts the category to which the node belongs. To make the project interactive and user-friendly, a simple web interface is developed using Flask, HTML, and CSS, allowing users to enter a node ID and obtain prediction results through a browser.

This project demonstrates how Graph Neural Networks can effectively learn relational patterns in graph-structured data and perform intelligent predictions in social network analysis.

## Technologies Used
- Python
- PyTorch Geometric
- Flask
- HTML
- CSS

## Features
- Node Classification
- Social Network Analysis
- Web Interface for Prediction

## Project Structure
- app.py : Flask backend
- templates/ : HTML pages
- requirements.txt : Dependencies
- README.md : Project documentation

## Dataset
Cora Dataset

## Author
Ravi Shankar
