---
layout: knowledge
title: Bag of Words
parent: Computer Vision
nav_order: 4
permalink: /knowledge-base/computer-vision/bag-of-words/
---

# Bag of Words

The Bag of Words (BoW) model is a popular technique in computer vision for image classification and retrieval. This section covers the implementation and applications of the BoW model in computer vision.

## Basic Concepts

### Feature Extraction

```python
import numpy as np
from sklearn.cluster import KMeans

def extract_features(images):
    # Extract SIFT features from images
    sift = cv2.SIFT_create()
    features = []
    
    for image in images:
        keypoints, descriptors = sift.detectAndCompute(image, None)
        if descriptors is not None:
            features.extend(descriptors)
    
    return np.array(features)
```

### Vocabulary Building

```python
def build_vocabulary(features, n_clusters=1000):
    # Cluster features to build vocabulary
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features)
    
    return kmeans
```

## Implementation

### Feature Quantization

```python
def quantize_features(descriptors, vocabulary):
    # Quantize features to visual words
    words = vocabulary.predict(descriptors)
    
    # Create histogram
    histogram = np.zeros(vocabulary.n_clusters)
    for word in words:
        histogram[word] += 1
    
    # Normalize histogram
    histogram = histogram / np.sum(histogram)
    
    return histogram
```

### Image Classification

```python
from sklearn.svm import SVC

def train_classifier(histograms, labels):
    # Train SVM classifier
    classifier = SVC(kernel='linear')
    classifier.fit(histograms, labels)
    
    return classifier

def classify_image(image, vocabulary, classifier):
    # Extract features
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    
    if descriptors is None:
        return None
    
    # Quantize features
    histogram = quantize_features(descriptors, vocabulary)
    
    # Predict class
    prediction = classifier.predict([histogram])[0]
    
    return prediction
```

## Applications

### Image Retrieval

```python
def compute_similarity(hist1, hist2):
    # Compute cosine similarity
    similarity = np.dot(hist1, hist2) / (np.linalg.norm(hist1) * np.linalg.norm(hist2))
    return similarity

def retrieve_similar_images(query_image, database_histograms, vocabulary, top_k=5):
    # Extract features from query image
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(query_image, None)
    
    if descriptors is None:
        return []
    
    # Quantize features
    query_hist = quantize_features(descriptors, vocabulary)
    
    # Compute similarities
    similarities = [compute_similarity(query_hist, hist) for hist in database_histograms]
    
    # Get top-k similar images
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    return top_indices
```

### Scene Recognition

```python
def recognize_scene(image, vocabulary, classifier, scene_categories):
    # Classify image
    prediction = classify_image(image, vocabulary, classifier)
    
    if prediction is None:
        return "Unknown"
    
    # Map prediction to scene category
    scene = scene_categories[prediction]
    
    return scene
```

## Additional Resources

- [Bag of Words Tutorial](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/csurka-eccv-04.pdf)
- [Feature Extraction Methods](https://www.robots.ox.ac.uk/~vgg/publications/2011/Chatfield11/chatfield11.pdf)
- [Image Classification with BoW](https://www.robots.ox.ac.uk/~vgg/publications/2010/everingham10/everingham10.pdf)