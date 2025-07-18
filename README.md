# 🛒 Product-Recommendation-System

A **machine learning-based product recommendation system** built using **K-Nearest Neighbors (KNN)** to help users discover relevant products from Walmart's in-store dataset. This system is deployed using **Streamlit** for a simple and interactive user experience.

---

##  Project Overview

This project utilizes a mock Walmart dataset and applies the **KNN algorithm** to provide personalized product recommendations based on customer preferences such as category, price range, and product metadata.

### 🔍 Key Features

-  KNN-based recommendation engine
-  Input-based dynamic filtering using:
  - Category
  - Price
-  Data preprocessing and feature scaling
-  Real-time recommendations via **Streamlit** web app
-  Scalable design for larger real-world datasets

---

## 📁 Project Structure

```bash
├── app.py                             # Streamlit frontend for the app
├── walmart_recommendation.ipynb      # Jupyter notebook for analysis & model development
├── scaler.pkl                        # Pre-fitted scaler used for prediction
├── recommendation-data.csv           # Encoded and transformed dataset used for recommendations
├── walmart_instore_mock_dataset.csv  # Raw mock dataset
├── README.md                         # Project documentation
