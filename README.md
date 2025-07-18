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
```
----
## How It Works
- The user selects product category and price range using sliders and dropdowns.

- The system filters potential products using similarity measures.

- KNN identifies the top-N most similar items.

- Recommended products are displayed instantly on the UI.
----
 ## Installation & Usage
 **Clone the Repository**
```bash
git clone https://github.com/Pranav-Uniyal/Product-Recommendation-System.git
cd Product-Recommendation-System
```
 **Prerequisites**
Ensure the following Python packages are installed:
```
pip install pandas numpy scikit-learn streamlit
```
**To Run the Streamlit App:**
```
streamlit run app.py
```
----
## Data Description
walmart_instore_mock_dataset.csv: Simulated transactional/product-level data from a Walmart-like inventory.

recommendation-data.csv: Transformed data after encoding and scaling.

scaler.pkl: StandardScaler object saved for consistent input processing.

----
### 📌 Future Improvements
Integrate user authentication and preference history

Add collaborative filtering (matrix factorization)

Deploy on HuggingFace Spaces or Streamlit Cloud


## Author
# Pranav Uniyal
