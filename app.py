import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors

# ------------------------
# Load data & models
# ------------------------

df = pd.read_csv('recommendation-data.csv')  # your processed CSV
scaler = joblib.load('scaler.pkl')

# KNN fit (you could also load from file)
feature_columns = [
    'category_encoded', 'offer_encoded', 'stock_status_encoded',
    'price', 'rating_encoded',
    'Keto', 'Gluten-Free', 'Low Sugar', 'Vegan', 'None',
    'Eggs', 'Milk', 'Nuts', 'Wheat', 'Soy','No-Allergens'
]
X = df[feature_columns]
X_scaled = scaler.transform(X)

knn = NearestNeighbors(n_neighbors=5)
knn.fit(X_scaled)

# ------------------------
# Streamlit UI
# ------------------------

st.title("🛒 SmartCart Product Recommender")
st.write("Get the best fit products for your needs directly from your local Walmart store.")

# ------------------------
# Inputs
# ------------------------

category_options = dict(zip(df['category'].unique(), df['category_encoded'].unique()))
category = st.selectbox("Select Category", list(category_options.keys()))
category_encoded = category_options[category]

price = st.slider("Price Range", float(df['price'].min()), float(df['price'].max()), step=0.5)

offer_map = {'10% off':1, 'Clearance':3, '$2 off':0, 'No-Offer':4, 'Buy 1 Get 1':2}
offer = st.selectbox("Offer", list(offer_map.keys()))
offer_encoded = offer_map[offer]

rating_map = {'Excellent':1, 'Very Good':4, 'Good':2, 'Average':0, 'Poor':3}
rating = st.selectbox("Customer Rating", list(rating_map.keys()))
rating_encoded = rating_map[rating]

stock_status = st.selectbox("Stock Status", ['In Stock'])
stock_encoded = 0  # only in stock

# Diet
diet_keto = st.checkbox("Keto")
diet_gf = st.checkbox("Gluten-Free")
diet_ls = st.checkbox("Low Sugar")
diet_vegan = st.checkbox("Vegan")
diet_none = st.checkbox("None")

# Allergens
no_allergens = st.checkbox("No Allergens")
if no_allergens:
    eggs = milk = nuts = wheat = soy = no_allergens_flag = 0
else:
    eggs = st.checkbox("Contains Eggs", value=False)*1
    milk = st.checkbox("Contains Milk", value=False)*1
    nuts = st.checkbox("Contains Nuts", value=False)*1
    wheat = st.checkbox("Contains Wheat", value=False)*1
    soy = st.checkbox("Contains Soy", value=False)*1
    no_allergens_flag = 0

# ------------------------
# Form query
# ------------------------

query = np.array([[
    category_encoded, offer_encoded, stock_encoded,
    price, rating_encoded,
    int(diet_keto), int(diet_gf), int(diet_ls), int(diet_vegan), int(diet_none),
    eggs, milk, nuts, wheat, soy, no_allergens_flag
]])

query_scaled = scaler.transform(query)

# ------------------------
# Recommendation
# ------------------------

if st.button("Recommend Products"):
    # Filter dataset first to same category & in-stock
    filtered_idx = df[(df['category_encoded']==category_encoded) & (df['stock_status_encoded']==0)].index
    X_filtered = X_scaled[filtered_idx]

    if len(X_filtered) < 1:
        st.warning("No products found in this category with In Stock status.")
    else:
        knn_partial = NearestNeighbors(n_neighbors=min(5, len(X_filtered)))
        knn_partial.fit(X_filtered)
        distances, indices = knn_partial.kneighbors(query_scaled)

        st.subheader("Top Recommended Products:")
        for idx, dist in zip(indices[0], distances[0]):
            real_idx = filtered_idx[idx]
            row = df.iloc[real_idx]
            st.markdown(f"✅ **{row['product_name']}** | Price: ${row['price']} | Offer: {row['offer']} | Rating: {row['customer_rating']}/5 | Category: {row['category']} (Distance: {dist:.2f})")

