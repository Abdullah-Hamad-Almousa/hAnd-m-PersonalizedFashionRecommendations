"""
H&M Recommendation System — Final Model
Uses the exported ONNX model to recommend articles for a customer.

How to use:
1. Run: conda activate Heavy
2. Run: python main.py
3. Change the CUSTOMER_ID below to test different customers
4. The script shows top 12 recommended articles with their images
"""

import os
import json
import numpy as np
import pandas as pd
import onnxruntime as ort
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ============================================================
#  INPUT — change this to test different customers
# ============================================================
CUSTOMER_ID = None  # Set to a specific customer_id string, or None to pick a random one

# Paths
MODEL_PATH = r"A:\IDEs\PyCharm\hAnd-m-PersonalizedFashionRecommendations\H&MRecommendationSystemModel.onnx"
METADATA_PATH = r"A:\IDEs\PyCharm\hAnd-m-PersonalizedFashionRecommendations\H&MRecommendationSystemModel_metadata.json"
DATA_DIR = r"A:\IDEs\PyCharm\hAnd-m-PersonalizedFashionRecommendations\H&M ready dataset"
IMAGE_DIR = r"A:\IDEs\PyCharm\hAnd-m-PersonalizedFashionRecommendations\dataset\images"

# Number of recommendations
TOP_K = 12


def load_model():
    """Load the ONNX model and metadata."""
    sess = ort.InferenceSession(MODEL_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    print(f"Model loaded: {metadata['model_type']}")
    print(f"Training accuracy: {metadata['accuracy']:.4f}")
    print(f"Features: {metadata['n_features']}")
    return sess, metadata


def load_data():
    """Load articles and customers from the H&M ready dataset."""
    articles = pd.read_parquet(os.path.join(DATA_DIR, "articles.parquet"))
    customers = pd.read_parquet(os.path.join(DATA_DIR, "customers.parquet"))
    transactions = pd.read_parquet(os.path.join(DATA_DIR, "transactions.parquet"))

    articles["article_id"] = articles["article_id"].astype(str)
    customers["customer_id"] = customers["customer_id"].astype(str)
    transactions["article_id"] = transactions["article_id"].astype(str)
    transactions["customer_id"] = transactions["customer_id"].astype(str)

    return articles, customers, transactions


def get_customer(customer_id, customers, transactions):
    """Get a customer by ID, or pick a random one with purchase history."""
    if customer_id is None:
        # Pick a random customer who has purchases
        active_customers = transactions["customer_id"].unique()
        customer_id = np.random.choice(active_customers)
        print(f"No customer_id given — picked random: {customer_id}")

    if customer_id not in customers["customer_id"].values:
        print(f"ERROR: customer_id {customer_id} not found in customers dataset")
        return None

    customer = customers[customers["customer_id"] == customer_id].iloc[0]
    print(f"\nCustomer: {customer_id}")
    print(f"  Age: {customer['age']}")
    print(f"  Club status: {customer['club_member_status']}")
    print(f"  Total purchases: {customer['customer_total_purchases']}")

    # Show what they bought before
    history = transactions[transactions["customer_id"] == customer_id]
    if len(history) > 0:
        print(f"  Previous purchases: {len(history)} items")

    return customer_id, customer


def build_features(customer, articles, metadata):
    """Build feature vectors for this customer paired with every article."""
    feature_names = metadata["feature_names"]

    # Create a dataframe: one row per article, all paired with this customer
    rows = []
    for _, article in articles.iterrows():
        row = {}
        for feat in feature_names:
            if feat in customer.index:
                row[feat] = customer[feat]
            elif feat in article.index:
                row[feat] = article[feat]
            elif feat == "month":
                row[feat] = 6  # default to June
            elif feat == "is_weekend":
                row[feat] = 1  # default to weekend
            else:
                row[feat] = 0
        rows.append(row)

    feat_df = pd.DataFrame(rows)

    # Encode categorical columns (same as training)
    from sklearn.preprocessing import LabelEncoder

    for col in feat_df.columns:
        if feat_df[col].dtype == "object" or str(feat_df[col].dtype).startswith("category"):
            le = LabelEncoder()
            feat_df[col] = le.fit_transform(feat_df[col].astype(str))

    # Fill any NaN
    for col in feat_df.columns:
        feat_df[col] = feat_df[col].fillna(0)

    return feat_df.values.astype(np.float32)


def get_recommendations(sess, customer, articles, metadata, top_k=12):
    """Score all articles and return the top K recommendations."""
    print(f"\nScoring {len(articles)} articles...")

    features = build_features(customer, articles, metadata)

    input_name = sess.get_inputs()[0].name
    preds = sess.run(None, {input_name: features})[0]

    # Handle different output formats
    if len(preds.shape) == 2 and preds.shape[1] == 2:
        scores = preds[:, 1]
    elif len(preds.shape) == 2 and preds.shape[1] == 1:
        scores = preds.flatten()
    else:
        scores = preds.flatten()

    # Get top K article indices
    top_indices = np.argsort(scores)[::-1][:top_k]

    recommendations = []
    for idx in top_indices:
        article = articles.iloc[idx]
        recommendations.append({
            "article_id": article["article_id"],
            "product_type": article.get("product_type_name", "N/A"),
            "product_group": article.get("product_group_name", "N/A"),
            "colour": article.get("colour_group_name", "N/A"),
            "department": article.get("department_name", "N/A"),
            "score": float(scores[idx]),
        })

    return recommendations


def get_image_path(article_id):
    """Find the image file for an article."""
    prefix = article_id[:3]
    img_path = os.path.join(IMAGE_DIR, prefix, f"{article_id}.jpg")
    if os.path.exists(img_path):
        return img_path
    return None


def show_recommendations(recommendations):
    """Display recommendations as a table and image grid."""
    # Print text table
    print(f"\n{'='*80}")
    print(f"  TOP {len(recommendations)} RECOMMENDATIONS")
    print(f"{'='*80}")
    print(f"{'#':<4} {'Article ID':<15} {'Type':<20} {'Colour':<15} {'Score':<10}")
    print(f"{'-'*80}")
    for i, rec in enumerate(recommendations):
        print(f"{i+1:<4} {rec['article_id']:<15} {rec['product_type']:<20} {rec['colour']:<15} {rec['score']:<10.4f}")

    # Show image grid (4 columns x 3 rows)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle("H&M Recommendations — Top 12 Articles", fontsize=16, fontweight="bold")

    for i, rec in enumerate(recommendations):
        row = i // 4
        col = i % 4
        ax = axes[row][col]

        img_path = get_image_path(rec["article_id"])
        if img_path:
            img = mpimg.imread(img_path)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "No Image", ha="center", va="center", fontsize=14, color="gray")
            ax.set_facecolor("#f5f5f7")

        ax.set_title(f"#{i+1} {rec['product_type']}\n{rec['colour']}\nScore: {rec['score']:.3f}",
                     fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(r"A:\IDEs\PyCharm\hAnd-m-PersonalizedFashionRecommendations\recommendations_output.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nImage saved to: recommendations_output.png")


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  H&M Recommendation System")
    print("=" * 60)

    # Load model and data
    sess, metadata = load_model()
    articles, customers, transactions = load_data()

    # Get customer
    customer_id, customer = get_customer(CUSTOMER_ID, customers, transactions)
    if customer is None:
        exit()

    # Get recommendations
    recommendations = get_recommendations(sess, customer, articles, metadata, top_k=TOP_K)

    # Show results
    show_recommendations(recommendations)

    print("\nDone!")