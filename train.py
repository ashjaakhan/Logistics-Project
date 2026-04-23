import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# ==============================
# LOAD DATA
# ==============================
def load_datasets():
    print(" Loading datasets...")
    return (
        pd.read_csv(r"C:\Users\NCC-HPC19\Documents\logistics_dataset\olist_orders_dataset.csv"),
        pd.read_csv(r"C:\Users\NCC-HPC19\Documents\logistics_dataset\olist_order_items_dataset.csv"),
        pd.read_csv(r"C:\Users\NCC-HPC19\Documents\logistics_dataset\olist_products_dataset.csv"),
        pd.read_csv(r"C:\Users\NCC-HPC19\Documents\logistics_dataset\olist_sellers_dataset.csv"),
        pd.read_csv(r"C:\Users\NCC-HPC19\Documents\logistics_dataset\olist_customers_dataset.csv"),
        pd.read_csv(r"C:\Users\NCC-HPC19\Documents\logistics_dataset\olist_order_payments_dataset.csv"),
    )

# ==============================
# MERGE
# ==============================
def merge_datasets(o, oi, p, s, c, pay):
    print(" Merging datasets...")
    pay = pay.groupby("order_id", as_index=False)["payment_value"].sum()

    df = o.merge(oi, on="order_id", how="left")
    df = df.merge(p, on="product_id", how="left")
    df = df.merge(s, on="seller_id", how="left")
    df = df.merge(c, on="customer_id", how="left")
    df = df.merge(pay, on="order_id", how="left")

    return df

# ==============================
# FEATURE ENGINEERING
# ==============================
def feature_engineering(df):
    print(" Engineering features...")

    # Convert numeric safely
    numeric_cols = [
        "price", "freight_value", "payment_value",
        "product_weight_g", "seller_zip_code_prefix",
        "customer_zip_code_prefix"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.fillna({
        'price': 0,
        'freight_value': 0,
        'payment_value': 0,
        'product_weight_g': 0
    }, inplace=True)

    # Datetime conversion
    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"], errors="coerce"
    )
    df["order_delivered_customer_date"] = pd.to_datetime(
        df["order_delivered_customer_date"], errors="coerce"
    )

    # Remove rows with invalid timestamps
    df = df.dropna(subset=["order_delivered_customer_date", "order_purchase_timestamp"])

    # Target
    df["delivery_time"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days

    # Remove negative or zero delivery times
    df = df[df["delivery_time"] > 0]
    df = df.dropna(subset=["delivery_time"])

    # Features
    df["distance"] = abs(
        df["seller_zip_code_prefix"] - df["customer_zip_code_prefix"]
    )

    df["traffic_delay"] = df["distance"] * 0.1
    df["traffic_level"] = df["traffic_delay"] / df["distance"].replace(0, 1)

    df["weight"] = df["product_weight_g"] / 1000
    df["price"] = df["price"]
    df["freight"] = df["freight_value"]
    df["payment"] = df["payment_value"]

    df["departure_hour"] = df["order_purchase_timestamp"].dt.hour.fillna(0)
    
    # Add mode based on distance (for initial training)
    df["mode"] = df["distance"].apply(
        lambda x: 1 if x < 500 else 2 if x < 1000 else 3
    )
    
    df["priority"] = np.random.randint(1, 4, size=len(df))

    # Select features
    df = df[[
        "distance", "traffic_delay", "traffic_level", "weight", "price",
        "freight", "payment", "departure_hour", "priority", "mode",
        "delivery_time"
    ]]

    df.fillna(0, inplace=True)
    df = df.drop_duplicates()

    return df

# ==============================
# TRAIN
# ==============================
def train_model(df):
    print(" Training model...")

    X = df.drop("delivery_time", axis=1)
    y = df["delivery_time"]

    print(f" Features: {list(X.columns)}")
    print(f" Training samples: {len(X)}")
    print(f" Target range: {y.min():.2f} - {y.max():.2f} days")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n Model Performance:")
    print(f"   MAE: {mae:.2f} days")
    print(f"   R² Score: {r2:.4f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n Feature Importance:")
    print(feature_importance.to_string(index=False))

    # Save
    joblib.dump(X.columns.tolist(), "features.pkl")

    return model

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":

    print("=" * 50)
    print("  LOGISTICS AI MODEL TRAINING")
    print("=" * 50)

    o, oi, p, s, c, pay = load_datasets()
    print(f" Orders: {len(o)}, Items: {len(oi)}, Products: {len(p)}")

    df = merge_datasets(o, oi, p, s, c, pay)
    print(f" Merged dataset: {df.shape}")

    df = feature_engineering(df)
    print(f" Final dataset: {df.shape}")

    model = train_model(df)

    joblib.dump(model, "model.pkl")

    print("\n" + "=" * 50)
    print(" Model Saved Successfully!")
    print("=" * 50)