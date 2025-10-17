import argparse
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def train_model(output_path: str = "artifacts/rf_iris.joblib", smoke_test: bool = False):
    """Train a simple RandomForest model on the Iris dataset."""
    print("Loading Iris dataset...")
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    # Smaller model for smoke test
    n_estimators = 2 if smoke_test else 100
    print(f"Training RandomForestClassifier with n_estimators={n_estimators}...")

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate (optional)
    acc = model.score(X_test, y_test)
    print(f"Model accuracy: {acc:.3f}")

    # Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a RandomForest model for Iris dataset.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick training for CI smoke test.")
    args = parser.parse_args()

    train_model(smoke_test=args.smoke_test)
