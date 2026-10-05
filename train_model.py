"""
train_model.py - Standalone training script for Student Performance Predictor.
Runs the complete reproducible training and evaluation pipeline.
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.model import train_and_evaluate


def main():
    print("=" * 70)
    print(" STUDENT PERFORMANCE PREDICTOR - MODEL TRAINING PIPELINE")
    print("=" * 70)

    data_path = os.path.join("data", "student-mat.csv")
    models_dir = "models"

    print(f"\n[1/4] Loading and validating dataset from '{data_path}'...")
    try:
        results = train_and_evaluate(
            data_path=data_path,
            save_dir=models_dir,
            test_size=0.2,
            random_state=42
        )
    except Exception as e:
        print(f"\n[ERROR] Pipeline execution failed: {e}")
        sys.exit(1)

    print(f"\n[2/4] Dataset statistics:")
    print(f"      Total records: {results['dataset_total_samples']}")
    print(f"      Training set:  {results['train_samples']} samples (80%)")
    print(f"      Held-out test: {results['test_samples']} samples (20%)")
    print(f"      Features:      {', '.join(results['feature_columns'])}")

    print(f"\n[3/4] Model Evaluation Comparison (Held-out Test Set):")
    print("-" * 70)
    print(f"{'Model':<24} | {'Accuracy':<10} | {'Macro Prec':<11} | {'Macro Rec':<10} | {'Macro F1':<10}")
    print("-" * 70)

    for m_key, m_name in [
        ("baseline", "Majority Baseline"),
        ("logistic_regression", "Logistic Regression"),
        ("random_forest", "Random Forest")
    ]:
        m = results["models"][m_key]
        print(
            f"{m_name:<24} | "
            f"{m['accuracy'] * 100:>8.2f}% | "
            f"{m['macro_precision'] * 100:>9.2f}% | "
            f"{m['macro_recall'] * 100:>8.2f}% | "
            f"{m['macro_f1'] * 100:>8.2f}%"
        )
    print("-" * 70)

    print(f"\n[4/4] Model Selection Decision:")
    print(f"      Champion Model: {results['selected_model']}")
    print(f"      Rationale:      {results['selection_rationale']}")
    print(f"\n[SUCCESS] Artifacts successfully serialized to '{models_dir}/'.")
    print("=" * 70)


if __name__ == "__main__":
    main()
