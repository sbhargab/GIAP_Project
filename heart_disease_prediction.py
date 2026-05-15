import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)


df = pd.read_csv("heart.csv")


print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistics:")
print(df.describe())

print("\nTarget Distribution:")
print(df["target"].value_counts())
print(f" 0 = No Disease: {(df['target'] == 0).sum()}")
print(f" 1 = Disease: {(df['target'] == 1).sum()}")


plt.figure(figsize=(5, 4))
df["target"].value_counts().plot(kind="bar", color=["steelblue", "tomato"], edgecolor="black")
plt.title("Target Distribution")
plt.xlabel("Heart Disease (0=No, 1=Yes)")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("plot_target_distribution.png", dpi=150)
plt.close()


plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("plot_correlation_heatmap.png", dpi=150)
plt.close()


plt.figure(figsize=(7, 5))
for label, color in zip([0, 1], ["steelblue", "tomato"]):
    subset = df[df["target"] == label]
    plt.scatter(subset["age"], subset["thalach"],
                label="No Disease" if label == 0 else "Disease",
                alpha=0.6, color=color)
plt.xlabel("Age")
plt.ylabel("Max Heart Rate (thalach)")
plt.title("Age vs Max Heart Rate by Heart Disease")
plt.legend()
plt.tight_layout()
plt.savefig("plot_age_vs_thalach.png", dpi=150)
plt.close()


X = df.drop("target", axis=1)
y = df["target"]

print("\nFeatures used:", list(X.columns))


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}

for name, model in models.items():
    X_tr = X_train_scaled if name == "Logistic Regression" else X_train
    X_te = X_test_scaled if name == "Logistic Regression" else X_test

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    cv = cross_val_score(model, X_tr, y_train, cv=5, scoring="accuracy").mean()

    results[name] = {"Accuracy": acc, "ROC-AUC": roc, "CV Accuracy": cv}

    print(f"\n{name}")
    print(f" Accuracy : {acc:.4f}")
    print(f" ROC-AUC : {roc:.4f}")
    print(f" 5-Fold CV : {cv:.4f}")
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    fname = f"plot_cm_{name.replace(' ', '_').lower()}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f" Saved: {fname}")


plt.figure(figsize=(7, 5))
for name, model in models.items():
    X_te = X_test_scaled if name == "Logistic Regression" else X_test
    y_prob = model.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")

plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - All Models")
plt.legend()
plt.tight_layout()
plt.savefig("plot_roc_curves.png", dpi=150)
plt.close()

rf_model = models["Random Forest"]
importances = rf_model.feature_importances_
feat_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print("\nFeature Importances (Random Forest):")
print(feat_df.to_string(index=False))

plt.figure(figsize=(8, 5))
sns.barplot(data=feat_df, x="Importance", y="Feature", palette="viridis")
plt.title("Feature Importances - Random Forest")
plt.tight_layout()
plt.savefig("plot_feature_importance.png", dpi=150)
plt.close()

print("\n\nFINAL RESULTS SUMMARY")

results_df = pd.DataFrame(results).T
print(results_df.round(4))

best_model = results_df["Accuracy"].idxmax()
print(f"\nBest Model by Accuracy: {best_model}")
print(f" Accuracy: {results_df.loc[best_model, 'Accuracy']:.4f}")
print(f" ROC-AUC : {results_df.loc[best_model, 'ROC-AUC']:.4f}")
