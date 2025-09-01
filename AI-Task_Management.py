# === AI Task Management System ===

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# Step 1: Sample Task Dataset
# ----------------------------
data = {
    "task": [
        "Finish the project report by tomorrow",
        "Fix the login bug in the website",
        "Prepare slides for client meeting",
        "Review the budget proposal",
        "Update documentation for API",
        "Respond to customer support emails",
        "Test the payment gateway integration",
        "Organize team meeting for next sprint"
    ],
    "category": ["report","bug","presentation","review","documentation","support","testing","meeting"],
    "priority": ["High","High","Medium","Medium","Low","Low","High","Medium"]
}
df = pd.DataFrame(data)

print(df.head())

# ----------------------------
# Step 2: Task Classification (NLP)
# ----------------------------
X = df["task"]
y = df["category"]

tfidf = TfidfVectorizer(stop_words="english")
X_tfidf = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

clf = MultinomialNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Task Classification Report:")
print(metrics.classification_report(y_test, y_pred))

# ----------------------------
# Step 3: Priority Prediction
# ----------------------------
X_priority = X_tfidf
y_priority = df["priority"]

Xp_train, Xp_test, yp_train, yp_test = train_test_split(X_priority, y_priority, test_size=0.3, random_state=42)

rf = RandomForestClassifier(random_state=42)
rf.fit(Xp_train, yp_train)

yp_pred = rf.predict(Xp_test)

print("Priority Prediction Report:")
print(metrics.classification_report(yp_test, yp_pred))

# ----------------------------
# Step 4: Visualization
# ----------------------------
sns.countplot(x="priority", data=df)
plt.title("Task Priority Distribution")
plt.show()
