
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


df = pd.read_csv("fraudTrain.csv")

print(df.head())


df = df[[
    "category",
    "amt",
    "gender",
    "city_pop",
    "job",
    "is_fraud"
]]


encoder = LabelEncoder()

df["category"] = encoder.fit_transform(
    df["category"]
)

df["gender"] = encoder.fit_transform(
    df["gender"]
)

df["job"] = encoder.fit_transform(
    df["job"]
)


X = df.drop("is_fraud", axis=1)

y = df["is_fraud"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = DecisionTreeClassifier()


model.fit(X_train, y_train)


pred = model.predict(X_test)


accuracy = accuracy_score(y_test, pred)

print("\nAccuracy :", accuracy)


print("\nClassification Report\n")

print(classification_report(y_test, pred))


cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(5,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="coolwarm"
)

plt.title("Fraud Detection Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()


fraud_counts = df["is_fraud"].value_counts()

plt.figure(figsize=(5,5))

plt.pie(
    fraud_counts,
    labels=["Legitimate", "Fraud"],
    autopct="%1.1f%%"
)

plt.title("Fraud vs Legitimate Transactions")

plt.show()

while True:

    choice = input(
        "\nTest Transaction? (yes/no): "
    )

    if choice.lower() != "yes":

        print("Program Ended")

        break

    print("\nEnter Transaction Details\n")

    category = int(input("Category Number: "))

    amt = float(input("Amount: "))

    gender = int(input("Gender (0-Female,1-Male): "))

    city_pop = int(input("City Population: "))

    job = int(input("Job Number: "))

    sample = [[
        category,
        amt,
        gender,
        city_pop,
        job
    ]]

    sample_df = pd.DataFrame(
        sample,
        columns=X.columns
    )

    result = model.predict(sample_df)[0]

    if result == 1:

        print("\n🚫 Fraud Transaction")

    else:

        print("\n✅ Legitimate Transaction")