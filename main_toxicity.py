import pandas as pd
import rdkit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#Load CSV
df = pd.read_csv("C:/Users/ximec/Downloads/BIOFACQUIM_V1.csv")

#Get molecular descriptors
from rdkit import Chem
from rdkit.Chem import Descriptors

def calc_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return pd.Series({
            "MW": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol),
            "HBD": Descriptors.NumHDonors(mol),
            "HBA": Descriptors.NumHAcceptors(mol)
        })
    else:
        return pd.Series([None]*5) #We need none for each descriptor
    
df_descriptors = df["SMILES"].apply(calc_descriptors)

#Combine datasets
df = pd.concat([df, df_descriptors], axis = 1)

#Define toxicity
df["Toxicity"] = df.apply(
    lambda row: 1 
    if (row["LogP"] > 3 and row["TPSA"] < 75 and row["MW"] > 300)
    else 0,
    axis=1)

#Clean
df = df.dropna()

#ML model
X = df[["HBD", "HBA"]]
Y = df["Toxicity"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2) #Common distribution 80/20

model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight={0:1, 1:2},
        random_state=42
)
model.fit(X_train, Y_train)

Y_prediction = model.predict(X_test)

#Cross validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, Y, cv=5) 
print("CV Accuracy:", scores.mean())

print("Accuracy:", accuracy_score(Y_test, Y_prediction))
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_prediction))