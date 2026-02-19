import pandas as pd
import numpy as np
import faiss
import os
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class PlacementEngine:
    def __init__(self, csv_path=r'C:\Users\user\Downloads\Placement.csv'):
        self.csv_path = csv_path
        self.model_embed = SentenceTransformer('all-MiniLM-L6-v2')

        self.df = None
        self.index = None
        self.text_data = []

        self.rf_placement = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_salary = RandomForestRegressor(n_estimators=100, random_state=42)

        self.cgpa_col = None
        self.intern_col = None
        self.placement_col = None
        self.salary_col = None
        self.skills_col = None

        self._prepare_data()

    # -------------------------------------------------
    # DATA PREPARATION
    # -------------------------------------------------
    def _prepare_data(self):

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Dataset not found at {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        self.df.fillna(0, inplace=True)

        # Normalize column names
        self.df.columns = self.df.columns.str.strip().str.lower()

        # Auto-detect important columns
        for col in self.df.columns:
            if "cgpa" in col:
                self.cgpa_col = col
            if "intern" in col:
                self.intern_col = col
            if "place" in col:
                self.placement_col = col
            if "salary" in col:
                self.salary_col = col
            if "skill" in col:
                self.skills_col = col

        if not all([self.cgpa_col, self.intern_col, self.placement_col, self.salary_col]):
            raise ValueError("Required columns not found in dataset.")

        # Convert placement column to numeric if needed
        if self.df[self.placement_col].dtype == object:
            self.df[self.placement_col] = self.df[self.placement_col].apply(
                lambda x: 1 if str(x).lower() in ["1", "yes", "placed"] else 0
            )

        # -------------------------------------------------
        # BUILD TEXT FOR RAG
        # -------------------------------------------------
        for _, row in self.df.iterrows():
            text = (
                f"Student Profile: CGPA {row[self.cgpa_col]}, "
                f"Internships: {row[self.intern_col]}, "
                f"Placement Status: {'Placed' if row[self.placement_col] == 1 else 'Not Placed'}, "
                f"Salary: {row[self.salary_col]} INR."
            )

            if self.skills_col:
                text += f" Skills: {row[self.skills_col]}."

            self.text_data.append(text)

        embeddings = self.model_embed.encode(self.text_data)
        embeddings = np.array(embeddings).astype("float32")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        # -------------------------------------------------
        # TRAIN ML MODELS
        # -------------------------------------------------
        X = self.df[[self.cgpa_col, self.intern_col]]

        self.rf_placement.fit(X, self.df[self.placement_col])
        self.rf_salary.fit(X, self.df[self.salary_col])

    # -------------------------------------------------
    # RAG RETRIEVER
    # -------------------------------------------------
    def get_retriever(self, query, k=3):
        query_vector = self.model_embed.encode([query]).astype("float32")
        distances, indices = self.index.search(query_vector, k)
        return [self.text_data[i] for i in indices[0]]

    # -------------------------------------------------
    # ML PREDICTIONS
    # -------------------------------------------------
    def predict_placement(self, cgpa, internships):
        prob = self.rf_placement.predict_proba([[cgpa, internships]])[0][1]
        return float(prob)

    def predict_salary(self, cgpa, internships):
        salary = self.rf_salary.predict([[cgpa, internships]])[0]
        return float(salary)

    # -------------------------------------------------
    # CONFIDENCE INTERVAL (BOOTSTRAP)
    # -------------------------------------------------
    def predict_with_confidence(self, cgpa, internships, n_samples=50):
        probs = []

        for _ in range(n_samples):
            sample = self.df.sample(frac=0.8, replace=True)
            X_sample = sample[[self.cgpa_col, self.intern_col]]
            y_sample = sample[self.placement_col]

            model = RandomForestClassifier(n_estimators=50)
            model.fit(X_sample, y_sample)

            prob = model.predict_proba([[cgpa, internships]])[0][1]
            probs.append(prob)

        return float(np.mean(probs)), float(np.std(probs))

    # -------------------------------------------------
    # FEATURE IMPORTANCE
    # -------------------------------------------------
    def get_feature_importance(self):
        return {
            "CGPA": float(self.rf_placement.feature_importances_[0]),
            "Internships": float(self.rf_placement.feature_importances_[1])
        }

    # -------------------------------------------------
    # DATASET STATISTICS
    # -------------------------------------------------
    def get_dataset_statistics(self):
        return {
            "avg_cgpa": float(self.df[self.cgpa_col].mean()),
            "max_salary": float(self.df[self.salary_col].max()),
            "placement_rate": float(self.df[self.placement_col].mean() * 100),
            "total_records": int(len(self.df))
        }

    # -------------------------------------------------
    # SKILL FREQUENCY ANALYSIS
    # -------------------------------------------------
    def get_skill_frequency(self):
        if not self.skills_col:
            return {}

        placed = self.df[self.df[self.placement_col] == 1]
        skills = placed[self.skills_col].dropna().astype(str)

        skill_list = []
        for s in skills:
            skill_list.extend([x.strip().lower() for x in s.split(",")])

        return Counter(skill_list)

    # -------------------------------------------------
    # TOP SKILLS (FOR DETERMINISTIC SCORING)
    # -------------------------------------------------
    def get_top_skills(self):
        if not self.skills_col:
            return {}

        skill_series = self.df[self.skills_col].dropna().astype(str)
        all_skills = []

        for s in skill_series:
            parts = [skill.strip().lower() for skill in s.split(",")]
            all_skills.extend(parts)

        return Counter(all_skills)


# -------------------------------------------------
# INITIALIZE ENGINE
# -------------------------------------------------
engine = PlacementEngine()


# -------------------------------------------------
# WRAPPER FUNCTIONS
# -------------------------------------------------
def get_retriever(query, k=3):
    return engine.get_retriever(query, k)


def predict_placement(cgpa, internships):
    return engine.predict_placement(cgpa, internships)


def predict_salary(cgpa, internships):
    return engine.predict_salary(cgpa, internships)


def predict_with_confidence(cgpa, internships):
    return engine.predict_with_confidence(cgpa, internships)


def get_feature_importance():
    return engine.get_feature_importance()


def get_dataset_statistics():
    return engine.get_dataset_statistics()


def get_skill_frequency():
    return engine.get_skill_frequency()


def get_top_skills():
    return engine.get_top_skills()
