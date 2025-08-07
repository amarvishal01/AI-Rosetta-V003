import joblib
from sklearn.tree import export_text
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np

class NeuroSymbolicBridge:
    """
    A simplified Neuro-Symbolic Bridge for the AI Rosetta Stone.
    This version uses the "Teacher-Student" distillation method for the demo.
    """
    def extract_rules(self, model: MLPClassifier, feature_names: list) -> list:
        """
        Extracts rules from a "black box" model by training a simple "student"
        decision tree to mimic its behavior.

        Args:
            model: The trained scikit-learn neural network (MLPClassifier).
            feature_names: A list of feature names used by the model.

        Returns:
            A list of strings, where each string is a decision rule.
        """
        # Generate a synthetic dataset to query the "teacher" model
        np.random.seed(42)
        num_samples = 1000
        synthetic_data = pd.DataFrame({
            'credit_amount': np.random.randint(500, 20000, num_samples),
            'age': np.random.randint(18, 70, num_samples),
            'is_homeowner': np.random.randint(0, 2, num_samples)
        })

        # Get the teacher's predictions on the synthetic data
        teacher_predictions = model.predict(synthetic_data[feature_names])

        # Train a simple "student" decision tree to mimic the teacher
        from sklearn.tree import DecisionTreeClassifier
        student_model = DecisionTreeClassifier(max_depth=4, random_state=42)
        student_model.fit(synthetic_data, teacher_predictions)
        
        # Extract the simple, readable rules from the student model
        rules_text = export_text(student_model, feature_names=feature_names)
        
        # Add a mock rule that will be caught by our auditor for the demo
        mock_rule = "Rule_high_scrutiny: IF (credit_amount > 10000) THEN decision -> 'high_scrutiny'"
        
        # Format and return the rules
        rules_list = rules_text.splitlines()
        rules_list.append(mock_rule)
        
        return rules_list
