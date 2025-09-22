import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Real-time monitoring
import time
import threading
from collections import deque
import json

class GlobalWaterQualityMonitor:
    """
    Global real-time water quality monitoring and contaminant detection system
    Supports country-specific standards and pollution patterns
    """
    
    def __init__(self, country='WHO'):
        self.country = country
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.contaminant_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.data_buffer = deque(maxlen=1000)
        self.alerts = []
        self.is_monitoring = False
        
        # Country-specific water quality standards and pollution patterns
        self.country_standards = {
            'WHO': {  # World Health Organization standards (baseline)
                'pH': {'min': 6.5, 'max': 8.5},
                'dissolved_oxygen': {'min': 5.0, 'max': 14.0},
                'turbidity': {'min': 0.0, 'max': 4.0},
                'temperature': {'min': 0.0, 'max': 35.0},
                'conductivity': {'min': 0.0, 'max': 1500.0},
                'total_dissolved_solids': {'min': 0.0, 'max': 1000.0},
                'chlorine': {'min': 0.2, 'max': 4.0},
                'ammonia': {'min': 0.0, 'max': 1.5},
                'nitrates': {'min': 0.0, 'max': 10.0},
                'heavy_metals': {'min': 0.0, 'max': 0.01},
                'fluoride': {'min': 0.0, 'max': 1.5},
                'arsenic': {'min': 0.0, 'max': 0.01}
            },
            'China': {  # More stringent due to industrial pollution concerns
                'pH': {'min': 6.5, 'max': 8.5},
                'dissolved_oxygen': {'min': 6.0, 'max': 14.0},
                'turbidity': {'min': 0.0, 'max': 3.0},
                'temperature': {'min': 0.0, 'max': 35.0},
                'conductivity': {'min': 0.0, 'max': 1200.0},
                'total_dissolved_solids': {'min': 0.0, 'max': 800.0},
                'chlorine': {'min': 0.3, 'max': 3.0},
                'ammonia': {'min': 0.0, 'max': 1.0},
                'nitrates': {'min': 0.0, 'max': 8.0},
                'heavy_metals': {'min': 0.0, 'max': 0.005},
                'fluoride': {'min': 0.0, 'max': 1.0},
                'arsenic': {'min': 0.0, 'max': 0.005}
            },
            'India': {  # Adapted for monsoon and groundwater issues
                'pH': {'min': 6.5, 'max': 8.5},
                'dissolved_oxygen': {'min': 5.0, 'max': 14.0},
                'turbidity': {'min': 0.0, 'max': 5.0},
                'temperature': {'min': 15.0, 'max': 40.0},
                'conductivity': {'min': 0.0, 'max': 2000.0},
                'total_dissolved_solids': {'min': 0.0, 'max': 1200.0},
                'chlorine': {'min': 0.2, 'max': 4.0},
                'ammonia': {'min': 0.0, 'max': 1.5},
                'nitrates': {'min': 0.0, 'max': 12.0},
                'heavy_metals': {'min': 0.0, 'max': 0.01},
                'fluoride': {'min': 0.0, 'max': 1.5},
                'arsenic': {'min': 0.0, 'max': 0.01}
            },
            'USA': {  # EPA standards
                'pH': {'min': 6.5, 'max': 8.5},
                'dissolved_oxygen': {'min': 5.0, 'max': 14.0},
                'turbidity': {'min': 0.0, 'max': 1.0},
                'temperature': {'min': 0.0, 'max': 35.0},
                'conductivity': {'min': 0.0, 'max': 1000.0},
                'total_dissolved_solids': {'min': 0.0, 'max': 500.0},
                'chlorine': {'min': 0.2, 'max': 4.0},
                'ammonia': {'min': 0.0, 'max': 0.5},
                'nitrates': {'min': 0.0, 'max': 10.0},
                'heavy_metals': {'min': 0.0, 'max': 0.005},
                'fluoride': {'min': 0.7, 'max': 2.0},
                'arsenic': {'min': 0.0, 'max': 0.01}
            },
            'Europe': {  # EU standards (generally strict)
                'pH': {'min': 6.5, 'max': 9.5},
                'dissolved_oxygen': {'min': 5.0, 'max': 14.0},
                'turbidity': {'min': 0.0, 'max': 1.0},
                'temperature': {'min': 0.0, 'max': 25.0},
                'conductivity': {'min': 0.0, 'max': 2500.0},
                'total_dissolved_solids': {'min': 0.0, 'max': 1500.0},
                'chlorine': {'min': 0.1, 'max': 0.5},
                'ammonia': {'min': 0.0, 'max': 0.5},
                'nitrates': {'min': 0.0, 'max': 11.3},
                'heavy_metals': {'min': 0.0, 'max': 0.005},
                'fluoride': {'min': 0.0, 'max': 1.5},
                'arsenic': {'min': 0.0, 'max': 0.01}
            },
            'Middle_East': {  # Adapted for arid climate and desalination
                'pH': {'min': 6.5, 'max': 8.5},
                'dissolved_oxygen': {'min': 4.0, 'max': 14.0},
                'turbidity': {'min': 0.0, 'max': 4.0},
                'temperature': {'min': 10.0, 'max': 45.0},
                'conductivity': {'min': 0.0, 'max': 3000.0},
                'total_dissolved_solids': {'min': 0.0, 'max': 2000.0},
                'chlorine': {'min': 0.2, 'max': 5.0},
                'ammonia': {'min': 0.0, 'max': 1.5},
                'nitrates': {'min': 0.0, 'max': 15.0},
                'heavy_metals': {'min': 0.0, 'max': 0.01},
                'fluoride': {'min': 0.0, 'max': 1.5},
                'arsenic': {'min': 0.0, 'max': 0.01}
            },
            'Canada': {  # Health Canada guidelines
                'pH': {'min': 6.5, 'max': 8.5},
                'dissolved_oxygen': {'min': 6.0, 'max': 14.0},
                'turbidity': {'min': 0.0, 'max': 1.0},
                'temperature': {'min': 0.0, 'max': 25.0},
                'conductivity': {'min': 0.0, 'max': 1000.0},
                'total_dissolved_solids': {'min': 0.0, 'max': 500.0},
                'chlorine': {'min': 0.2, 'max': 4.0},
                'ammonia': {'min': 0.0, 'max': 0.5},
                'nitrates': {'min': 0.0, 'max': 10.0},
                'heavy_metals': {'min': 0.0, 'max': 0.005},
                'fluoride': {'min': 0.0, 'max': 1.5},
                'arsenic': {'min': 0.0, 'max': 0.01}
            },
            'Australia': {  # Australian guidelines
                'pH': {'min': 6.5, 'max': 8.5},
                'dissolved_oxygen': {'min': 5.0, 'max': 14.0},
                'turbidity': {'min': 0.0, 'max': 5.0},
                'temperature': {'min': 5.0, 'max': 35.0},
                'conductivity': {'min': 0.0, 'max': 1500.0},
                'total_dissolved_solids': {'min': 0.0, 'max': 1000.0},
                'chlorine': {'min': 0.2, 'max': 5.0},
                'ammonia': {'min': 0.0, 'max': 1.0},
                'nitrates': {'min': 0.0, 'max': 11.0},
                'heavy_metals': {'min': 0.0, 'max': 0.007},
                'fluoride': {'min': 0.6, 'max': 1.5},
                'arsenic': {'min': 0.0, 'max': 0.007}
            }
        }
        
        self.country_pollution_patterns = {
            'China': {
                'industrial_heavy_metals': 0.25,  # High industrial pollution
                'agricultural_runoff': 0.20,
                'urban_waste': 0.15,
                'typical_contaminants': ['heavy_metals', 'ammonia', 'turbidity', 'conductivity']
            },
            'India': {
                'agricultural_runoff': 0.30,  # High agricultural pollution
                'urban_waste': 0.25,
                'industrial_heavy_metals': 0.15,
                'typical_contaminants': ['nitrates', 'ammonia', 'turbidity', 'heavy_metals']
            },
            'USA': {
                'agricultural_runoff': 0.15,  # Moderate, well-regulated
                'urban_waste': 0.10,
                'industrial_heavy_metals': 0.08,
                'typical_contaminants': ['nitrates', 'chlorine']
            },
            'Europe': {
                'agricultural_runoff': 0.12,  # Well-regulated
                'urban_waste': 0.08,
                'industrial_heavy_metals': 0.05,
                'typical_contaminants': ['nitrates', 'fluoride']
            },
            'Middle_East': {
                'desalination_byproducts': 0.20,  # High salinity issues
                'urban_waste': 0.15,
                'agricultural_runoff': 0.10,
                'typical_contaminants': ['conductivity', 'total_dissolved_solids', 'chlorine']
            },
            'Canada': {
                'agricultural_runoff': 0.10,  # Low pollution
                'urban_waste': 0.08,
                'industrial_heavy_metals': 0.05,
                'typical_contaminants': ['nitrates', 'fluoride']
            },
            'Australia': {
                'agricultural_runoff': 0.15,  # Moderate agricultural impact
                'mining_runoff': 0.12,
                'urban_waste': 0.08,
                'typical_contaminants': ['heavy_metals', 'turbidity', 'fluoride']
            }
        }
        
        self.standards = self.country_standards.get(country, self.country_standards['WHO'])
        
        # Risk levels
        self.risk_levels = {
            0: 'Safe',
            1: 'Low Risk',
            2: 'Medium Risk', 
            3: 'High Risk',
            4: 'Critical'
        }
    
    def set_country(self, country):
        """Change monitoring country and update standards"""
        if country in self.country_standards:
            self.country = country
            self.standards = self.country_standards[country]
            print(f"Switched to {country} water quality standards")
        else:
            print(f"Country '{country}' not supported. Available: {list(self.country_standards.keys())}")
    
    def generate_country_specific_data(self, n_samples=10000, country=None):
        """Generate synthetic water quality data based on country-specific patterns"""
        if country is None:
            country = self.country
            
        np.random.seed(42)
        
        # Get country-specific pollution pattern
        pollution_pattern = self.country_pollution_patterns.get(country, {
            'industrial_heavy_metals': 0.10,
            'agricultural_runoff': 0.10,
            'urban_waste': 0.05
        })
        
        # Calculate total contamination rate (exclude non-numeric values)
        contamination_rates = {k: v for k, v in pollution_pattern.items() if isinstance(v, (int, float))}
        total_contamination_rate = sum(contamination_rates.values()) if country != 'WHO' else 0.15
        
        normal_samples = int(n_samples * (1 - total_contamination_rate))
        contaminated_samples = n_samples - normal_samples
        
        # Generate country-specific baseline values
        if country == 'China':
            base_values = {
                'pH': (7.0, 0.6), 'dissolved_oxygen': (7.5, 1.5), 'turbidity': (2.0, 1.5),
                'temperature': (18, 8), 'conductivity': (700, 250), 'total_dissolved_solids': (400, 150),
                'chlorine': (0.8, 0.4), 'ammonia': (0.3, 0.2), 'nitrates': (3.0, 2.0),
                'heavy_metals': (0.003, 0.002), 'fluoride': (0.5, 0.3), 'arsenic': (0.002, 0.001)
            }
        elif country == 'India':
            base_values = {
                'pH': (7.3, 0.7), 'dissolved_oxygen': (8.0, 1.5), 'turbidity': (3.0, 2.0),
                'temperature': (25, 8), 'conductivity': (800, 400), 'total_dissolved_solids': (500, 200),
                'chlorine': (1.2, 0.5), 'ammonia': (0.4, 0.3), 'nitrates': (4.0, 3.0),
                'heavy_metals': (0.004, 0.003), 'fluoride': (0.8, 0.4), 'arsenic': (0.003, 0.002)
            }
        elif country in ['USA', 'Canada']:
            base_values = {
                'pH': (7.4, 0.3), 'dissolved_oxygen': (9.0, 1.0), 'turbidity': (0.5, 0.3),
                'temperature': (15, 6), 'conductivity': (400, 100), 'total_dissolved_solids': (250, 75),
                'chlorine': (1.5, 0.3), 'ammonia': (0.1, 0.05), 'nitrates': (1.5, 1.0),
                'heavy_metals': (0.001, 0.0005), 'fluoride': (1.0, 0.2), 'arsenic': (0.001, 0.0005)
            }
        elif country == 'Europe':
            base_values = {
                'pH': (7.5, 0.4), 'dissolved_oxygen': (9.5, 1.0), 'turbidity': (0.3, 0.2),
                'temperature': (12, 5), 'conductivity': (300, 100), 'total_dissolved_solids': (200, 50),
                'chlorine': (0.3, 0.1), 'ammonia': (0.1, 0.05), 'nitrates': (2.0, 1.5),
                'heavy_metals': (0.001, 0.0005), 'fluoride': (0.7, 0.3), 'arsenic': (0.001, 0.0005)
            }
        elif country == 'Middle_East':
            base_values = {
                'pH': (7.2, 0.5), 'dissolved_oxygen': (6.5, 1.5), 'turbidity': (1.5, 1.0),
                'temperature': (28, 10), 'conductivity': (1200, 500), 'total_dissolved_solids': (800, 300),
                'chlorine': (2.0, 0.8), 'ammonia': (0.3, 0.2), 'nitrates': (5.0, 3.0),
                'heavy_metals': (0.003, 0.002), 'fluoride': (0.6, 0.3), 'arsenic': (0.002, 0.001)
            }
        elif country == 'Australia':
            base_values = {
                'pH': (7.1, 0.4), 'dissolved_oxygen': (8.5, 1.2), 'turbidity': (1.2, 0.8),
                'temperature': (20, 8), 'conductivity': (600, 200), 'total_dissolved_solids': (350, 100),
                'chlorine': (1.8, 0.4), 'ammonia': (0.2, 0.1), 'nitrates': (2.5, 1.5),
                'heavy_metals': (0.002, 0.001), 'fluoride': (1.2, 0.2), 'arsenic': (0.002, 0.001)
            }
        else:  # Default WHO values
            base_values = {
                'pH': (7.2, 0.5), 'dissolved_oxygen': (8.5, 1.2), 'turbidity': (1.0, 0.5),
                'temperature': (20, 5), 'conductivity': (500, 150), 'total_dissolved_solids': (300, 100),
                'chlorine': (1.0, 0.3), 'ammonia': (0.2, 0.1), 'nitrates': (2.0, 1.0),
                'heavy_metals': (0.002, 0.001), 'fluoride': (0.8, 0.3), 'arsenic': (0.002, 0.001)
            }
        
        # Generate normal data
        normal_data = {}
        for param, (mean, std) in base_values.items():
            if param in ['ammonia', 'nitrates', 'heavy_metals', 'arsenic']:
                normal_data[param] = np.random.exponential(mean, normal_samples)
            else:
                normal_data[param] = np.random.normal(mean, std, normal_samples)
                normal_data[param] = np.clip(normal_data[param], 0, None)  # Ensure positive values
        
        # Generate contaminated data based on country-specific patterns
        contaminated_data = {}
        typical_contaminants = pollution_pattern.get('typical_contaminants', 
                                                   ['heavy_metals', 'turbidity', 'ammonia'])
        
        for param in base_values.keys():
            mean, std = base_values[param]
            
            if param in typical_contaminants:
                # Higher contamination for typical contaminants
                if param == 'heavy_metals':
                    contaminated_data[param] = np.random.exponential(mean * 10, contaminated_samples)
                elif param == 'turbidity':
                    contaminated_data[param] = np.random.exponential(mean * 8, contaminated_samples)
                elif param == 'ammonia':
                    contaminated_data[param] = np.random.exponential(mean * 15, contaminated_samples)
                elif param == 'nitrates':
                    contaminated_data[param] = np.random.exponential(mean * 6, contaminated_samples)
                elif param == 'conductivity':
                    contaminated_data[param] = np.random.normal(mean * 2.5, std * 1.5, contaminated_samples)
                elif param == 'total_dissolved_solids':
                    contaminated_data[param] = np.random.normal(mean * 2.8, std * 1.5, contaminated_samples)
                elif param == 'pH':
                    # Create both acidic and basic contamination
                    acidic_count = contaminated_samples // 3
                    basic_count = contaminated_samples // 3
                    normal_count = contaminated_samples - acidic_count - basic_count
                    contaminated_data[param] = np.concatenate([
                        np.random.normal(4.5, 1.0, acidic_count),
                        np.random.normal(10.0, 1.0, basic_count),
                        np.random.normal(mean, std, normal_count)
                    ])
                else:
                    contaminated_data[param] = np.random.normal(mean * 1.8, std * 2, contaminated_samples)
            else:
                # Moderate contamination for other parameters
                contaminated_data[param] = np.random.normal(mean * 1.3, std * 1.5, contaminated_samples)
            
            # Ensure positive values
            contaminated_data[param] = np.clip(contaminated_data[param], 0, None)
        
        # Combine data
        data = {}
        for param in base_values.keys():
            data[param] = np.concatenate([normal_data[param], contaminated_data[param]])
        
        # Create risk level labels
        normal_labels = np.zeros(normal_samples)
        contaminated_labels = []
        
        for i in range(contaminated_samples):
            risk_score = 0
            sample_data = {param: data[param][normal_samples + i] for param in data.keys()}
            
            # Calculate risk based on parameter violations
            for param, value in sample_data.items():
                if param in self.standards:
                    std = self.standards[param]
                    if value < std['min'] or value > std['max']:
                        if value < std['min']:
                            violation_ratio = (std['min'] - value) / std['min'] if std['min'] > 0 else 1
                        else:
                            violation_ratio = (value - std['max']) / std['max'] if std['max'] > 0 else 1
                        
                        # Weight violations by parameter importance
                        weight = 2.0 if param in typical_contaminants else 1.0
                        
                        if violation_ratio > 2.0:
                            risk_score += 2 * weight
                        elif violation_ratio > 1.0:
                            risk_score += 1.5 * weight
                        elif violation_ratio > 0.5:
                            risk_score += 1 * weight
                        else:
                            risk_score += 0.5 * weight
            
            # Convert risk score to risk level
            if risk_score >= 10:
                risk_level = 4  # Critical
            elif risk_score >= 7:
                risk_level = 3  # High Risk
            elif risk_score >= 4:
                risk_level = 2  # Medium Risk
            elif risk_score >= 2:
                risk_level = 1  # Low Risk
            else:
                risk_level = 0  # Safe
            
            contaminated_labels.append(risk_level)
        
        labels = np.concatenate([normal_labels, np.array(contaminated_labels)])
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df['risk_level'] = labels
        df['country'] = country
        df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='H')
        
        # Add temporal patterns
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['month'] = df['timestamp'].dt.month
        
        # Country-specific seasonal adjustments
        if country in ['India', 'Middle_East']:
            # Monsoon/dry season effects
            monsoon_months = [6, 7, 8, 9]  # June-September
            monsoon_mask = df['month'].isin(monsoon_months)
            df.loc[monsoon_mask, 'turbidity'] *= 1.5
            df.loc[monsoon_mask, 'dissolved_oxygen'] *= 0.9
        
        if country in ['China', 'India']:
            # Industrial activity patterns
            industrial_hours = list(range(8, 18))  # 8 AM - 6 PM
            industrial_mask = df['hour'].isin(industrial_hours)
            df.loc[industrial_mask, 'heavy_metals'] *= 1.3
            df.loc[industrial_mask, 'ammonia'] *= 1.2
        
        # Temperature seasonal variation
        df['temperature'] += 10 * np.sin(2 * np.pi * df['day_of_year'] / 365)
        
        return df
    
    def train_models(self, data):
        """Train anomaly detection and contaminant classification models"""
        print(f"Training models for {self.country} water quality standards...")
        
        feature_cols = [col for col in data.columns 
                       if col not in ['risk_level', 'timestamp', 'hour', 'day_of_year', 'month', 'country']]
        X = data[feature_cols]
        y = data['risk_level']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train anomaly detector on normal data only
        normal_data = X_scaled[y == 0]
        self.anomaly_detector.fit(normal_data)
        
        # FIX 1: Check for classes with only one member before stratifying.
        # This prevents a crash if a class is too small to be split.
        stratify_option = y
        if y.value_counts().min() < 2:
            print("\nWarning: A class has only one sample. Disabling stratification for this split to prevent a crash.\n")
            stratify_option = None

        # Train contaminant classifier
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, 
                                                           random_state=42, stratify=stratify_option)
        
        # FIX 2: Added the missing .fit() call to train the classifier.
        # This resolves the NotFittedError.
        self.contaminant_classifier.fit(X_train, y_train)
        
        # Evaluate classifier
        y_pred = self.contaminant_classifier.predict(X_test)
        # Filter target_names to match unique classes in y_test
        unique_labels = sorted(set(y_test))
        target_names = [self.risk_levels[int(label)] for label in unique_labels]
        print(f"\n{self.country} Contaminant Classification Report:")
        # Added zero_division=0 to prevent warnings on scores if a class has no predicted samples.
        print(classification_report(y_test, y_pred, target_names=target_names, labels=unique_labels, zero_division=0))
        
        return X_test, y_test, y_pred
    
    def detect_anomalies(self, sample):
        """Detect anomalies in water quality sample using country-specific models"""
        feature_cols = ['pH', 'dissolved_oxygen', 'turbidity', 'temperature', 'conductivity',
                       'total_dissolved_solids', 'chlorine', 'ammonia', 'nitrates', 'heavy_metals',
                       'fluoride', 'arsenic']
        
        if isinstance(sample, dict):
            sample_array = np.array([[sample.get(col, 0) for col in feature_cols]])
        else:
            sample_array = sample.reshape(1, -1)
        
        sample_scaled = self.scaler.transform(sample_array)
        
        # Anomaly detection
        anomaly_score = self.anomaly_detector.decision_function(sample_scaled)[0]
        is_anomaly = self.anomaly_detector.predict(sample_scaled)[0] == -1
        
        # Risk level prediction
        risk_level = self.contaminant_classifier.predict(sample_scaled)[0]
        risk_probabilities = self.contaminant_classifier.predict_proba(sample_scaled)[0]
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_score,
            'risk_level': risk_level,
            'risk_label': self.risk_levels[risk_level],
            'risk_probabilities': dict(zip(self.risk_levels.values(), risk_probabilities)),
            'country_standards': self.country
        }
    
    def check_standards_compliance(self, sample):
        """Check if sample meets country-specific standards"""
        violations = []
        
        for param, value in sample.items():
            if param in self.standards:
                std = self.standards[param]
                if value < std['min']:
                    violations.append(f"{param}: {value:.3f} (below {self.country} minimum {std['min']})")
                elif value > std['max']:
                    violations.append(f"{param}: {value:.3f} (above {self.country} maximum {std['max']})")
        
        return violations
    
    def simulate_country_sensor_reading(self, country=None):
        """Simulate real-time sensor reading based on country pollution patterns"""
        if country is None:
            country = self.country
            
        # Get country-specific patterns
        pollution_pattern = self.country_pollution_patterns.get(country, {})
        contamination_rates = {k: v for k, v in pollution_pattern.items() if isinstance(v, (int, float))}
        contamination_chance = sum(contamination_rates.values()) if contamination_rates else 0.05
        
        # Country-specific base values (same as in generate_country_specific_data)
        if country == 'China':
            base_values = {
                'pH': 7.0, 'dissolved_oxygen': 7.5, 'turbidity': 2.0, 'temperature': 18,
                'conductivity': 700, 'total_dissolved_solids': 400, 'chlorine': 0.8,
                'ammonia': 0.3, 'nitrates': 3.0, 'heavy_metals': 0.003,
                'fluoride': 0.5, 'arsenic': 0.002
            }
        elif country == 'India':
            base_values = {
                'pH': 7.3, 'dissolved_oxygen': 8.0, 'turbidity': 3.0, 'temperature': 25,
                'conductivity': 800, 'total_dissolved_solids': 500, 'chlorine': 1.2,
                'ammonia': 0.4, 'nitrates': 4.0, 'heavy_metals': 0.004,
                'fluoride': 0.8, 'arsenic': 0.003
            }
        elif country in ['USA', 'Canada']:
            base_values = {
                'pH': 7.4, 'dissolved_oxygen': 9.0, 'turbidity': 0.5, 'temperature': 15,
                'conductivity': 400, 'total_dissolved_solids': 250, 'chlorine': 1.5,
                'ammonia': 0.1, 'nitrates': 1.5, 'heavy_metals': 0.001,
                'fluoride': 1.0, 'arsenic': 0.001
            }
        elif country == 'Europe':
            base_values = {
                'pH': 7.5, 'dissolved_oxygen': 9.5, 'turbidity': 0.3, 'temperature': 12,
                'conductivity': 300, 'total_dissolved_solids': 200, 'chlorine': 0.3,
                'ammonia': 0.1, 'nitrates': 2.0, 'heavy_metals': 0.001,
                'fluoride': 0.7, 'arsenic': 0.001
            }
        elif country == 'Middle_East':
            base_values = {
                'pH': 7.2, 'dissolved_oxygen': 6.5, 'turbidity': 1.5, 'temperature': 28,
                'conductivity': 1200, 'total_dissolved_solids': 800, 'chlorine': 2.0,
                'ammonia': 0.3, 'nitrates': 5.0, 'heavy_metals': 0.003,
                'fluoride': 0.6, 'arsenic': 0.002
            }
        elif country == 'Australia':
            base_values = {
                'pH': 7.1, 'dissolved_oxygen': 8.5, 'turbidity': 1.2, 'temperature': 20,
                'conductivity': 600, 'total_dissolved_solids': 350, 'chlorine': 1.8,
                'ammonia': 0.2, 'nitrates': 2.5, 'heavy_metals': 0.002,
                'fluoride': 1.2, 'arsenic': 0.002
            }
        else:  # Default WHO values
            base_values = {
                'pH': 7.2, 'dissolved_oxygen': 8.5, 'turbidity': 1.0, 'temperature': 20,
                'conductivity': 500, 'total_dissolved_solids': 300, 'chlorine': 1.0,
                'ammonia': 0.2, 'nitrates': 2.0, 'heavy_metals': 0.002,
                'fluoride': 0.8, 'arsenic': 0.002
            }
        
        # Generate reading with country-specific contamination patterns
        reading = {}
        typical_contaminants = pollution_pattern.get('typical_contaminants', [])
        
        for param, base_value in base_values.items():
            # Check if contamination occurs
            if np.random.random() < contamination_chance:
                # Contamination event - prioritize typical contaminants
                if param in typical_contaminants or np.random.random() < 0.3:
                    if param == 'pH':
                        reading[param] = np.random.choice([
                            np.random.normal(4.5, 0.5), 
                            np.random.normal(9.5, 0.5)
                        ])
                    elif param in ['heavy_metals', 'arsenic']:
                        reading[param] = base_value * np.random.exponential(10)
                    elif param == 'turbidity':
                        reading[param] = base_value * np.random.exponential(5)
                    elif param in ['ammonia', 'nitrates']:
                        reading[param] = base_value * np.random.exponential(8)
                    elif param in ['conductivity', 'total_dissolved_solids']:
                        reading[param] = base_value * (1 + np.random.exponential(1.5))
                    else:
                        reading[param] = base_value * (1 + np.random.exponential(0.8))
                else:
                    # Normal reading with small variation
                    variation = np.random.normal(0, base_value * 0.1)
                    reading[param] = max(0, base_value + variation)
            else:
                # Normal reading with small variation
                variation = np.random.normal(0, base_value * 0.1)
                reading[param] = max(0, base_value + variation)
        
        return reading
    
    def generate_alert(self, sample, analysis_result, violations):
        """Generate country-specific alert"""
        timestamp = datetime.now()
        alert = {
            'timestamp': timestamp,
            'country': self.country,
            'risk_level': analysis_result['risk_level'],
            'risk_label': analysis_result['risk_label'],
            'is_anomaly': analysis_result['is_anomaly'],
            'violations': violations,
            'sample': sample,
            'alert_message': self._create_country_alert_message(analysis_result, violations)
        }
        
        # Add to alerts if significant
        if analysis_result['risk_level'] > 1 or analysis_result['is_anomaly'] or violations:
            self.alerts.append(alert)
            print(f"\n🚨 {self.country.upper()} WATER QUALITY ALERT - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Risk Level: {analysis_result['risk_label']}")
            print(f"Alert: {alert['alert_message']}")
            
            if violations:
                print(f"{self.country} Standards Violations:")
                for violation in violations:
                    print(f"  • {violation}")
        
        return alert
    
    def _create_country_alert_message(self, analysis_result, violations):
        """Create country-specific alert message"""
        risk_level = analysis_result['risk_level']
        country_context = {
            'China': "Consider industrial pollution sources",
            'India': "Check for agricultural runoff or monsoon effects",
            'USA': "Review EPA compliance protocols",
            'Europe': "Verify EU water directive compliance",
            'Middle_East': "Consider desalination process issues",
            'Canada': "Follow Health Canada guidelines",
            'Australia': "Check mining or agricultural impacts",
            'WHO': "Follow WHO international guidelines"
        }
        
        base_message = ""
        if risk_level == 4:
            base_message = f"CRITICAL: Water unsafe for consumption in {self.country}. Immediate action required."
        elif risk_level == 3:
            base_message = f"HIGH RISK: Significant contamination detected in {self.country}. Avoid consumption."
        elif risk_level == 2:
            base_message = f"MEDIUM RISK: Moderate contamination in {self.country}. Treatment recommended."
        elif risk_level == 1:
            base_message = f"LOW RISK: Minor quality issues detected in {self.country}. Monitor closely."
        elif analysis_result['is_anomaly']:
            base_message = f"ANOMALY: Unusual water quality patterns detected in {self.country}."
        else:
            base_message = f"Water quality within {self.country} acceptable parameters."
        
        # Add country-specific context
        context = country_context.get(self.country, "")
        if context and risk_level > 0:
            base_message += f" {context}."
        
        return base_message
    
    def start_monitoring(self, duration_minutes=10, country=None):
        """Start real-time monitoring for specific country"""
        if country and country != self.country:
            self.set_country(country)
            
        print(f"Starting real-time water quality monitoring for {self.country} ({duration_minutes} minutes)...")
        print("=" * 70)
        
        self.is_monitoring = True
        start_time = time.time()
        reading_count = 0
        
        while self.is_monitoring and (time.time() - start_time) < duration_minutes * 60:
            # Simulate country-specific sensor reading
            sample = self.simulate_country_sensor_reading()
            reading_count += 1
            
            # Add to buffer
            sample['timestamp'] = datetime.now()
            sample['country'] = self.country
            self.data_buffer.append(sample.copy())
            
            # Analyze sample
            analysis = self.detect_anomalies(sample)
            violations = self.check_standards_compliance(sample)
            
            # Generate alert if needed
            alert = self.generate_alert(sample, analysis, violations)
            
            # Print status every 10 readings
            if reading_count % 10 == 0:
                print(f"Reading #{reading_count} ({self.country}): {analysis['risk_label']} "
                      f"(pH: {sample['pH']:.2f}, Heavy Metals: {sample['heavy_metals']:.4f})")
            
            # Wait for next reading
            time.sleep(2)  # 2 seconds for demo
        
        print(f"\nMonitoring completed for {self.country}. Processed {reading_count} readings.")
        print(f"Total alerts generated: {len(self.alerts)}")
    
    def compare_country_standards(self):
        """Compare water quality standards across countries"""
        print("\n🌍 Global Water Quality Standards Comparison")
        print("=" * 60)
        
        # Key parameters to compare
        key_params = ['pH', 'turbidity', 'heavy_metals', 'nitrates', 'chlorine']
        
        comparison_df = pd.DataFrame()
        for country, standards in self.country_standards.items():
            country_data = {}
            for param in key_params:
                if param in standards:
                    country_data[f"{param}_min"] = standards[param]['min']
                    country_data[f"{param}_max"] = standards[param]['max']
            comparison_df[country] = pd.Series(country_data)
        
        print(comparison_df.round(3))
        
        # Highlight differences
        print(f"\n📊 Key Differences:")
        print(f"Strictest turbidity standards: Europe, USA, Canada (≤1.0 NTU)")
        print(f"Most lenient turbidity: India, Australia (≤5.0 NTU)")
        print(f"Lowest heavy metals limit: China, USA, Europe, Canada (≤0.005 mg/L)")
        print(f"Highest chlorine tolerance: Middle East, Australia (≤5.0 mg/L)")
        print(f"Lowest chlorine requirement: Europe (≥0.1 mg/L)")
        
        return comparison_df
    
    def visualize_global_monitoring(self):
        """Create comprehensive global monitoring dashboard"""
        if not self.data_buffer:
            print("No monitoring data to visualize.")
            return
        
        # Convert buffer to DataFrame
        df = pd.DataFrame(list(self.data_buffer))
        
        # Create comprehensive dashboard with smaller size
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f'{self.country} Water Quality Monitoring Dashboard', fontsize=14)
        
        # pH over time with country standards
        axes[0, 0].plot(df.index, df['pH'], color='blue', alpha=0.7, label='pH readings')
        axes[0, 0].axhline(y=self.standards['pH']['min'], color='red', linestyle='--', alpha=0.5, label=f'{self.country} Min pH')
        axes[0, 0].axhline(y=self.standards['pH']['max'], color='red', linestyle='--', alpha=0.5, label=f'{self.country} Max pH')
        axes[0, 0].set_title(f'pH Levels - {self.country} Standards', fontsize=10)
        axes[0, 0].set_ylabel('pH', fontsize=8)
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Heavy Metals with country-specific limits
        axes[0, 1].plot(df.index, df['heavy_metals'], color='red', alpha=0.7, label='Heavy Metals')
        axes[0, 1].axhline(y=self.standards['heavy_metals']['max'], color='red', linestyle='--', alpha=0.5, 
                          label=f'{self.country} Limit')
        axes[0, 1].set_title(f'Heavy Metals - {self.country} Standards', fontsize=10)
        axes[0, 1].set_ylabel('mg/L', fontsize=8)
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Turbidity
        axes[0, 2].plot(df.index, df['turbidity'], color='brown', alpha=0.7, label='Turbidity')
        axes[0, 2].axhline(y=self.standards['turbidity']['max'], color='red', linestyle='--', alpha=0.5,
                          label=f'{self.country} Max')
        axes[0, 2].set_title(f'Turbidity - {self.country} Standards', fontsize=10)
        axes[0, 2].set_ylabel('NTU', fontsize=8)
        axes[0, 2].legend(fontsize=8)
        axes[0, 2].grid(True, alpha=0.3)
        
        # Dissolved Oxygen
        axes[1, 0].plot(df.index, df['dissolved_oxygen'], color='green', alpha=0.7)
        axes[1, 0].axhline(y=self.standards['dissolved_oxygen']['min'], color='red', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('Dissolved Oxygen', fontsize=10)
        axes[1, 0].set_ylabel('mg/L', fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Nitrates and Ammonia
        axes[1, 1].plot(df.index, df['nitrates'], color='orange', alpha=0.7, label='Nitrates')
        axes[1, 1].plot(df.index, df['ammonia'], color='purple', alpha=0.7, label='Ammonia')
        axes[1, 1].axhline(y=self.standards['nitrates']['max'], color='orange', linestyle='--', alpha=0.5)
        axes[1, 1].axhline(y=self.standards['ammonia']['max'], color='purple', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Nitrogen Compounds', fontsize=10)
        axes[1, 1].set_ylabel('mg/L', fontsize=8)
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Conductivity and TDS
        ax1 = axes[1, 2]
        ax2 = ax1.twinx()
        line1 = ax1.plot(df.index, df['conductivity'], color='cyan', alpha=0.7, label='Conductivity')
        line2 = ax2.plot(df.index, df['total_dissolved_solids'], color='magenta', alpha=0.7, label='TDS')
        ax1.set_ylabel('Conductivity (μS/cm)', color='cyan', fontsize=8)
        ax2.set_ylabel('TDS (mg/L)', color='magenta', fontsize=8)
        ax1.set_title('Conductivity & TDS', fontsize=10)
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Chlorine levels
        axes[2, 0].plot(df.index, df['chlorine'], color='lime', alpha=0.7)
        axes[2, 0].axhline(y=self.standards['chlorine']['min'], color='red', linestyle='--', alpha=0.5, label='Min')
        axes[2, 0].axhline(y=self.standards['chlorine']['max'], color='red', linestyle='--', alpha=0.5, label='Max')
        axes[2, 0].set_title(f'Chlorine - {self.country} Standards', fontsize=10)
        axes[2, 0].set_ylabel('mg/L', fontsize=8)
        axes[2, 0].legend(fontsize=8)
        axes[2, 0].grid(True, alpha=0.3)
        
        # Risk level distribution
        if self.alerts:
            risk_levels = [alert['risk_level'] for alert in self.alerts]
            risk_counts = pd.Series(risk_levels).value_counts().sort_index()
            colors = ['green', 'yellow', 'orange', 'red', 'darkred']
            
            # Convert float indices to integers for color mapping
            risk_indices = [int(i) for i in risk_counts.index]
            axes[2, 1].bar(range(len(risk_counts)), risk_counts.values, 
                          color=[colors[i] for i in risk_indices])
            axes[2, 1].set_title('Alert Risk Distribution', fontsize=10)
            axes[2, 1].set_ylabel('Count', fontsize=8)
            axes[2, 1].set_xlabel('Risk Level', fontsize=8)
            axes[2, 1].set_xticks(range(len(risk_counts)))
            axes[2, 1].set_xticklabels([self.risk_levels[int(i)] for i in risk_counts.index], rotation=45, fontsize=8)
        else:
            axes[2, 1].text(0.5, 0.5, 'No Alerts Generated', ha='center', va='center', 
                           transform=axes[2, 1].transAxes, fontsize=8)
            axes[2, 1].set_title('Alert Risk Distribution', fontsize=10)
        
        # Temperature
        axes[2, 2].plot(df.index, df['temperature'], color='red', alpha=0.7)
        axes[2, 2].axhline(y=self.standards['temperature']['max'], color='red', linestyle='--', alpha=0.5)
        axes[2, 2].set_title('Temperature', fontsize=10)
        axes[2, 2].set_ylabel('°C', fontsize=8)
        axes[2, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print country-specific summary
        self._print_country_summary(df)
    
    def _print_country_summary(self, df):
        """Print country-specific monitoring summary"""
        print(f"\n📋 {self.country} Water Quality Summary:")
        print("=" * 40)
        
        # Get country-specific pollution patterns
        pollution_pattern = self.country_pollution_patterns.get(self.country, {})
        typical_contaminants = pollution_pattern.get('typical_contaminants', [])
        
        print(f"Monitoring Location: {self.country}")
        print(f"Total Readings: {len(df)}")
        print(f"Alert Rate: {len(self.alerts)/len(df)*100:.1f}%")
        
        if typical_contaminants:
            print(f"\nTypical Contaminants for {self.country}:")
            for contaminant in typical_contaminants:
                if contaminant in df.columns:
                    avg_val = df[contaminant].mean()
                    max_val = df[contaminant].max()
                    limit = self.standards.get(contaminant, {}).get('max', 'N/A')
                    print(f"  • {contaminant}: Avg {avg_val:.4f}, Max {max_val:.4f} (Limit: {limit})")
        
        if self.alerts:
            print(f"\nRecent Alerts:")
            for alert in self.alerts[-3:]:  # Last 3 alerts
                print(f"  • {alert['timestamp'].strftime('%H:%M:%S')}: {alert['risk_label']} - {alert['alert_message']}")

    def print_all_countries(self):
        """Print all supported countries"""
        print("\n🌍 Supported Countries for Water Quality Monitoring:")
        print("=" * 60)
        for country in self.country_standards.keys():
            print(f"  • {country}")

def demo_global_monitoring():
    """Demonstrate global water quality monitoring capabilities"""
    print("🌍 GLOBAL WATER QUALITY MONITORING SYSTEM")
    print("=" * 50)
    print("WHO = World Health Organization (International Standards)")
    print("=" * 50)
    
    # Initialize monitor
    monitor = GlobalWaterQualityMonitor()
    
    # Print all supported countries
    monitor.print_all_countries()
    
    # Compare standards across countries
    monitor.compare_country_standards()
    
    # Test all supported countries
    test_countries = ['WHO', 'China', 'India', 'USA', 'Europe', 'Middle_East', 'Canada', 'Australia']
    
    for i, country in enumerate(test_countries):
        print(f"\n🏭 Training models for {country} ({i+1}/{len(test_countries)})...")
        
        # Set country and generate data
        monitor.set_country(country)
        training_data = monitor.generate_country_specific_data(n_samples=3000, country=country)
        
        print(f"Generated {len(training_data)} samples for {country}")
        print(f"Risk distribution: {dict(training_data['risk_level'].value_counts().sort_index())}")
        
        # Train models
        X_test, y_test, y_pred = monitor.train_models(training_data)
        
        # Brief monitoring simulation
        print(f"\nStarting 20-second monitoring for {country}...")
        monitor.start_monitoring(duration_minutes=0.33)  # 20 seconds
        
        # Visualize results
        try:
            monitor.visualize_global_monitoring()
        except Exception as e:
            print(f"Visualization error for {country}: {e}")
            print("Continuing to next country...")
        
        # Clear alerts and buffer for next country
        monitor.alerts.clear()
        monitor.data_buffer.clear()
        
        print(f"✅ Completed monitoring for {country}")
        print("-" * 50)
    
    print("\n🎯 Global monitoring demonstration complete!")
    print("Tested countries with different pollution patterns:")
    for country in test_countries:
        pattern = monitor.country_pollution_patterns.get(country, {'typical_contaminants': []})
        contamination_rates = {k: v for k, v in pattern.items() if isinstance(v, (int, float))}
        rate = sum(contamination_rates.values()) * 100 if contamination_rates else 15
        context = {
            'WHO': 'International baseline standards',
            'China': 'High industrial pollution',
            'India': 'High agricultural pollution',
            'USA': 'Well-regulated system',
            'Europe': 'Strict EU regulations',
            'Middle_East': 'Desalination and salinity issues',
            'Canada': 'Low pollution, strict guidelines',
            'Australia': 'Moderate agricultural and mining impact'
        }
        print(f"• {country}: {context.get(country, 'Default patterns')} ({rate:.0f}% contamination rate)")
    
    print("\nThe system successfully adapts to different country standards and pollution patterns.")
    
    return monitor

if __name__ == "__main__":
    monitor = demo_global_monitoring()