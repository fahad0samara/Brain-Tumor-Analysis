from datetime import datetime, timedelta
import random

def generate_sample_records():
    """Generate sample patient records for demonstration"""
    sample_patients = [
        {
            "name": "John Smith",
            "records": 5,
            "base_age": 45,
            "base_tumor_size": 2.5,
            "base_genetic_risk": 6,
            "base_survival_rate": 75
        },
        {
            "name": "Sarah Johnson",
            "records": 3,
            "base_age": 62,
            "base_tumor_size": 3.2,
            "base_genetic_risk": 8,
            "base_survival_rate": 60
        },
        {
            "name": "Michael Chen",
            "records": 4,
            "base_age": 35,
            "base_tumor_size": 1.8,
            "base_genetic_risk": 4,
            "base_survival_rate": 85
        },
        {
            "name": "Emma Davis",
            "records": 6,
            "base_age": 52,
            "base_tumor_size": 2.8,
            "base_genetic_risk": 7,
            "base_survival_rate": 70
        }
    ]
    
    all_records = []
    base_date = datetime.now() - timedelta(days=180)  # Start 6 months ago
    
    for patient in sample_patients:
        for visit in range(patient["records"]):
            # Add some random variation to measurements
            record = {
                "name": patient["name"],
                "age": patient["base_age"] + random.uniform(-1, 1),
                "gender": random.choice(["Male", "Female"]),
                "tumor_size": max(0.1, patient["base_tumor_size"] + random.uniform(-0.5, 0.5)),
                "genetic_risk": max(1, min(10, patient["base_genetic_risk"] + random.uniform(-1, 1))),
                "survival_rate": max(1, min(100, patient["base_survival_rate"] + random.uniform(-5, 5))),
                "smoking": random.choice([True, False]),
                "alcohol": random.choice([True, False]),
                "family_history": random.choice([True, False]),
                "notes": f"Follow-up visit {visit + 1}",
                "timestamp": (base_date + timedelta(days=30 * visit)).isoformat()
            }
            
            # Calculate risk score
            record["Risk_Score"] = (
                record["genetic_risk"] * 0.4 +
                record["age"]/100 * 0.3 +
                record["tumor_size"]/10 * 0.3
            )
            
            # Calculate medical complexity
            record["Medical_Complexity"] = sum([
                record["genetic_risk"] > 7,
                record["age"] > 50,
                record["tumor_size"] > 3,
                record["survival_rate"] < 50,
                record["smoking"],
                record["alcohol"],
                record["family_history"]
            ])
            
            # Calculate prediction and probability based on risk factors
            base_prob = (
                (record["genetic_risk"] / 10) * 0.3 +
                (record["tumor_size"] / 5) * 0.3 +
                (1 - record["survival_rate"] / 100) * 0.2 +
                (record["age"] / 100) * 0.2
            )
            record["probability"] = max(0.01, min(0.99, base_prob + random.uniform(-0.1, 0.1)))
            record["prediction"] = 1 if record["probability"] > 0.5 else 0
            
            all_records.append(record)
    
    return all_records
