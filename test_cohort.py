import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000

actual_positive = np.random.choice([0, 1], p=[0.8, 0.2], size=n)

ai_prob = np.where(
    actual_positive == 1,
    np.random.beta(a=6, b=3, size=n),  # Mean ~0.66 (yields misses for traditional)
    np.random.beta(a=2, b=8, size=n)   # Mean ~0.2
)

trad_pred = (ai_prob > 0.5).astype(int)

ai_uncertainty = np.where(
    ((actual_positive == 1) & (ai_prob < 0.5)) | ((actual_positive == 0) & (ai_prob > 0.5)),
    np.random.uniform(0.3, 0.6, size=n), 
    np.random.uniform(0.0, 0.2, size=n)  
)

severity = np.where(
    actual_positive == 1,
    np.random.uniform(3.0, 5.0, size=n),
    np.random.uniform(1.0, 2.5, size=n)
)

data = pd.DataFrame({
    "Patient_ID": range(n),
    "Actual_Positive": actual_positive, 
    "AI_Prob": ai_prob,
    "AI_Uncertainty": ai_uncertainty,
    "Severity": severity,
    "Trad_Pred": trad_pred
})

trad_missed = len(data[(data['Actual_Positive'] == 1) & (data['Trad_Pred'] == 0)])
trad_false_alerts = len(data[(data['Actual_Positive'] == 0) & (data['Trad_Pred'] == 1)])
trad_review_load = len(data[data['Trad_Pred'] == 1])

data['Collab_Risk'] = data['AI_Prob'] * data['Severity']
data['Collab_Reliability'] = 0.9 * (1 - data['AI_Uncertainty']) 
data['Collab_Adjusted_Risk'] = data['Collab_Risk'] + (1 - data['Collab_Reliability']) * 0.5

T1, T2 = 3.5, 1.8
data['Collab_Priority'] = 'Routine'
data['Collab_Priority'] = np.where(data['Collab_Adjusted_Risk'] >= T2, 'Review', data['Collab_Priority'])
data['Collab_Priority'] = np.where(data['Collab_Adjusted_Risk'] >= T1, 'Urgent', data['Collab_Priority'])

collab_missed = len(data[(data['Actual_Positive'] == 1) & (data['Collab_Priority'] == 'Routine')])
collab_false_alerts = len(data[(data['Actual_Positive'] == 0) & (data['Collab_Priority'] == 'Urgent')])
collab_review_load = len(data[data['Collab_Priority'] != 'Routine'])

print(f"Trad Missed: {trad_missed}, Collab Missed: {collab_missed}")
print(f"Trad False Alerts: {trad_false_alerts}, Collab False Alerts: {collab_false_alerts}")
print(f"Trad Load (Review + Urgent): {trad_review_load}, Collab Load: {collab_review_load}")
print(f"Missed Reduction: {((trad_missed - collab_missed)/trad_missed)*100}%")
