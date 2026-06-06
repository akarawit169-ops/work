"""
UNIFIED FRAUD DETECTION PIPELINE — HIGH-PERFORMANCE PRODUCTION EDITION
======================================================================
Combines Phase 1, 2, and 3 into an optimized, big-data safe pipeline:
  ✓ 20 clean behavioral features (Vectorized via Pandas/NumPy)
  ✓ Scikit-Learn MLP Autoencoder backbone (Replaces slow custom loops)
  ✓ Fast PageRank Structural Graph Scoring (Removed heavy Betweenness Centrality)
  ✓ Time-decay weighting & Production-ready output handling

Run: python fraud_detection_unified_clean.py
Out: fraud_output_final.json
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random, json, os, warnings
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
import networkx as nx

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)

print("="*75)
print("FRAUD DETECTION MODEL — UNIFIED PIPELINE (OPTIMIZED PRODUCTION)")
print("="*75)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA INGESTION (Optimized Mock Data Setup)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/7] Loading data...")

DEPTS = ['Retail Banking','Operations','Customer Service','Compliance','IT Support']
POSITIONS = ['Teller','Officer','Supervisor','Manager','Analyst']
BRANCHES = [f'BR{i:03d}' for i in range(1,11)]
APP_TYPES = ['balance','passbook','statement']
ACC_TYPES = ['Savings','Current','Fixed Deposit','Loan']
ACC_STAT = ['Active','Dormant','Closed','Frozen']

N_EMP, N_CUST, N_LOGS = 100, 5000, 1000000

employees = pd.DataFrame({
    'employee_id': [f'EMP{i:04d}' for i in range(1, N_EMP+1)],
    'dept_name': np.random.choice(DEPTS, N_EMP),
    'position': np.random.choice(POSITIONS, N_EMP),
    'branch_number': np.random.choice(BRANCHES, N_EMP),
    'shift_start': np.random.choice([7,8,9,10], N_EMP),
})
employees['shift_end'] = employees['shift_start'] + 8

customers = pd.DataFrame({
    'acct_no': [f'ACC{i:06d}' for i in range(1, N_CUST+1)],
    'account_type': np.random.choice(ACC_TYPES, N_CUST, p=[.5, .3, .15, .05]),
    'current_balance': np.random.lognormal(10, 2, N_CUST).round(2),
    'account_status': np.random.choice(ACC_STAT, N_CUST, p=[.7, .1, .1, .1]),
    'customer_type': np.random.choice(['personal','company'], N_CUST, p=[.8, .2]),
})

# Vectorized Log Generation
emp_idx = np.random.randint(0, N_EMP, N_LOGS)
cust_idx = np.random.randint(0, N_CUST, N_LOGS)
days_offset = np.random.randint(0, 180, N_LOGS)
hours_offset = np.random.randint(0, 24, N_LOGS)
base_date = datetime.now()
log_dates = base_date - pd.to_timedelta(days_offset, unit='D') - pd.to_timedelta(hours_offset, unit='h')

df = pd.DataFrame({
    'log_id': [f'LOG{i:06d}' for i in range(N_LOGS)],
    'log_date': log_dates,
    'employee_id': employees['employee_id'].values[emp_idx],
    'dept_name': employees['dept_name'].values[emp_idx],
    'position': employees['position'].values[emp_idx],
    'branch_number': employees['branch_number'].values[emp_idx],
    'shift_start': employees['shift_start'].values[emp_idx],
    'shift_end': employees['shift_end'].values[emp_idx],
    'acct_no': customers['acct_no'].values[cust_idx],
    'app_desc': np.random.choice(APP_TYPES, N_LOGS),
    'current_balance': customers['current_balance'].values[cust_idx],
    'account_type': customers['account_type'].values[cust_idx],
    'account_status': customers['account_status'].values[cust_idx],
    'customer_type': customers['customer_type'].values[cust_idx],
})

# Inject Synthetic Fraud Patterns (Vectorized)
hb_idx = np.random.choice(N_LOGS, 300, replace=False)
top_cust = customers.nlargest(50, 'current_balance')
df.loc[hb_idx, 'acct_no'] = np.random.choice(top_cust['acct_no'], 300)
df.loc[hb_idx, 'current_balance'] = customers.set_index('acct_no').loc[df.loc[hb_idx, 'acct_no'], 'current_balance'].values

oh_idx = np.random.choice(N_LOGS, 300, replace=False)
df.loc[oh_idx, 'log_date'] = df.loc[oh_idx, 'log_date'].apply(lambda x: x.replace(hour=random.randint(0,4)))

df['log_date'] = pd.to_datetime(df['log_date'])
df = df.sort_values(['employee_id', 'log_date']).reset_index(drop=True)

print(f"   Generated {len(df):,} logs from {df['employee_id'].nunique()} employees")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: FEATURE ENGINEERING (Vectorized & Scaled for Performance)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/7] Engineering 20 clean features (Vectorized Engine)...")

df['hour'] = df['log_date'].dt.hour
df['is_weekend'] = (df['log_date'].dt.dayofweek >= 5).astype(int)
df['is_off_hours'] = ((df['hour'] < 8) | (df['hour'] > 18)).astype(int)
df['is_after_midnight'] = (df['hour'] < 5).astype(int)

# 1. Temporal Personalization
df['cum_hour_mean'] = df.groupby('employee_id')['hour'].transform(lambda x: x.expanding().mean())
df['cum_hour_std'] = df.groupby('employee_id')['hour'].transform(lambda x: x.expanding().std()).fillna(0)
df['hour_zscore'] = ((df['hour'] - df['cum_hour_mean']) / (df['cum_hour_std'] + 1)).clip(-5, 5)

df['cum_weekend_avg'] = df.groupby('employee_id')['is_weekend'].transform(lambda x: x.expanding().mean())
df['weekend_surprise'] = (df['is_weekend'] - df['cum_weekend_avg']).clip(0, 1)

df['is_dormant_access'] = df['account_status'].isin(['Dormant','Closed','Frozen']).astype(int)
df['days_ago'] = (datetime.now() - df['log_date']).dt.days.clip(lower=0)

# 2. Vectorized Burst Velocity
df = df.set_index('log_date')
df['burst_5min'] = df.groupby('employee_id')['log_id'].rolling('5min').count().reset_index(level=0, drop=True)
df['accts_last_hour'] = df.groupby('employee_id')['log_id'].rolling('60min').count().reset_index(level=0, drop=True)
df = df.reset_index()

# 3. Workload Spikes
df['date_only'] = df['log_date'].dt.date
df['day_count_so_far'] = df.groupby(['employee_id', 'date_only']).cumcount() + 1
daily_counts = df.groupby(['employee_id', 'date_only']).size().rename('daily_total').reset_index()
daily_counts['hist_avg_workload'] = daily_counts.groupby('employee_id')['daily_total'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)
daily_counts['hist_std_workload'] = daily_counts.groupby('employee_id')['daily_total'].transform(lambda x: x.shift(1).expanding().std()).fillna(0)
df = df.merge(daily_counts[['employee_id', 'date_only', 'hist_avg_workload', 'hist_std_workload']], on=['employee_id', 'date_only'], how='left')
df['workload_zscore'] = ((df['day_count_so_far'] - df['hist_avg_workload']) / (df['hist_std_workload'] + 1)).clip(-5, 5)

# 4. Access Patterns
df['emp_acct_total'] = df.groupby(['employee_id', 'acct_no']).cumcount() + 1
df['first_acct_date'] = df.groupby(['employee_id', 'acct_no'])['log_date'].transform('first')
df['emp_acct_days_span'] = (df['log_date'] - df['first_acct_date']).dt.days + 1
df['emp_acct_freq_per_day'] = df['emp_acct_total'] / df['emp_acct_days_span']
df['acct_unique_employees'] = df.groupby('acct_no')['employee_id'].transform(lambda x: (~x.duplicated()).cumsum())

# 5. Financial Surprise
df['cum_bal_mean'] = df.groupby('employee_id')['current_balance'].transform(lambda x: x.expanding().mean())
df['cum_bal_std'] = df.groupby('employee_id')['current_balance'].transform(lambda x: x.expanding().std()).fillna(0)
df['balance_surprise_zscore'] = ((df['current_balance'] - df['cum_bal_mean']) / (df['cum_bal_std'] + 1)).clip(-5, 5)

df['time_since_last_log_sec'] = df.groupby('employee_id')['log_date'].diff().dt.total_seconds().fillna(0)
df['time_since_acct_accessed_sec'] = df.groupby('acct_no')['log_date'].diff().dt.total_seconds().fillna(0)

# 6. Escalation Triggers
APP_SENSITIVITY = {'balance': 3, 'statement': 2, 'passbook': 1}
df['app_sensitivity'] = df['app_desc'].map(APP_SENSITIVITY).fillna(1)
df['prev_app_desc'] = df.groupby('employee_id')['app_desc'].shift(1).fillna('none')
df['is_repeated_app'] = (df['app_desc'] == df['prev_app_desc']).astype(int)
df['rolling_sensitivity_avg'] = df.groupby('employee_id')['app_sensitivity'].shift(1).rolling(2, min_periods=1).mean().fillna(1)
df['is_escalating'] = (df['app_sensitivity'] > df['rolling_sensitivity_avg']).astype(int)
df['escalation_delta'] = (df['app_sensitivity'] - df['rolling_sensitivity_avg']).clip(lower=0)

le = LabelEncoder()
df['app_enc'] = le.fit_transform(df['app_desc'].astype(str))

FEATURES = [
    'hour_zscore', 'weekend_surprise', 'is_off_hours', 'is_after_midnight', 'burst_5min', 'accts_last_hour',
    'time_since_last_log_sec', 'time_since_acct_accessed_sec', 'workload_zscore', 'emp_acct_total',
    'emp_acct_days_span', 'emp_acct_freq_per_day', 'acct_unique_employees', 'is_dormant_access',
    'balance_surprise_zscore', 'app_sensitivity', 'is_repeated_app', 'is_escalating', 'escalation_delta', 'app_enc'
]
X_all = df[FEATURES].fillna(0)
print(f"   Features engineered successfully: {len(FEATURES)}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: FEEDBACK LOOP
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/7] Loading feedback...")
df['feedback_label'] = 'unlabeled'
X_train_feedback = X_all

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: HIGH-PERFORMANCE AUTOENCODER
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/7] Training peer-clustered native autoencoders...")
df['peer_group'] = df['dept_name'] + ' | ' + df['position']
ae_scaler = MinMaxScaler()
X_scaled_all = ae_scaler.fit_transform(X_all)

df['ae_score'] = 0.0
for group in df['peer_group'].unique():
    mask = df['peer_group'] == group
    X_grp_tr = X_scaled_all[mask]
    if len(X_grp_tr) >= 50:
        ae = MLPRegressor(hidden_layer_sizes=(24, 12, 24), activation='relu', solver='adam', 
                          max_iter=10, batch_size=256, random_state=42)
        ae.fit(X_grp_tr, X_grp_tr)
        preds = ae.predict(X_grp_tr)
        errors = np.mean((X_grp_tr - preds) ** 2, axis=1)
        thresh = np.percentile(errors, 90)
        mx = errors.max()
        normed = np.where(errors >= thresh, ((errors - thresh) / max((mx - thresh), 1e-9) * 100), 0)
        df.loc[mask, 'ae_score'] = normed.round(1)

print(f"   Autoencoders completed.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: RAPID ENSEMBLE ENGINE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/7] Executing Parallel Ensemble (IF + DBSCAN + SVM)...")
gs = StandardScaler()
Xgs = gs.fit_transform(X_all)

g_if = IsolationForest(n_estimators=50, contamination=0.02, random_state=42, n_jobs=-1)
if_raw = g_if.fit_predict(Xgs)
df['if_score'] = np.where(if_raw == -1, 100.0, 0.0)

db_sample_idx = np.random.choice(len(Xgs), min(50000, len(Xgs)), replace=False)
g_db = DBSCAN(eps=2.5, min_samples=50, n_jobs=-1)
db_preds = g_db.fit_predict(Xgs[db_sample_idx])
df['dbscan_score'] = 20.0
df.loc[df.index[db_sample_idx[db_preds == -1]], 'dbscan_score'] = 85.0

g_svm = OneClassSVM(kernel='rbf', nu=0.02, gamma='scale')
svm_train_sample = Xgs[np.random.choice(len(Xgs), min(5000, len(Xgs)), replace=False)]
g_svm.fit(svm_train_sample)
svm_raw = g_svm.score_samples(Xgs)
df['svm_score'] = ((svm_raw - svm_raw.min()) / max((svm_raw.max() - svm_raw.min()), 1e-9) * 100).clip(0,100).round(1)

print(f"   Ensemble calculations processed.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: LIGHTWEIGHT STRUCTURAL GRAPH SCORING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6/7] Extracting topological graph risks...")
G = nx.Graph()
edge_counts = df.groupby(['employee_id','acct_no']).size().reset_index(name='weight')
G.add_weighted_edges_from(edge_counts[['employee_id', 'acct_no', 'weight']].values)
pr = nx.pagerank(G, weight='weight', alpha=0.85, max_iter=100)
emp_nodes = employees['employee_id'].values
pr_vals = np.array([pr.get(eid, 0) for eid in emp_nodes])
pr_min, pr_max = pr_vals.min(), pr_vals.max()
graph_scores = {eid: float((pr.get(eid, 0) - pr_min) / max((pr_max - pr_min), 1e-9) * 100) for eid in emp_nodes}
df['graph_score'] = df['employee_id'].map(graph_scores).fillna(0)
df['graph_score'] = np.where(df['graph_score'] < 80, 0, (df['graph_score'] - 80) * 5)
print(f"   Graph completed.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: TIME DECAY, AGGREGATION & BULK JSON GENERATION
# ══════════════════════════════════════════════════════════════════════════════

df['ensemble_score'] = (0.50 * df['ae_score'].fillna(0) + 0.30 * df['if_score'] + 0.20 * df['graph_score']).round(1)
df['risk_tier'] = pd.cut(df['ensemble_score'], bins=[-1, 30, 60, 80, 101], labels=['Low','Medium','High','Critical'])

def get_flags(row):
    f = []
    if row['is_off_hours']: f.append('Off-hours')
    if row['is_after_midnight']: f.append('After midnight')
    if row['hour_zscore'] > 2.5: f.append('Time Anomaly')
    if row['weekend_surprise'] > 0.7: f.append('Weekend Surprise')
    if row['burst_5min'] > 3: f.append(f"Burst ({int(row['burst_5min'])} in 5min)")
    if row['is_dormant_access']: f.append('Dormant/Closed')
    if row['is_escalating']: f.append('Escalation')
    if row['balance_surprise_zscore'] > 2.5: f.append('Balance Surprise')
    if row['workload_zscore'] > 2.5: f.append('Volume Spike')
    return f

# 10. Local Explainability (Top Alert Reasons)
top_alerts_idx = df.nlargest(100, 'ensemble_score').index
dormant_alerts_idx = df[df['is_dormant_access'] == 1].nlargest(100, 'ensemble_score').index
explain_idx = top_alerts_idx.union(dormant_alerts_idx)

alert_reasons = {}
for idx in explain_idx:
    row = df.loc[idx]
    reasons = []
    if row['is_off_hours']: reasons.append("Off-hours Access")
    if row['burst_5min'] > 5: reasons.append(f"High Velocity ({int(row['burst_5min'])} logs/5m)")
    if row['balance_surprise_zscore'] > 2.0: reasons.append("Targeting High Balance")
    if row['hour_zscore'] > 2.0: reasons.append("Time Anomaly")
    if row['is_dormant_access']: reasons.append("Dormant Account Access")
    if row['workload_zscore'] > 2.0: reasons.append("Volume Spike")
    if row['is_escalating']: reasons.append("Sensitivity Escalation")
    if row['accts_last_hour'] > 20: reasons.append("Wide Account Scouting")
    alert_reasons[row['log_id']] = ", ".join(reasons[:3]) if reasons else "Multiple Behavioral Anomalies"

print("\n[7/7] Computing final matrix blends and building output objects...")

# 2. Risk Trends
df['date_str'] = df['log_date'].dt.strftime('%Y-%m-%d')
trend_data = df.groupby(['employee_id', 'date_str'])['ensemble_score'].mean().reset_index()
risk_trend = {}
for eid in trend_data['employee_id'].unique():
    risk_trend[eid] = trend_data[trend_data['employee_id'] == eid][['date_str', 'ensemble_score']].rename(columns={'date_str':'date', 'ensemble_score':'avg_risk'}).to_dict(orient='records')

# 4. Account Network
net_df = df.nlargest(500, 'ensemble_score').copy()
net_df['flags'] = net_df.apply(get_flags, axis=1)
nodes, links, added_nodes = [], [], set()
for _, row in net_df.iterrows():
    if row['employee_id'] not in added_nodes:
        nodes.append({'id': row['employee_id'], 'type': 'employee', 'dept': row['dept_name'], 'risk': float(row['ensemble_score'])})
        added_nodes.add(row['employee_id'])
    if row['acct_no'] not in added_nodes:
        nodes.append({'id': row['acct_no'], 'type': 'account', 'status': row['account_status'], 'balance': float(row['current_balance']), 'access_count': int(df[df['acct_no']==row['acct_no']].shape[0])})
        added_nodes.add(row['acct_no'])
    links.append({'source': row['employee_id'], 'target': row['acct_no'], 'weight': 1})

# 5. Dormant Alerts
dormant_alerts = df[df['is_dormant_access'] == 1].nlargest(100, 'ensemble_score').copy()
dormant_alerts['reason'] = dormant_alerts['log_id'].map(alert_reasons).fillna('Dormant Account Access')
dormant_alerts['log_date_str'] = dormant_alerts['log_date'].dt.strftime('%Y-%m-%d %H:%M')
dormant_alerts['flags'] = dormant_alerts.apply(get_flags, axis=1)
dormant_alerts.rename(columns={'ensemble_score': 'risk_score'}, inplace=True)

# 6. Timeline Data (Optimized for On-Demand Loading)
print("\n[6/7] Splitting timelines into separate files...")
TL_DIR = 'timelines'
if not os.path.exists(TL_DIR): os.makedirs(TL_DIR)
df_timeline = df.copy()
df_timeline['log_date_str'] = df_timeline['log_date'].dt.strftime('%Y-%m-%d %H:%M')
df_timeline['is_burst'] = (df_timeline['burst_5min'] > 3).astype(int)
for eid in df_timeline['employee_id'].unique():
    emp_logs = df_timeline[df_timeline['employee_id'] == eid].sort_values('log_date', ascending=False).copy()
    emp_logs['flags'] = emp_logs.apply(get_flags, axis=1)
    timeline_data = emp_logs[['log_id','log_date_str','acct_no','ensemble_score','flags','app_desc','current_balance','account_status','risk_tier','is_dormant_access','is_burst']].rename(columns={'log_date_str':'log_date', 'ensemble_score':'risk_score'}).to_dict(orient='records')
    with open(f'{TL_DIR}/{eid}.json', 'w') as f_tl:
        json.dump(timeline_data, f_tl, default=str)

# 7. Aggregates
df['decay_weight'] = np.exp(-0.03 * df['days_ago'])
df['weighted_score_part'] = df['ensemble_score'] * df['decay_weight']
decay_agg = df.groupby('employee_id').agg({'weighted_score_part': 'sum', 'decay_weight': 'sum'})
decay_scores = (decay_agg['weighted_score_part'] / decay_agg['decay_weight'].clip(lower=1e-9)).round(1).rename('weighted_avg_risk')

emp_base = df.groupby('employee_id').agg(
    dept_name=('dept_name','first'), position=('position','first'), branch_number=('branch_number','first'), total_logs=('log_id','count'),
    max_score=('ensemble_score','max'), unique_accounts=('acct_no','nunique'),
    high_risk_events=('risk_tier', lambda x: x.isin(['High','Critical']).sum()),
    time_anomalies=('hour_zscore', lambda x: (x > 2.5).sum()),
    burst_events=('burst_5min', lambda x: (x > 3).sum()),
    dormant_events=('is_dormant_access','sum'),
    volume_anomalies=('workload_zscore', lambda x: (x > 2.5).sum()),
    avg_risk_score=('ensemble_score','mean'),
    most_recent_log=('log_date','max')
).reset_index()

emp_risk = emp_base.merge(decay_scores, on='employee_id', how='left').fillna(0)
emp_risk['composite_risk'] = emp_risk['weighted_avg_risk']
emp_risk = emp_risk.sort_values('composite_risk', ascending=False).reset_index(drop=True)
emp_risk['rank'] = emp_risk.index + 1

top_alerts = df.nlargest(100, 'ensemble_score').copy()
top_alerts['flags'] = top_alerts.apply(get_flags, axis=1)
top_alerts['reason'] = top_alerts['log_id'].map(alert_reasons).fillna('Complex Pattern')
top_alerts['log_date_str'] = top_alerts['log_date'].dt.strftime('%Y-%m-%d %H:%M')
top_alerts.rename(columns={'ensemble_score': 'risk_score'}, inplace=True)

# 8. Top Risk Triggers (Vectorized)
risk_counts = {
    'Off-hours': int(df['is_off_hours'].sum()),
    'After midnight': int(df['is_after_midnight'].sum()),
    'Time Anomaly': int((df['hour_zscore'] > 2.5).sum()),
    'Weekend Surprise': int((df['weekend_surprise'] > 0.7).sum()),
    'Burst': int((df['burst_5min'] > 3).sum()),
    'Dormant/Closed': int(df['is_dormant_access'].sum()),
    'Escalation': int(df['is_escalating'].sum()),
    'Balance Surprise': int((df['balance_surprise_zscore'] > 2.5).sum()),
    'Volume Spike': int((df['workload_zscore'] > 2.5).sum()),
}
top_triggers = dict(sorted(risk_counts.items(), key=lambda x: x[1], reverse=True)[:5])

# 9. Feature Importance (Surrogate Explainer)
print("\n[8/7] Analyzing global feature importance...")
sample_idx = np.random.choice(len(X_all), min(20000, len(X_all)), replace=False)
X_sample = X_all.iloc[sample_idx]
y_sample = df.loc[df.index[sample_idx], 'ensemble_score']
explainer = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
explainer.fit(X_sample, y_sample)
importances = dict(zip(FEATURES, explainer.feature_importances_))
feature_importance = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

hourly_avg = df.groupby('hour')['ensemble_score'].mean().round(1).to_dict()
hourly_pattern = {int(k): float(v) for k, v in hourly_avg.items()}

output_payload = {
    'summary': {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'total_logs': len(df),
        'total_employees': df['employee_id'].nunique(),
        'total_accounts': df['acct_no'].nunique(),
        'critical_events': int((df['risk_tier']=='Critical').sum()),
        'high_events': int((df['risk_tier']=='High').sum()),
        'burst_events': int((df['burst_5min'] > 3).sum()),
        'dormant_events': int(df['is_dormant_access'].sum()),
        'features_used': len(FEATURES),
        'status': 'Optimized for Scaled Environments'
    },
    'risk_distribution': {str(k): int(v) for k, v in df['risk_tier'].value_counts().to_dict().items()},
    'hourly_risk_pattern': hourly_pattern,
    'top_triggers': top_triggers,
    'feature_importance': feature_importance,
    'risk_trend': risk_trend,
    'account_network': {'nodes': nodes, 'links': links},
    'top_risky_employees': emp_risk.head(20).to_dict(orient='records'),
    'top_alerts': top_alerts[['log_id','log_date_str','employee_id','acct_no','risk_score','reason','flags','risk_tier','dept_name','position','branch_number','app_desc','current_balance','account_status']].rename(columns={'log_date_str':'log_date'}).to_dict(orient='records'),
    'dormant_alerts': dormant_alerts[['log_id','log_date_str','employee_id','acct_no','risk_score','reason','flags','risk_tier','dept_name','position','branch_number','app_desc','current_balance','account_status']].rename(columns={'log_date_str':'log_date'}).to_dict(orient='records')
}

with open('fraud_output_final.json', 'w') as f:
    json.dump(output_payload, f, default=str, indent=2)

print(f"\n✓ Output successfully compiled: fraud_output_final.json")
print("="*75)
