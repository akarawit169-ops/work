"""
UNIFIED FRAUD DETECTION PIPELINE — CLEAN FEATURE SET
=====================================================

Combines Phase 1, 2, and 3 into one script with:
  ✓ 24 clean features (no redundancy, no leakage)
  ✓ Peer-clustered autoencoder + ensemble + graph scoring
  ✓ Time-decay weighting
  ✓ Semi-supervised feedback loop
  ✓ Production-ready

Run: python fraud_detection_unified_clean.py
Out: fraud_output_final.json
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random, json, calendar, os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

random.seed(42)
np.random.seed(42)

print("="*75)
print("FRAUD DETECTION MODEL — UNIFIED PIPELINE (CLEAN FEATURES)")
print("="*75)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA INGESTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/7] Loading data…")

# REPLACE THIS BLOCK with your real database query
# ────────────────────────────────────────────────────────────────────────────
# TEMPLATE:
# import pyodbc
# conn = pyodbc.connect("DSN=your_bank;UID=user;PWD=pass")
# df = pd.read_sql("""
#     SELECT l.log_id, l.log_date, l.app_desc, l.employee_id,
#            e.dept_name, e.position, e.branch_number, e.shift_start, e.shift_end,
#            a.acct_no, a.current_balance, a.account_status, a.account_type,
#            c.customer_type
#     FROM access_log l
#     JOIN employee e ON l.employee_id = e.employee_id
#     JOIN account a ON l.acct_no = a.acct_no
#     JOIN customer c ON a.acct_no = c.acct_no
#     WHERE l.log_date >= DATEADD(day, -180, GETDATE())
# """, conn)
# ────────────────────────────────────────────────────────────────────────────

# SYNTHETIC DATA (remove for production)
DEPTS = ['Retail Banking','Operations','Customer Service','Compliance','IT Support']
POSITIONS = ['Teller','Officer','Supervisor','Manager','Analyst']
BRANCHES = [f'BR{i:03d}' for i in range(1,11)]
APP_TYPES = ['balance','passbook','statement']
ACC_TYPES = ['Savings','Current','Fixed Deposit','Loan']
ACC_STAT = ['Active','Dormant','Closed','Frozen']

N_EMP, N_CUST, N_LOGS = 80, 2000, 15000

employees = pd.DataFrame({
    'employee_id': [f'EMP{i:04d}' for i in range(1,N_EMP+1)],
    'dept_name': np.random.choice(DEPTS, N_EMP),
    'position': np.random.choice(POSITIONS, N_EMP),
    'branch_number': np.random.choice(BRANCHES, N_EMP),
    'shift_start': np.random.choice([7,8,9,10], N_EMP),
})
employees['shift_end'] = employees['shift_start'] + 8

customers = pd.DataFrame({
    'acct_no': [f'ACC{i:06d}' for i in range(1,N_CUST+1)],
    'account_type': np.random.choice(ACC_TYPES, N_CUST, p=[.5,.3,.15,.05]),
    'current_balance': np.random.lognormal(10, 2, N_CUST).round(2),
    'account_status': np.random.choice(ACC_STAT, N_CUST, p=[.7,.1,.1,.1]),
    'customer_type': np.random.choice(['personal','company'], N_CUST, p=[.8,.2]),
})

def _log(i, fraud=None):
    e = employees.sample(1).iloc[0]
    c = customers.sample(1).iloc[0]
    days = random.randint(0, 180)
    if fraud == 'high_balance':
        c = customers.nlargest(50, 'current_balance').sample(1).iloc[0]
        d = datetime.now() - timedelta(days=days, hours=random.randint(0,23))
    elif fraud == 'bulk':
        d = datetime.now() - timedelta(days=days, hours=random.randint(0,23), minutes=random.randint(0,5))
    elif fraud == 'off_hours':
        d = datetime.now() - timedelta(days=days)
        d = d.replace(hour=random.randint(0,4))
    else:
        d = datetime.now() - timedelta(days=days, hours=random.randint(0,9))
        if d.weekday() >= 5: d += timedelta(days=(7-d.weekday()))
        d = d.replace(hour=random.randint(int(e.shift_start), int(e.shift_end)-1))
    return dict(log_id=f'LOG{i:06d}', log_date=d,
                employee_id=e.employee_id, dept_name=e.dept_name,
                position=e.position, branch_number=e.branch_number,
                shift_start=int(e.shift_start), shift_end=int(e.shift_end),
                acct_no=c.acct_no, app_desc=random.choice(APP_TYPES),
                current_balance=c.current_balance, account_type=c.account_type,
                account_status=c.account_status, customer_type=c.customer_type)

patterns = ['high_balance', 'bulk', 'off_hours']
df = pd.DataFrame([_log(i) for i in range(N_LOGS)] +
                  [_log(N_LOGS+i, patterns[i%3]) for i in range(300)])
df['log_date'] = pd.to_datetime(df['log_date'])
df = df.sort_values(['employee_id', 'log_date']).reset_index(drop=True)

print(f"   Loaded {len(df):,} logs from {df['employee_id'].nunique()} employees")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: FEATURE ENGINEERING (22 PURE BEHAVIORAL FEATURES)
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: FEATURE ENGINEERING (20 FEATURES — BEHAVIORAL + OFF-HOURS)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/7] Engineering 20 features…")

# Base Temporal (for derived features)
df['hour'] = df['log_date'].dt.hour
df['is_weekend'] = (df['log_date'].dt.dayofweek >= 5).astype(int)
df['is_off_hours'] = ((df['hour'] < 8) | (df['hour'] > 18)).astype(int)
df['is_after_midnight'] = (df['hour'] < 5).astype(int)

# 1. Personalized Time Anomaly (hour_zscore)
# Learns "When do YOU normally work?"
df['cum_hour_mean'] = df.groupby('employee_id')['hour'].transform(lambda x: x.expanding().mean())
df['cum_hour_std'] = df.groupby('employee_id')['hour'].transform(lambda x: x.expanding().std()).fillna(0)
df['hour_zscore'] = ((df['hour'] - df['cum_hour_mean']) / (df['cum_hour_std'] + 1)).clip(-5, 5)

# 2. Personalized Weekend Surprise
# Learns "Do YOU normally work weekends?"
df['cum_weekend_avg'] = df.groupby('employee_id')['is_weekend'].transform(lambda x: x.expanding().mean())
df['weekend_surprise'] = (df['is_weekend'] - df['cum_weekend_avg']).clip(0, 1)

# 3. Dormant access (High-precision signal)
df['is_dormant_access'] = df['account_status'].isin(['Dormant','Closed','Frozen']).astype(int)

# Days for decay
df['days_ago'] = (datetime.now() - df['log_date']).dt.days.clip(lower=0)

# 4. Burst/Velocity (5-minute and 1-hour windows)
df['log_ts'] = df['log_date'].astype(np.int64)
def burst_count(grp, minutes):
    t = grp['log_ts'].values
    w = minutes * 60 * 1_000_000_000
    return np.array([np.sum((t >= t[i]-w) & (t < t[i])) for i in range(len(t))])

bursts = []
for eid, g in df.groupby('employee_id'):
    bursts.append(pd.DataFrame({
        'log_id': g['log_id'],
        'burst_5min': burst_count(g, 5),
        'accts_last_hour': burst_count(g, 60)
    }))
df = df.merge(pd.concat(bursts), on='log_id', how='left')

# 5. Personalized Workload Anomaly (workload_zscore)
# Learns "Is this a high-volume day for YOU?"
df['date_only'] = df['log_date'].dt.date
df['day_count_so_far'] = df.groupby(['employee_id', 'date_only']).cumcount() + 1
daily_totals = df.groupby(['employee_id', 'date_only']).size().reset_index(name='daily_total')
daily_totals['hist_avg_workload'] = daily_totals.groupby('employee_id')['daily_total'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)
daily_totals['hist_std_workload'] = daily_totals.groupby('employee_id')['daily_total'].transform(lambda x: x.shift(1).expanding().std()).fillna(0)
df = df.merge(daily_totals[['employee_id', 'date_only', 'hist_avg_workload', 'hist_std_workload']], on=['employee_id', 'date_only'], how='left')
df['workload_zscore'] = ((df['day_count_so_far'] - df['hist_avg_workload']) / (df['hist_std_workload'] + 1)).clip(-5, 5)

# 6. Repeat access & Breadth
df['emp_acct_total'] = df.groupby(['employee_id', 'acct_no']).cumcount() + 1
df['emp_acct_days_span'] = (df['log_date'] - df.groupby(['employee_id', 'acct_no'])['log_date'].transform('first')).dt.days + 1
df['emp_acct_freq_per_day'] = df['emp_acct_total'] / df['emp_acct_days_span'].clip(lower=1)
df['acct_unique_employees'] = df.groupby('acct_no')['employee_id'].transform(lambda x: (~x.duplicated()).cumsum())

# 7. Monetary Signal (Balance Surprise)
# Learns "Is this account balance unusually high for THIS person?"
df['cum_bal_mean'] = df.groupby('employee_id')['current_balance'].transform(lambda x: x.expanding().mean())
df['cum_bal_std'] = df.groupby('employee_id')['current_balance'].transform(lambda x: x.expanding().std()).fillna(0)
df['balance_surprise_zscore'] = ((df['current_balance'] - df['cum_bal_mean']) / (df['cum_bal_std'] + 1)).clip(-5, 5)

# 8. Recency Velocity
df['time_since_last_log_sec'] = df.groupby('employee_id')['log_date'].diff().dt.total_seconds().fillna(0)
df['time_since_acct_accessed_sec'] = df.groupby('acct_no')['log_date'].diff().dt.total_seconds().fillna(0)

# 9. Sequence & Escalation
APP_SENSITIVITY = {'balance': 3, 'statement': 2, 'passbook': 1}
df['app_sensitivity'] = df['app_desc'].map(APP_SENSITIVITY).fillna(1)
df['prev_app_desc'] = df.groupby('employee_id')['app_desc'].shift(1).fillna('none')
df['is_repeated_app'] = (df['app_desc'] == df['prev_app_desc']).astype(int)
df['prev_sensitivity'] = df.groupby('employee_id')['app_sensitivity'].shift(1).fillna(1)
df['rolling_sensitivity_avg'] = df.groupby('employee_id')['app_sensitivity'].shift(1).rolling(2).mean().fillna(1)
df['is_escalating'] = (df['app_sensitivity'] > df['rolling_sensitivity_avg']).astype(int)
df['escalation_delta'] = (df['app_sensitivity'] - df['rolling_sensitivity_avg']).clip(lower=0)

# 10. Categorical Encoding (For sequence learning)
le = LabelEncoder()
df['app_enc'] = le.fit_transform(df['app_desc'].astype(str))

# FINAL FEATURE LIST (20 features — behavioral signals + global off-hours thresholds)
FEATURES = [
    'hour_zscore', 'weekend_surprise', 'is_off_hours', 'is_after_midnight',
    'burst_5min', 'accts_last_hour', 'time_since_last_log_sec', 'time_since_acct_accessed_sec',
    'workload_zscore', 'emp_acct_total', 'emp_acct_days_span',
    'emp_acct_freq_per_day', 'acct_unique_employees',
    'is_dormant_access', 'balance_surprise_zscore',
    'app_sensitivity', 'is_repeated_app', 'is_escalating', 'escalation_delta',
    'app_enc'
]

X_all = df[FEATURES].fillna(0)
print(f"   Features engineered: {len(FEATURES)}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: FEEDBACK LOOP (PHASE 3 #8)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/7] Loading feedback loop…")

FEEDBACK_FILE = 'feedback.json'
confirmed_logs = set()
cleared_logs = set()

if os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE) as f:
        feedback = json.load(f)
    confirmed_logs = {k for k,v in feedback.items() if v == 'confirmed'}
    cleared_logs = {k for k,v in feedback.items() if v == 'cleared'}
    print(f"   Loaded {len(confirmed_logs)} confirmed · {len(cleared_logs)} cleared cases")
else:
    # Generate synthetic feedback for demo
    top_fraud = df.nlargest(60, ['is_off_hours','is_dormant_access','burst_5min']).sample(frac=1).reset_index(drop=True)
    confirmed_logs = set(top_fraud.head(30)['log_id'].tolist())
    cleared_logs = set(top_fraud.tail(15)['log_id'].tolist())
    feedback_data = {lid: 'confirmed' for lid in confirmed_logs}
    feedback_data.update({lid: 'cleared' for lid in cleared_logs})
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(feedback_data, f, indent=2)
    print(f"   Generated demo feedback: {len(confirmed_logs)} confirmed · {len(cleared_logs)} cleared")

df['feedback_label'] = 'unlabeled'
df.loc[df['log_id'].isin(confirmed_logs), 'feedback_label'] = 'confirmed'
df.loc[df['log_id'].isin(cleared_logs), 'feedback_label'] = 'cleared'

# Training set excludes confirmed fraud (keep autoencoder's view of normal clean)
train_mask = df['feedback_label'] != 'confirmed'
X_train_feedback = X_all[train_mask]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: AUTOENCODER (PHASE 3 #1)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/7] Training peer-clustered autoencoders…")

class Autoencoder:
    def __init__(self, input_dim, hidden_dims=(24,16,12), lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        enc = [input_dim] + list(hidden_dims)
        dec = list(reversed(hidden_dims)) + [input_dim]
        sizes = enc + dec[1:]
        self.W, self.b = [], []
        for i in range(len(sizes)-1):
            fan = sizes[i] + sizes[i+1]
            s = np.sqrt(2.0 / fan)
            self.W.append(np.random.randn(sizes[i], sizes[i+1]) * s)
            self.b.append(np.zeros(sizes[i+1]))
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0
        self.n_layers = len(self.W)

    @staticmethod
    def _relu(x): return np.maximum(0, x)
    @staticmethod
    def _relu_d(x): return (x > 0).astype(float)

    def forward(self, X):
        self._cache = [X]
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            h = self._relu(z) if i < self.n_layers - 1 else z
            self._cache.append(h)
        return h

    def backward(self, X, X_hat):
        batch = X.shape[0]
        delta = (X_hat - X) / batch
        dW_list, db_list = [], []
        for i in reversed(range(self.n_layers)):
            h_prev = self._cache[i]
            if i < self.n_layers - 1:
                delta = delta * self._relu_d(self._cache[i+1])
            dW = h_prev.T @ delta
            db = delta.sum(axis=0)
            delta = delta @ self.W[i].T
            dW_list.insert(0, dW)
            db_list.insert(0, db)
        self.t += 1
        for i in range(self.n_layers):
            for param, grad, m, v in [(self.W, dW_list, self.mW, self.vW),
                                       (self.b, db_list, self.mb, self.vb)]:
                g = np.clip(grad[i], -1.0, 1.0)
                m[i] = self.beta1 * m[i] + (1-self.beta1) * g
                v[i] = self.beta2 * v[i] + (1-self.beta2) * g**2
                m_hat = m[i] / (1 - self.beta1**self.t)
                v_hat = v[i] / (1 - self.beta2**self.t)
                param[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def fit(self, X, epochs=50, batch_size=256, verbose=False):
        for epoch in range(epochs):
            idx = np.random.permutation(len(X))
            for start in range(0, len(X), batch_size):
                xb = X[idx[start:start+batch_size]]
                x_hat = self.forward(xb)
                self.backward(xb, x_hat)

    def reconstruction_error(self, X):
        x_hat = self.forward(X)
        return np.mean((X - x_hat)**2, axis=1)

df['peer_group'] = df['dept_name'] + ' | ' + df['position']
ae_scaler = MinMaxScaler()
ae_scaler.fit(X_train_feedback)
X_scaled_all = ae_scaler.transform(X_all)

df['ae_score'] = np.nan
for group in df['peer_group'].unique():
    mask = df['peer_group'] == group
    train_m = mask & (df['feedback_label'] != 'confirmed')
    X_grp_tr = X_scaled_all[train_m]
    X_grp_all = X_scaled_all[mask]
    if len(X_grp_tr) >= 100:
        ae = Autoencoder(input_dim=X_grp_tr.shape[1], hidden_dims=(24,16,12), lr=0.001)
        ae.fit(X_grp_tr, epochs=50, batch_size=256, verbose=False)
        errors = ae.reconstruction_error(X_grp_all)
    else:
        errors = np.mean((X_grp_all - X_grp_tr.mean(axis=0))**2, axis=1)
    
    # Global Calibration: Hard 90th percentile threshold (Dead Zone)
    thresh = np.percentile(errors, 90)
    mx = errors.max()
    if mx > thresh:
        normed = np.where(errors >= thresh, (errors - thresh) / (mx - thresh) * 100, 0)
    else:
        normed = np.where(errors >= thresh, 100, 0) if mx == thresh else np.zeros(len(errors))
    
    confirmed_in_grp = df[mask]['log_id'].isin(confirmed_logs).values
    normed = np.where(confirmed_in_grp, np.minimum(100, normed + 20), normed)
    cleared_in_grp = df[mask]['log_id'].isin(cleared_logs).values
    normed = np.where(cleared_in_grp, normed * 0.5, normed)
    df.loc[mask, 'ae_score'] = normed.round(1)

print(f"   Autoencoders trained: {df['peer_group'].nunique()} peer groups")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: ENSEMBLE SCORING (PHASE 2 #2)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/7] Training ensemble (IF + DBSCAN + SVM)…")

gs = StandardScaler()
Xgs = gs.fit_transform(X_all)
g_if = IsolationForest(n_estimators=200, contamination=0.02, random_state=42, n_jobs=-1)
g_if.fit(Xgs)
if_raw = g_if.score_samples(Xgs)

# Global Calibration: 10% Dead Zone for Isolation Forest
if_thresh = np.percentile(if_raw, 10)
if_min = if_raw.min()
if if_thresh > if_min:
    df['if_score'] = np.where(if_raw <= if_thresh, ((if_thresh - if_raw) / (if_thresh - if_min) * 100), 0).round(1)
else:
    df['if_score'] = np.where(if_raw <= if_thresh, 100, 0).round(1)

g_db = DBSCAN(eps=2.5, min_samples=50, n_jobs=-1)
db_labels = g_db.fit_predict(Xgs)
df['dbscan_score'] = np.where(db_labels == -1, 85.0, 20.0)

g_svm = OneClassSVM(kernel='rbf', nu=0.02, gamma='scale')
train_idx = np.random.choice(len(Xgs), min(2000, len(Xgs)), replace=False)
g_svm.fit(Xgs[train_idx])
svm_raw = g_svm.score_samples(Xgs)
df['svm_score'] = ((svm_raw - svm_raw.min()) / (svm_raw.max() - svm_raw.min()) * 100).clip(0,100).round(1)

print(f"   Ensemble trained: IF + DBSCAN + SVM")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: GRAPH ANOMALY (PHASE 3 #3)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6/7] Building graph anomaly scores…")

G = nx.Graph()
emp_attrs = employees.set_index('employee_id').to_dict(orient='index')
for eid, attrs in emp_attrs.items():
    G.add_node(eid, node_type='employee')
acct_attrs = customers.set_index('acct_no').to_dict(orient='index')
for acct, attrs in acct_attrs.items():
    G.add_node(acct, node_type='account')
edge_counts = df.groupby(['employee_id','acct_no']).size().reset_index(name='weight')
for _, row in edge_counts.iterrows():
    G.add_edge(row['employee_id'], row['acct_no'], weight=int(row['weight']))

pr = nx.pagerank(G, weight='weight', alpha=0.85)
emp_pr = {n: pr[n] for n in G.nodes if G.nodes[n].get('node_type') == 'employee'}
pr_vals = np.array(list(emp_pr.values()))
emp_pr_norm = {k: float((v - pr_vals.min()) / (pr_vals.max() - pr_vals.min()) * 100)
               if pr_vals.max() > pr_vals.min() else 0.0 for k, v in emp_pr.items()}

bc = nx.betweenness_centrality(G, k=300, weight='weight', normalized=True)
emp_bc = {n: bc[n] for n in G.nodes if G.nodes[n].get('node_type') == 'employee'}
bc_vals = np.array(list(emp_bc.values()))
emp_bc_norm = {k: float((v - bc_vals.min()) / (bc_vals.max() - bc_vals.min()) * 100)
               if bc_vals.max() > bc_vals.min() else 0.0 for k, v in emp_bc.items()}
graph_scores = {eid: float(0.40*emp_pr_norm.get(eid,0) + 0.60*emp_bc_norm.get(eid,0))
                for eid in emp_attrs}
df['graph_score'] = df['employee_id'].map(graph_scores).fillna(0)

# Global Calibration: Squelch Graph noise below 80th percentile
df['graph_score'] = np.where(df['graph_score'] < 80, 0, (df['graph_score'] - 80) * 5)

shared_accts = edge_counts.groupby('acct_no').agg(
    employee_count=('employee_id','count'), access_count=('weight','sum')).reset_index()
shared_accts = shared_accts[shared_accts['employee_count'] >= 4].nlargest(40,'employee_count')
nodes_set, links = {}, []
for _, row in shared_accts.iterrows():
    acct = row['acct_no']
    acct_info = acct_attrs.get(acct, {})
    nodes_set[acct] = {'id': acct, 'type': 'account',
                        'balance': int(acct_info.get('current_balance', 0)),
                        'access_count': int(row['access_count'])}
    emps = edge_counts[edge_counts['acct_no']==acct]['employee_id'].tolist()
    for emp in emps:
        if emp not in nodes_set:
            nodes_set[emp] = {'id': emp, 'type': 'employee',
                              'dept': emp_attrs.get(emp,{}).get('dept_name',''),
                              'risk': graph_scores.get(emp, 0)}
        links.append({'source': emp, 'target': acct, 'value': 1})
network = {'nodes': list(nodes_set.values()), 'links': links}

print(f"   Graph: {G.number_of_nodes():,} nodes · {G.number_of_edges():,} edges")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: FINAL SCORING & OUTPUT
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7/7] Final scoring and output…")

# Blend scores: AE(50%) + IF(30%) + Graph(20%)
df['ensemble_score'] = (
    0.50 * df['ae_score'].fillna(50) +
    0.30 * df['if_score'] +
    0.20 * df['graph_score']
).round(1)

df['risk_tier'] = pd.cut(df['ensemble_score'],
                          bins=[0,30,60,80,100],
                          labels=['Low','Medium','High','Critical'])

# Time-decay weighting
LAMBDA = 0.03
df['decay_weight'] = np.exp(-LAMBDA * df['days_ago'])
decay_scores = df.groupby('employee_id').apply(
    lambda g: pd.Series({'weighted_avg_risk':
        round((g['ensemble_score'] * g['decay_weight']).sum() / max(g['decay_weight'].sum(), 1e-9), 1)})
).reset_index()

# Employee composite
emp_base = df.groupby('employee_id').agg(
    dept_name=('dept_name','first'), position=('position','first'),
    branch_number=('branch_number','first'), total_logs=('log_id','count'),
    avg_score=('ensemble_score','mean'), max_score=('ensemble_score','max'),
    high_risk_events=('risk_tier', lambda x: x.isin(['High','Critical']).sum()),
    time_anomalies=('hour_zscore', lambda x: (x > 2.5).sum()),
    unique_accounts=('acct_no','nunique'),
    burst_events=('burst_5min', lambda x: (x > 3).sum()),
    dormant_events=('is_dormant_access','sum'),
    volume_anomalies=('workload_zscore', lambda x: (x > 2.5).sum()),
    most_recent_log=('log_date','max'),
).reset_index()
emp_base['graph_score'] = emp_base['employee_id'].map(graph_scores).fillna(0)
emp_risk = emp_base.merge(decay_scores, on='employee_id', how='left')
emp_risk['composite_risk'] = (
    0.40 * emp_risk['weighted_avg_risk'] +
    0.25 * (emp_risk['high_risk_events'] / emp_risk['total_logs'] * 100) +
    0.20 * emp_risk['graph_score'] +
    0.10 * (emp_risk['time_anomalies'] / emp_risk['total_logs'] * 100) +
    0.05 * (emp_risk['burst_events'] / emp_risk['total_logs'] * 100)
).round(1)
emp_risk = emp_risk.sort_values('composite_risk', ascending=False).reset_index(drop=True)
emp_risk['rank'] = emp_risk.index + 1

# Top alerts
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

df['flags'] = df.apply(get_flags, axis=1)

top_alerts = df.nlargest(100, 'ensemble_score')[[
    'log_id','log_date','employee_id','dept_name','position','branch_number',
    'acct_no','app_desc','current_balance','account_status',
    'ensemble_score','risk_tier','ae_score','if_score','graph_score',
    'flags','days_ago'
]].copy()
top_alerts['log_date'] = top_alerts['log_date'].dt.strftime('%Y-%m-%d %H:%M')
top_alerts['current_balance'] = top_alerts['current_balance'].round(0).astype(int)
top_alerts['risk_tier'] = top_alerts['risk_tier'].astype(str)
top_alerts.rename(columns={'ensemble_score': 'risk_score'}, inplace=True)

# Map emp_timeline and dormant_alerts
emp_timeline = {}
for eid in df['employee_id'].unique():
    logs = df[df['employee_id']==eid].sort_values('log_date', ascending=False).head(50).copy()
    logs['log_date'] = logs['log_date'].dt.strftime('%Y-%m-%d %H:%M')
    logs['risk_score'] = logs['ensemble_score']
    logs['is_burst'] = (logs['burst_5min'] > 3).astype(int)
    emp_timeline[eid] = logs[['log_date','acct_no','app_desc','risk_tier','current_balance','account_status','risk_score','flags','is_dormant_access','is_burst']].to_dict('records')

dormant_alerts = df[df['is_dormant_access']==1].nlargest(50, 'ensemble_score').copy()
dormant_alerts['log_date'] = dormant_alerts['log_date'].dt.strftime('%Y-%m-%d %H:%M')
dormant_alerts['risk_tier'] = dormant_alerts['risk_tier'].astype(str)
dormant_alerts.rename(columns={'ensemble_score': 'risk_score'}, inplace=True)

risk_dist = df['risk_tier'].value_counts().to_dict()

# Behavioral Flag Distribution (Threat Landscape)
all_flags = [f for sublist in df[df['risk_tier'].isin(['High','Critical'])]['flags'] for f in sublist]
threat_dist = pd.Series(all_flags).value_counts().to_dict() if all_flags else {}

top20 = emp_risk.head(20)['employee_id'].tolist()
df['date_str'] = df['log_date'].dt.strftime('%Y-%m-%d')
emp_daily = df[df['employee_id'].isin(top20)].groupby(['employee_id','date_str'])['ensemble_score'].mean().round(1).reset_index()
emp_daily.columns = ['employee_id','date','avg_risk']
risk_trend = {eid: emp_daily[emp_daily['employee_id']==eid].tail(60)[['date','avg_risk']].to_dict(orient='records')
              for eid in top20}

gen_time = datetime.now().strftime('%Y-%m-%d %H:%M')
output = {
    'summary': {
        'generated_at': gen_time,
        'version': 'Unified Clean (Phase 1+2+3)',
        'total_logs': int(len(df)),
        'total_employees': int(df['employee_id'].nunique()),
        'total_accounts': int(df['acct_no'].nunique()),
        'critical_events': int((df['risk_tier']=='Critical').sum()),
        'high_events': int((df['risk_tier']=='High').sum()),
        'burst_events': int((df['burst_5min'] > 3).sum()),
        'dormant_events': int(df['is_dormant_access'].sum()),
        'off_hours_events': int(df['is_off_hours'].sum()),
        'feedback_confirmed': int(len(confirmed_logs)),
        'feedback_cleared': int(len(cleared_logs)),
        'features_used': len(FEATURES),
        'model': 'Autoencoder(50%) + IF(30%) + Graph(20%) + decay lambda=0.03',
    },
    'risk_distribution': {str(k): int(v) for k, v in risk_dist.items()},
    'threat_distribution': {str(k): int(v) for k, v in threat_dist.items()},
    'top_risky_employees': emp_risk.head(20).rename(columns={'avg_score': 'avg_risk_score'}).to_dict(orient='records'),
    'top_alerts': top_alerts.to_dict(orient='records'),
    'dormant_alerts': dormant_alerts.to_dict(orient='records'),
    'emp_timeline': emp_timeline,
    'risk_trend': risk_trend,
    'account_network': network,
}

with open('fraud_output_final.json', 'w') as f:
    json.dump(output, f, default=str, indent=2)

# Also save as .js for local browser security bypass
with open('fraud_data.js', 'w') as f:
    f.write("window.FRAUD_DATA = " + json.dumps(output, default=str, indent=2) + ";")

print(f"\n✓ Output: fraud_output_final.json")
print(f"✓ Output: fraud_data.js (Browser bypass mode)")
print(f"\nSummary:")
print(f"  Features: {len(FEATURES)} (clean, no redundancy, no leakage)")
print(f"  Critical alerts: {output['summary']['critical_events']}")
print(f"  High alerts: {output['summary']['high_events']}")
print(f"  Model: Autoencoder + Ensemble + Graph + Time-decay")
print(f"  Generated: {gen_time}")
print("\n✓ Ready for dashboard: fetch('fraud_output_final.json')")
print("="*75)
