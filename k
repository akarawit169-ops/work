# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: GLOBAL FEATURE IMPORTANCE & MASTER SHAP ENGINE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7/7] Training global surrogate model & initializing SHAP TreeExplainer...")
import shap

# 1. Train the Explainer Model
sample_idx = np.random.choice(len(X_all), min(20000, len(X_all)), replace=False)
X_sample = X_all.iloc[sample_idx]
y_sample = df.loc[df.index[sample_idx], 'ensemble_score']

explainer = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
explainer.fit(X_sample, y_sample)

importances = dict(zip(FEATURES, explainer.feature_importances_))
feature_importance = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

# Initialize the SHAP engine
shap_explainer = shap.TreeExplainer(explainer)

# 2. Compute SHAP globally for the ENTIRE dataset upfront
print("[7/7] Calculating SHAP explanations across all transaction records...")
X_full = df[FEATURES].fillna(0).astype(float)
raw_shap_all = shap_explainer.shap_values(X_full)

# Normalize array dimensions across different versions of the SHAP library
if isinstance(raw_shap_all, list): 
    raw_shap_all = raw_shap_all[0]
elif len(raw_shap_all.shape) == 3: 
    raw_shap_all = raw_shap_all[:, :, 0]

# Initialize columns in the master dataframe to hold the calculated explanations
df['shap_reason'] = "Multiple Behavioral Indicators"
df['shap_explanation_json'] = "{}"

# Loop through every single row to extract its drivers and cache them into the master df
print("[7/7] Stamping AI reasons directly into the master database...")
for idx in range(len(df)):
    row_values = [float(val) for val in raw_shap_all[idx]]
    row_shap = dict(zip(FEATURES, row_values))
    sorted_shap = sorted(row_shap.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Relative Safeguard: Grab the top 2 absolute drivers that contributed something
    top_drivers = [k for k, v in sorted_shap[:2] if abs(v) > 0.0001]
    
    if top_drivers:
        df.loc[df.index[idx], 'shap_reason'] = " + ".join([k.replace('_', ' ').title() for k in top_drivers])
    
    # Store top 5 factors as a clean, standardized dictionary for frontend UI charts
    top_5_dict = {k: float(round(v * 100, 2)) for k, v in sorted_shap[:5]}
    df.loc[df.index[idx], 'shap_explanation_json'] = json.dumps(top_5_dict)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: TIMELINE ISOLATION & BULK JSON TIMELINE GENERATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8/7] Splitting enriched timelines into separate JSON files...")
TL_DIR = 'timelines'
if not os.path.exists(TL_DIR): 
    os.makedirs(TL_DIR)

df_timeline = df.copy()
df_timeline['log_date_str'] = df_timeline['log_date'].dt.strftime('%Y-%m-%d %H:%M')
df['is_burst'] = 0

for eid in df_timeline['employee_id'].unique():
    emp_logs = df_timeline[df_timeline['employee_id'] == eid].sort_values('log_date', ascending=False).copy()
    
    # Re-calculate burst velocity locally inside this employee's timeline context
    emp_logs = emp_logs.set_index('log_date')
    emp_logs['burst_5min_local'] = emp_logs['log_id'].rolling('5min').count().reset_index(level=0, drop=True).values
    emp_logs = emp_logs.reset_index()
    emp_logs['is_burst'] = (emp_logs['burst_5min_local'] > 3).astype(int)
    
    # Sync local burst calculations back to master dataframe to prevent subsequent key mismatches
    df.loc[df['log_id'].isin(emp_logs[emp_logs['is_burst'] == 1]['log_id']), 'is_burst'] = 1
    
    def get_local_flags(row):
        f = []
        if row['is_off_hours']: f.append('Off-hours')
        if row['is_after_midnight']: f.append('After midnight')
        if row['hour_zscore'] > 2.5: f.append('Time Anomaly')
        if row['weekend_surprise'] > 0.7: f.append('Weekend Surprise')
        if row['burst_5min_local'] > 3: f.append(f"Burst ({int(row['burst_5min_local'])} in 5min)")
        if row['is_dormant_access']: f.append('Dormant/Closed')
        if row['is_escalating']: f.append('Escalation')
        if row['balance_surprise_zscore'] > 2.5: f.append('Balance Surprise')
        if row['workload_zscore'] > 2.5: f.append('Volume Spike')
        return f

    emp_logs['flags'] = emp_logs.apply(get_local_flags, axis=1)
    
    # Compile individual timelines (cleanly inheriting pre-calculated SHAP reasons)
    timeline_data = [{
        'log_id': r['log_id'], 
        'log_date': r['log_date_str'], 
        'acct_no': r['acct_no'],
        'risk_score': float(r['ensemble_score']), 
        'flags': r['flags'], 
        'app_desc': r['app_desc'],
        'current_balance': r['current_balance'], 
        'account_status': r['account_status'],
        'risk_tier': r['risk_tier'], 
        'is_dormant_access': int(r['is_dormant_access']), 
        'is_burst': int(r['is_burst']),
        'reason': r['shap_reason'], 
        'shap_explanation': json.loads(r['shap_explanation_json'])
    } for r in emp_logs.to_dict(orient='records')]
    
    with open(f'{TL_DIR}/{eid}.json', 'w') as f_tl:
        json.dump(timeline_data, f_tl, default=str)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9: RISK COUNTS, AGGREGATIONS, TRENDS & GRAPH NETWORKS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[~] Compiling global aggregates, risk trends, and topology networks...")

# 1. Calculate Risk Counts & Triggers
risk_counts = {
    'Off-hours': int(df['is_off_hours'].sum()),
    'After midnight': int(df['is_after_midnight'].sum()),
    'Volume Spike': int((df['workload_zscore'] > 2.5).sum()),
    'Time Anomaly': int((df['hour_zscore'] > 2.5).sum()),
    'Balance Surprise': int((df['balance_surprise_zscore'] > 2.5).sum()),
    'Burst Event': int((df['is_burst'] == 1).sum()),
    'Dormant Access': int((df['is_dormant_access'] == 1).sum()),
    'Privilege Escalation': int((df['is_escalating'] == 1).sum())
}
top_triggers = dict(sorted(risk_counts.items(), key=lambda x: x[1], reverse=True)[:5])

# 2. Build Top Risky Employees High-Level Summary Matrix
print("[~] Aggregating employee composite risk metrics...")
emp_agg = df.groupby('employee_id').agg(
    avg_risk_score=('ensemble_score', 'mean'),
    max_score=('ensemble_score', 'max'),
    total_logs=('log_id', 'count'),
    unique_accounts=('acct_no', 'nunique'),
    burst_events=('is_burst', 'sum'),
    dormant_events=('is_dormant_access', 'sum'),
    dept_name=('dept_name', 'first'),
    position=('position', 'first'),
    branch_number=('branch_number', 'first')
).reset_index()

# Calculate dynamic composite score matching the frontend tracking table formula
emp_agg['composite_risk'] = (
    emp_agg['avg_risk_score'] * 0.4 + 
    emp_agg['max_score'] * 0.3 + 
    (emp_agg['total_logs'] * 0.05).clip(0, 10) + 
    (emp_agg['burst_events'] * 2.0).clip(0, 10) + 
    (emp_agg['dormant_events'] * 3.0).clip(0, 10)
)
top_risky_employees = emp_agg.sort_values('composite_risk', ascending=False).head(20).to_dict(orient='records')

# 3. Calculate 30-Day Rolling Risk Trend Payload
print("[~] Processing rolling historical timelines...")
trend_payload = {}
df_trend = df.copy()
df_trend['date_str'] = df_trend['log_date'].dt.strftime('%Y-%m-%d')

for eid in df['employee_id'].unique():
    emp_trend = df_trend[df_trend['employee_id'] == eid].groupby('date_str')['ensemble_score'].mean().reset_index()
    emp_trend['rolling_risk'] = emp_trend['ensemble_score'].rolling(window=3, min_periods=1).mean().round(1)
    trend_payload[eid] = [
        {'date': r['date_str'], 'avg_risk': float(r['rolling_risk'])} 
        for r in emp_trend.to_dict(orient='records')
    ]

# 4. Generate Account Network Data (D3 cross-access graph nodes)
print("[~] Constructing cross-access network graph nodes...")
acct_cross_access = df.groupby('acct_no')['employee_id'].nunique()
shared_accounts = acct_cross_access[acct_cross_access > 1].index.tolist()

if len(shared_accounts) > 0:
    net_df = df[df['acct_no'].isin(shared_accounts)].copy()
    net_df['unique_emp_count'] = net_df['acct_no'].map(acct_cross_access)
    net_df = net_df.sort_values(by=['unique_emp_count', 'ensemble_score'], ascending=[False, False])
    net_df = net_df.drop_duplicates(subset=['employee_id', 'acct_no'])
    if len(net_df) > 1000:
        net_df = net_df.head(1000)
else:
    net_df = df.nlargest(500, 'ensemble_score').copy()

nodes, links, added_nodes = [], [], set()
for _, row in net_df.iterrows():
    if row['employee_id'] not in added_nodes:
        nodes.append({
            'id': row['employee_id'], 
            'type': 'employee', 
            'dept': row['dept_name'], 
            'risk': float(row['ensemble_score'])
        })
        added_nodes.add(row['employee_id'])
    if row['acct_no'] not in added_nodes:
        nodes.append({
            'id': row['acct_no'], 
            'type': 'account', 
            'status': row['account_status'], 
            'balance': float(row['current_balance']), 
            'access_count': int(df[df['acct_no'] == row['acct_no']].shape[0])
        })
        added_nodes.add(row['acct_no'])
    links.append({'source': row['employee_id'], 'target': row['acct_no'], 'weight': 1})

# 5. Extract Final Clean Alert Slices for Dashboard Payload
top_alerts_df = df.nlargest(100, 'ensemble_score').copy()
top_alerts_df['flags'] = top_alerts_df.apply(get_flags, axis=1)
top_alerts_df['log_date_str'] = top_alerts_df['log_date'].dt.strftime('%Y-%m-%d %H:%M')
top_alerts_df.rename(columns={'ensemble_score': 'risk_score'}, inplace=True)
top_alerts_final = top_alerts_df.to_dict(orient='records')

dormant_alerts_df = df[df['is_dormant_access'] == 1].nlargest(100, 'ensemble_score').copy()
dormant_alerts_df['flags'] = dormant_alerts_df.apply(get_flags, axis=1)
dormant_alerts_df['log_date_str'] = dormant_alerts_df['log_date'].dt.strftime('%Y-%m-%d %H:%M')
dormant_alerts_df.rename(columns={'ensemble_score': 'risk_score'}, inplace=True)
dormant_alerts_final = dormant_alerts_df.to_dict(orient='records')

hourly_avg = df.groupby('hour')['ensemble_score'].mean().round(1).to_dict()
hourly_pattern = {int(k): float(v) for k, v in hourly_avg.items()}


# ══════════════════════════════════════════════════════════════════════════════
# MASTER PAYLOAD EXPORT GENERATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[~] Consolidating metrics into fraud_output_final.json...")

output_payload = {
    'summary': {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'status': 'Ensemble Production Active',
        'total_logs': int(len(df)),
        'total_employees': int(df['employee_id'].nunique()),
        'total_accounts': int(df['acct_no'].nunique()),
        'critical_events': int((df['ensemble_score'] >= 80).sum()),
        'high_events': int(((df['ensemble_score'] >= 60) & (df['ensemble_score'] < 80)).sum()),
        'burst_events': int((df['is_burst'] == 1).sum()),
        'dormant_events': int((df['is_dormant_access'] == 1).sum())
    },
    'risk_distribution': {
        'Low': int((df['ensemble_score'] < 30).sum()),
        'Medium': int(((df['ensemble_score'] >= 30) & (df['ensemble_score'] < 60)).sum()),
        'High': int(((df['ensemble_score'] >= 60) & (df['ensemble_score'] < 80)).sum()),
        'Critical': int((df['ensemble_score'] >= 80).sum())
    },
    'top_triggers': top_triggers,
    'feature_importance': feature_importance,
    'hourly_risk_pattern': hourly_pattern,
    'top_risky_employees': top_risky_employees,
    'risk_trend': trend_payload,
    'account_network': {
        'nodes': nodes,
        'links': links
    },
    'top_alerts': [{
        'log_id': r['log_id'], 'log_date': r['log_date_str'], 'employee_id': r['employee_id'],
        'acct_no': r['acct_no'], 'risk_score': r['risk_score'], 'ensemble_score': r['risk_score'],
        'reason': r['shap_reason'], 'shap_explanation': json.loads(r['shap_explanation_json']),
        'flags': r['flags'], 'risk_tier': r['risk_tier'], 'dept_name': r['dept_name'],
        'position': r['position'], 'branch_number': r['branch_number'], 'app_desc': r['app_desc'],
        'current_balance': r['current_balance'], 'account_status': r['account_status'],
        'is_dormant_access': int(r['is_dormant_access']), 'is_burst': int(r['is_burst'])
    } for r in top_alerts_final],
    'dormant_alerts': [{
        'log_id': r['log_id'], 'log_date': r['log_date_str'], 'employee_id': r['employee_id'],
        'acct_no': r['acct_no'], 'risk_score': r['risk_score'], 'ensemble_score': r['risk_score'],
        'reason': r['shap_reason'], 'shap_explanation': json.loads(r['shap_explanation_json']),
        'flags': r['flags'], 'risk_tier': r['risk_tier'], 'dept_name': r['dept_name'],
        'position': r['position'], 'branch_number': r['branch_number'], 'app_desc': r['app_desc'],
        'current_balance': r['current_balance'], 'account_status': r['account_status'],
        'is_dormant_access': int(r['is_dormant_access']), 'is_burst': int(r['is_burst'])
    } for r in dormant_alerts_final]
}

with open('fraud_output_final.json', 'w') as f_out:
    json.dump(output_payload, f_out, default=str)

print("\n[✓] Pipeline execution finished successfully! Core data structures generated.")
