import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

# 1. LOAD THE EXTENSIVE EXPANDED 105-INDICATOR GRID
input_file = "ohlcv_40_m15_expanded_100.csv"
print(f"Loading feature space matrix from: '{input_file}'...")
df = pd.read_csv(input_file)

# Identify indicator column arrays (any column heading starting with a number digit)
indicator_cols = [col for col in df.columns if col[0].isdigit()]

# 2. RUN ASSESSMENT TARGETS AT THE LOOKBACK WINDOW TIMEOUT TRIGGER STEP (16)
df_trigger_step = df[df['m15_sequence'] == 16]
X_trigger = df_trigger_step[indicator_cols]
y_trigger = df_trigger_step['label']

print("Computing non-linear Information Gain via Mutual Information Classifiers...")
mi_scores = mutual_info_classif(X_trigger, y_trigger, random_state=42)

# Structure information gain outputs into an aligned DataFrame
df_mi = pd.DataFrame({
    'Indicator': indicator_cols,
    'MI_Score': mi_scores
}).sort_values(by='MI_Score', ascending=False)

# Extract a sorted list based on raw informational credibility score
ranked_indicators = df_mi['Indicator'].tolist()

# 3. INITIALIZE THE GREEDY COLLINEARITY DE-DUPLICATION LOOP
print("Computing absolute cross-correlation matrix coefficients...")
corr_matrix = X_trigger[ranked_indicators].corr().abs()

selected_alpha_pool = []
collinearity_threshold = 0.85  # Strict limit: prune any feature sharing >=85% correlation

for feature in ranked_indicators:
    is_redundant = False
    for selected_feature in selected_alpha_pool:
        # Check cross-correlation against currently approved independent features
        if corr_matrix.loc[feature, selected_feature] >= collinearity_threshold:
            is_redundant = True
            break
            
    if not is_redundant:
        selected_alpha_pool.append(feature)

# 4. ENFORCE STRUCTURAL BOUNDARY TO TOP 35 ALPHAS FOR PHASE 2
phase2_features = selected_alpha_pool[:35]

print("\n=========================================================================")
print(f"📊 PHASE 2 ASSESSMENT REPORT")
print("=========================================================================")
print(f"Total Raw Inputs Tested            : {len(indicator_cols)}")
print(f"Independent Structural Alphas Found: {len(selected_alpha_pool)}")
print(f"Hard-Selected Features for Phase 2 : {len(phase2_features)}")
print("=========================================================================\n")

# Compile summary breakdown showing information gain values
df_summary = df_mi[df_mi['Indicator'].isin(phase2_features)].sort_values(by='Information_Gain' if 'Information_Gain' in df_mi.columns else 'MI_Score', ascending=False)
print(df_summary.to_string(index=False))

# 5. GENERATE CLEAN PHASE 2 SUB-DATASET FOR ALGORITHMIC MODELLING STACKS
meta_columns = [c for c in df.columns if not c[0].isdigit()]
phase2_columns_layout = meta_columns + phase2_features

df_phase2_output = df[phase2_columns_layout]
output_filename = "ohlcv_40_m15_phase2_selected.csv"
df_phase2_output.to_csv(output_filename, index=False)

print(f"\n✔️ Slicing Complete. Clean Phase 2 sub-matrix written to: '{output_filename}'")
print(f"✔️ Sub-matrix Final Geometry Dimensions: {df_phase2_output.shape}")