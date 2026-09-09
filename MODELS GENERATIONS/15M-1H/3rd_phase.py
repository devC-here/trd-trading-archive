import numpy as np
import pandas as pd

# 1. LOAD THE PRUNED PHASE 2 SELECTED DATASET
phase2_file = "ohlcv_40_m15_phase2_selected.csv"
print(f"Opening pruned matrix file: '{phase2_file}'...")
df = pd.read_csv(phase2_file)

# Extract any column heading that begins with a numeric character digit
phase2_indicators = [col for col in df.columns if col[0].isdigit()]

# 2. ISOLATE TIMELINE ENDPOINTS FOR VALUE GROUPING
df_trigger = df[df['m15_sequence'] == 16]

assessment_report_rows = []

print("Analyzing structural indicators properties...")
for indicator in phase2_indicators:
    # Extract class-conditional parameters
    buy_mean = df_trigger[df_trigger['label'] == 'BUY'][indicator].mean()
    sell_mean = df_trigger[df_trigger['label'] == 'SELL'][indicator].mean()
    hold_mean = df_trigger[df_trigger['label'] == 'HOLD'][indicator].mean()
    
    # Calculate global tracking variance across target states
    class_variance = np.var([buy_mean, sell_mean, hold_mean])
    
    # Measure Sequential Evolution Maturity (Unique states traversed across the 16-period matrix)
    unique_trajectories = df.groupby(['condition_id', 'sample'])[indicator].nunique().mean()
    
    # Evaluate Symmetrical Integrity (Compares original configurations to augmented inverse loops)
    orig_buy = df_trigger[(df_trigger['condition_type'] == 'ORIGINAL') & (df_trigger['label'] == 'BUY')][indicator].mean()
    inv_sell = df_trigger[(df_trigger['condition_type'] == 'INVERSE') & (df_trigger['label'] == 'SELL')][indicator].mean()
    
    # An ideal symmetrical oscillator should cross cleanly (Orig BUY == -Inv SELL)
    symmetry_coefficient = 1.0 - abs(orig_buy + inv_sell) if (abs(orig_buy) > 0 and abs(inv_sell) > 0) else 1.0
    if np.isnan(symmetry_coefficient):
        symmetry_coefficient = 1.0
        
    assessment_report_rows.append({
        'Indicator_Code': indicator,
        'BUY_Mean_Value': buy_mean,
        'SELL_Mean_Value': sell_mean,
        'HOLD_Mean_Value': hold_mean,
        'Separation_Variance': class_variance,
        'Avg_Evolution_Steps': unique_trajectories,
        'Symmetry_Score': symmetry_coefficient
    })

# Compile tabular summary matrix
df_report = pd.DataFrame(assessment_report_rows)
df_report = df_report.sort_values(by='Separation_Variance', ascending=False)

print("\n=========================================================================")
print("📊 INDICATORS STRUCTURAL ASSESSMENT REPORT")
print("=========================================================================")
print(df_report.to_string(index=False, formatters={
    'BUY_Mean_Value': '{:,.5f}'.format,
    'SELL_Mean_Value': '{:,.5f}'.format,
    'HOLD_Mean_Value': '{:,.5f}'.format,
    'Separation_Variance': '{:,.6f}'.format,
    'Avg_Evolution_Steps': '{:,.2f}'.format,
    'Symmetry_Score': '{:,.4f}'.format
}))
print("=========================================================================\n")

# Save detailed profiling specs out to a reference file
df_report.to_csv("phase2_indicators_assessment_metrics.csv", index=False)
print("✔️ Assessment metrics safely saved to: 'phase2_indicators_assessment_metrics.csv'")