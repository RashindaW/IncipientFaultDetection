"""Create a comprehensive summary visualization of missing value patterns."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read all summary stats
files = ['BaselineTestA', 'BaselineTestB', 'BaselineTestC', 'BaselineTestD', 'BaselineTestE']
all_stats = []

for f in files:
    df = pd.read_csv(f'missing_pattern_analysis/{f}/missing_summary_stats.csv')
    df['File'] = f
    all_stats.append(df)

combined = pd.concat(all_stats, ignore_index=True)

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Missing percentage by file and column
ax1 = fig.add_subplot(gs[0, :2])
pivot = combined.pivot(index='File', columns='Column', values='Missing %')
pivot.plot(kind='bar', ax=ax1, color=['#e74c3c', '#3498db'], width=0.7, edgecolor='black', linewidth=1.5)
ax1.set_title('Missing Value Percentage by File', fontsize=16, fontweight='bold', pad=15)
ax1.set_ylabel('Missing %', fontsize=13)
ax1.set_xlabel('File', fontsize=13)
ax1.legend(title='Column', fontsize=11, title_fontsize=12)
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, max(40, pivot.max().max() + 5))

# Add value labels on bars
for container in ax1.containers:
    ax1.bar_label(container, fmt='%.1f%%', fontsize=9)

# 2. Number of gaps
ax2 = fig.add_subplot(gs[0, 2])
gap_data = combined[combined['Num Gaps'] > 0]
if not gap_data.empty:
    gap_summary = gap_data.groupby('Column')['Num Gaps'].sum()
    colors = ['#e74c3c' if 'EEVin' in col else '#3498db' for col in gap_summary.index]
    bars = ax2.bar(range(len(gap_summary)), gap_summary.values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(gap_summary)))
    ax2.set_xticklabels([col.replace('T-', '').replace('-In', '') for col in gap_summary.index], rotation=45, ha='right')
    ax2.set_title('Total Number of Gaps', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Gaps', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 3. Average gap length
ax3 = fig.add_subplot(gs[1, :2])
for col in combined['Column'].unique():
    col_data = combined[combined['Column'] == col]
    color = '#e74c3c' if 'EEVin' in col else '#3498db'
    marker = 'o' if 'EEVin' in col else 's'
    ax3.plot(col_data['File'], col_data['Avg Gap Length'], 
            marker=marker, markersize=12, linewidth=3, label=col, color=color, alpha=0.7)

ax3.set_title('Average Gap Length by File', fontsize=16, fontweight='bold', pad=15)
ax3.set_ylabel('Average Gap Length (samples)', fontsize=13)
ax3.set_xlabel('File', fontsize=13)
ax3.legend(fontsize=11)
ax3.grid(alpha=0.3)
ax3.set_xticks(range(len(files)))
ax3.set_xticklabels(files, rotation=45, ha='right')

# 4. Longest gap (log scale)
ax4 = fig.add_subplot(gs[1, 2])
longest_data = combined[combined['Longest Gap'] > 0]
if not longest_data.empty:
    pivot_longest = longest_data.pivot(index='File', columns='Column', values='Longest Gap')
    pivot_longest.plot(kind='bar', ax=ax4, color=['#e74c3c', '#3498db'], width=0.7, 
                      edgecolor='black', linewidth=1.5, logy=True)
    ax4.set_title('Longest Gap (Log Scale)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Samples (log scale)', fontsize=12)
    ax4.set_xlabel('File', fontsize=12)
    ax4.legend(title='Column', fontsize=10, title_fontsize=11)
    ax4.grid(axis='y', alpha=0.3, which='both')
    ax4.axhline(y=1000, color='red', linestyle='--', linewidth=2, alpha=0.5, label='1000 samples')

# 5. Value statistics - Mean
ax5 = fig.add_subplot(gs[2, 0])
for col in combined['Column'].unique():
    col_data = combined[combined['Column'] == col]
    color = '#e74c3c' if 'EEVin' in col else '#3498db'
    ax5.bar(col_data['File'], col_data['Value Mean'], alpha=0.7, label=col, color=color)

ax5.set_title('Mean Value by File\n(Valid Data Only)', fontsize=13, fontweight='bold')
ax5.set_ylabel('Mean Value', fontsize=11)
ax5.set_xlabel('File', fontsize=11)
ax5.legend(fontsize=9)
ax5.grid(axis='y', alpha=0.3)
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)

# 6. Value statistics - Std
ax6 = fig.add_subplot(gs[2, 1])
for col in combined['Column'].unique():
    col_data = combined[combined['Column'] == col]
    color = '#e74c3c' if 'EEVin' in col else '#3498db'
    ax6.bar(col_data['File'], col_data['Value Std'], alpha=0.7, label=col, color=color)

ax6.set_title('Std Deviation by File\n(Valid Data Only)', fontsize=13, fontweight='bold')
ax6.set_ylabel('Std Deviation', fontsize=11)
ax6.set_xlabel('File', fontsize=11)
ax6.legend(fontsize=9)
ax6.grid(axis='y', alpha=0.3)
plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)

# 7. Summary table
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('off')

summary_text = """
SUMMARY STATISTICS

T-BP-EEVin:
  • Avg Missing: 12.72%
  • Files with 0%: 2 out of 5
  • Max Missing: 34.90%
  • Total Gaps: 1,905
  
T-GC-Fan1-In:
  • Avg Missing: 7.73%
  • Files with 0%: 3 out of 5
  • Max Missing: 20.16%
  • Total Gaps: 588

RECOMMENDATIONS:
✓ Train on clean files first
✓ Use multi-stage imputation
✓ Consider missing_mask input
"""

ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Overall title
fig.suptitle('Missing Value Pattern Analysis: T-BP-EEVin & T-GC-Fan1-In', 
            fontsize=20, fontweight='bold', y=0.98)

# Save
plt.savefig('missing_pattern_analysis/SUMMARY_VISUALIZATION.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved comprehensive summary: missing_pattern_analysis/SUMMARY_VISUALIZATION.png")

plt.close()

print("\n📊 Summary statistics:")
print("\nT-BP-EEVin:")
eev_data = combined[combined['Column'] == 'T-BP-EEVin']
print(f"  Average missing: {eev_data['Missing %'].mean():.2f}%")
print(f"  Files with 0% missing: {(eev_data['Missing %'] == 0).sum()} out of {len(eev_data)}")
print(f"  Total gaps across all files: {eev_data['Num Gaps'].sum()}")

print("\nT-GC-Fan1-In:")
fan_data = combined[combined['Column'] == 'T-GC-Fan1-In']
print(f"  Average missing: {fan_data['Missing %'].mean():.2f}%")
print(f"  Files with 0% missing: {(fan_data['Missing %'] == 0).sum()} out of {len(fan_data)}")
print(f"  Total gaps across all files: {fan_data['Num Gaps'].sum()}")

