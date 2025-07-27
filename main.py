import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

htru2 = fetch_ucirepo(id=372)
X = htru2.data.features
y = htru2.data.targets

X.columns = ['Profile_mean', 'Profile_stdev', 'Profile_skewness', 'Profile_kurtosis',
             'DM_mean', 'DM_stdev', 'DM_skewness', 'DM_kurtosis']

df = pd.concat([X, y.iloc[:, 0].rename('class')], axis=1)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.canvas.manager.set_window_title('All Feature Distributions (Pulsars vs Non-Pulsars)')

features = X.columns
for idx, feature in enumerate(features):
    row, col = divmod(idx, 4)
    ax = axes[row, col]
    
    for label in [0, 1]:
        subset = df[df['class'] == label]
        type_label = 'Non-pulsar' if label == 0 else 'Pulsar'
        ax.hist(subset[feature], bins=50, alpha=0.5, label=type_label)
    
    ax.set_title(feature)
    ax.set_xlim(df[feature].min(), df[feature].max())
    ax.grid(True, linestyle='--', alpha=0.6)
    if row == 1:
        ax.set_xlabel('Value')
    if col == 0:
        ax.set_ylabel('Count')

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', fontsize='large')

plt.tight_layout(rect=[0, 0, 0.95, 1])
plt.show()

print("\nAverage feature values by class (0 = non-pulsar, 1 = pulsar):")
print(df.groupby('class').mean())
