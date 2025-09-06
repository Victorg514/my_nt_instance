# Run this to diagnose the data distribution
import pandas as pd

df = pd.read_parquet("data/deviation_advanced.parquet")

# Check class balance
print("Class distribution in splits:")
for split in ['train', 'val', 'test']:
    print(f"\n{split.upper()}:")
    counts = df[df['split']==split]['label_name'].value_counts()
    print(counts)
    print(f"Control ratio: {counts['control'] / counts.sum():.2%}")

# Check feature quality
print("\n\nFeature statistics:")
for col in ['len_delta', 'sent_delta', 'tpd_delta']:
    if col in df.columns:
        print(f"{col}: mean={df[col].mean():.3f}, std={df[col].std():.3f}, nulls={df[col].isna().sum()}")
