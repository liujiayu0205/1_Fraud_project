import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# 业务成本参数
C_fraud = 120    # 平均每笔欺诈损失（$）
C_false = 8      # 平均每笔误报处理成本（$）

thresholds = np.arange(0, 1.01, 0.01)
results = []

for t in thresholds:
    y_pred = (y_pred_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    cost = fn * C_fraud + fp * C_false
    results.append({
        'threshold': t,
        'fn': fn,
        'fp': fp,
        'cost': cost,
        'recall': tp / (tp + fn) if (tp+fn) > 0 else 0,
        'precision': tp / (tp + fp) if (tp+fp) > 0 else 0
    })

df_cost = pd.DataFrame(results)
best_row = df_cost.loc[df_cost['cost'].idxmin()]

print("=" * 60)
print("基于业务成本的最优阈值选择")
print("=" * 60)
print(f"业务成本参数：欺诈损失 = ${C_fraud}/笔，误报成本 = ${C_false}/笔")
print(f"\n最优阈值: {best_row['threshold']:.3f}")
print(f"最小总成本: ${best_row['cost']:.2f}")
print(f"对应召回率: {best_row['recall']:.3f} (漏报 {best_row['fn']})")
print(f"对应精确率: {best_row['precision']:.3f} (误报 {best_row['fp']})")

# 绘制成本曲线
plt.figure(figsize=(10, 6))
plt.plot(df_cost['threshold'], df_cost['cost'], 'b-')
plt.axvline(best_row['threshold'], color='red', linestyle='--', label=f"最优阈值 = {best_row['threshold']:.3f}")
plt.xlabel('Threshold')
plt.ylabel('Total Cost ($)')
plt.title('Cost vs. Threshold')
plt.grid(True)
plt.legend()
plt.show()