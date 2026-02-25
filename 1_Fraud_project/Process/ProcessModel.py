# 验证集是从 fraudTrain.csv 中按时间排序后取最后20%得到的。因此，

# 重新加载 fraudTrain.csv

# 按时间排序并划分出验证集（与训练时完全相同的划分逻辑）

# 用已保存的模型对验证集进行预测，得到概率

import pandas as pd
import h2o
from h2o.automl import H2OAutoML

# ==================== 配置路径 ====================
TRAIN_PATH = "./dataset/fraudTrain.csv"          # 原始训练数据路径
MODEL_PATH = "D:/demo/models/DeepLearning_1_AutoML_1_20260225_213704"  # 你保存的模型路径

# ==================== 重新加载数据并划分验证集 ====================
# 读取训练数据
train_df = pd.read_csv(TRAIN_PATH)

# 转换时间列并排序（与训练时一致）
train_df['trans_date_trans_time'] = pd.to_datetime(train_df['trans_date_trans_time'])
train_df = train_df.sort_values('unix_time').reset_index(drop=True)

# 划分：前80%训练，后20%验证
split_point = int(len(train_df) * 0.8)
valid_time_df = train_df.iloc[split_point:].copy()   # 这就是验证集

print(f"验证集大小: {valid_time_df.shape}")

# 注意：验证集还需要包含原始特征（包括之前删除的列？）不需要，因为模型训练时使用的特征列表是固定的。
# 但在预测时，需要提供与训练时相同的特征列（包括生成的时间特征等）。
# 所以需要对验证集应用相同的特征工程。

# 对验证集应用特征工程（与训练时完全一致）
# 从 'trans_date_trans_time' 提取时间特征
valid_time_df['hour'] = valid_time_df['trans_date_trans_time'].dt.hour
valid_time_df['day_of_week'] = valid_time_df['trans_date_trans_time'].dt.dayofweek
valid_time_df['day_of_month'] = valid_time_df['trans_date_trans_time'].dt.day
valid_time_df['month'] = valid_time_df['trans_date_trans_time'].dt.month
valid_time_df['is_weekend'] = (valid_time_df['day_of_week'] >= 5).astype(int)

# 删除训练时删除的列（与训练代码保持一致）
drop_cols = ['Unnamed: 0', 'trans_num', 'cc_num', 'first', 'last', 'street', 'zip', 'dob', 'trans_date_trans_time']
valid_time_df = valid_time_df.drop(columns=[c for c in drop_cols if c in valid_time_df.columns])

# 确保验证集中没有缺失值
# 若有缺失，填充方式需与训练时一致（中位数/众数）

# ==================== 加载模型并预测 ====================
h2o.init(max_mem_size="4G")  # 启动 H2O

# 加载已保存的模型
model = h2o.load_model(MODEL_PATH)

# 将验证集转换为 H2OFrame
valid_h2o = h2o.H2OFrame(valid_time_df)

# 进行预测
pred_valid = model.predict(valid_h2o)

# 提取预测概率（欺诈类别的概率，列名为 'p1'）
pred_df = pred_valid.as_data_frame()
y_pred_prob = pred_df['p1'].values

# 提取真实标签
y_val = valid_h2o['is_fraud'].as_data_frame().values.ravel()

print(f"预测概率范围: {y_pred_prob.min():.4f} - {y_pred_prob.max():.4f}")
print(f"真实欺诈比例: {y_val.mean():.4f}")


import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# ==================== 业务成本参数（请根据实际情况调整）====================
C_fraud = 120    # 平均每笔欺诈损失（美元）
C_false = 8      # 平均每笔误报处理成本（美元）
# ====================================================================

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
