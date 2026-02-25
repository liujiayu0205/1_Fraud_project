# -*- coding: utf-8 -*-
"""
信用卡欺诈检测 - H2O AutoML 建模（使用原始特征）
数据集：fraudTrain.csv（训练集）、fraudTest.csv（测试集）
要求：Python 3.6+，安装 h2o, pandas
"""

import pandas as pd
import h2o
from h2o.automl import H2OAutoML

# -------------------- 1. 文件路径设置 --------------------
TRAIN_PATH = "./dataset/fraudTrain.csv"
TEST_PATH = "./dataset/fraudTest.csv"

# -------------------- 2. 读取数据 --------------------
print("读取训练集...")
train_df = pd.read_csv(TRAIN_PATH)
print(f"训练集形状：{train_df.shape}")
print("读取测试集...")
test_df = pd.read_csv(TEST_PATH)
print(f"测试集形状：{test_df.shape}")

# -------------------- 3. 数据概览与预处理 --------------------
# 目标列名
target = 'is_fraud'

# 检查目标列是否存在
if target not in train_df.columns or target not in test_df.columns:
    raise ValueError(f"数据集中缺少目标列 '{target}'，请检查列名。")

# 特征列：排除目标列，其余全部保留（原始特征）
features = [col for col in train_df.columns if col != target]
print(f"原始特征数量：{len(features)}")
print(f"特征列表：{features}")

# 检查缺失值
print("\n训练集缺失值统计：")
print(train_df.isnull().sum()[train_df.isnull().sum() > 0])
print("\n测试集缺失值统计：")
print(test_df.isnull().sum()[test_df.isnull().sum() > 0])

# 处理缺失值（H2O 不允许缺失值，此处用简单填充）
# 数值列用中位数填充，分类列用众数填充
for df in [train_df, test_df]:
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

# 查看类别分布（训练集）
print("\n训练集目标变量分布：")
print(train_df[target].value_counts())

# -------------------- 4. 初始化 H2O 并加载数据 --------------------
h2o.init(max_mem_size="8G")  # 根据机器内存调整，建议至少 4G

train = h2o.H2OFrame(train_df)
test = h2o.H2OFrame(test_df)

# 定义特征与目标
x = features
y = target

# 将目标列转换为因子（分类）类型
train[y] = train[y].asfactor()
test[y] = test[y].asfactor()

# -------------------- 5. 运行 H2O AutoML --------------------
print("\n开始 H2O AutoML 训练...")
aml = H2OAutoML(
    max_runtime_secs=600,           # 最大运行时间（秒），可按需调整
    max_models=None,                 # 不限制模型数量
    seed=42,
    exclude_algos=None,              # 可排除某些算法，如 ["DeepLearning"]
    nfolds=5,                        # 交叉验证折数
    balance_classes=True,            # 处理类别不平衡（欺诈样本通常较少）
    class_sampling_factors=None,
    max_after_balance_size=5.0,
    keep_cross_validation_predictions=True,
    sort_metric="AUC",               # 不平衡分类常用 AUC
)

aml.train(x=x, y=y, training_frame=train)

# -------------------- 6. 查看结果 --------------------
# 获取排行榜
lb = aml.leaderboard
print("\nAutoML 排行榜（按 AUC 排序）：")
print(lb.head(rows=lb.nrows))

# 获取最佳模型
best_model = aml.leader
print("\n最佳模型：")
print(best_model)

# -------------------- 7. 在测试集上评估最佳模型 --------------------
perf = best_model.model_performance(test)
print("\n测试集评估结果：")
print(perf)

# 可选：保存最佳模型
model_path = h2o.save_model(model=best_model, path="./h2o_models", force=True)
print(f"\n最佳模型已保存至：{model_path}")

# -------------------- 8. 关闭 H2O --------------------
h2o.cluster().shutdown(prompt=False)

print("\n运行完成！")