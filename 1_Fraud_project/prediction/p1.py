# 模型加载、数据预处理、预测和应用阈值整合成一个完整的流程，用于对新的交易数据进行批量预测。它可以视为一个可重复使用的处理流程

import pandas as pd
import h2o

# 初始化 H2O
h2o.init(max_mem_size="2G")

# 加载已保存的模型
model_path = "D:/demo/models/DeepLearning_1_AutoML_1_20260225_213704"
model = h2o.load_model(model_path)

def preprocess_new_data(df):
    """
    对新数据应用与训练时完全相同的预处理
    df: 原始数据的 pandas DataFrame（应包含 'trans_date_trans_time' 等原始列）
    返回预处理后的 DataFrame（不含目标列，但包含所有特征）
    """
    # 复制以免修改原始数据
    data = df.copy()
    
    # 转换时间列
    data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
    
    # 提取时间特征
    data['hour'] = data['trans_date_trans_time'].dt.hour
    data['day_of_week'] = data['trans_date_trans_time'].dt.dayofweek
    data['day_of_month'] = data['trans_date_trans_time'].dt.day
    data['month'] = data['trans_date_trans_time'].dt.month
    data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
    
    # 删除训练时删除的列
    drop_cols = ['Unnamed: 0', 'trans_num', 'cc_num', 'first', 'last', 'street', 'zip', 'dob', 'trans_date_trans_time']
    data = data.drop(columns=[c for c in drop_cols if c in data.columns], errors='ignore')
    
    # 确保所有特征列都存在
    # 检查缺失值并填充
    for col in model._model_json['output']['names']:
        if col not in data.columns:
            data[col] = 0  # 假设缺失特征用 0 填充
    return data

# 读取新交易数据
new_df = pd.read_csv("D:\\demo\\new_transactions.csv")
processed_df = preprocess_new_data(new_df)

# 转换为 H2OFrame
new_h2o = h2o.H2OFrame(processed_df)

# 预测
pred = model.predict(new_h2o)
pred_df = pred.as_data_frame()  # 包含 'predict' 和 'p1'（欺诈概率）

# 应用阈值
threshold = 0.11
pred_df['fraud_flag'] = (pred_df['p1'] >= threshold).astype(int)

# 将结果与原始数据合并
result = pd.concat([new_df, pred_df[['p1', 'fraud_flag']]], axis=1)

# 输出结果
result.to_csv("predictions_with_threshold.csv", index=False)
print("预测完成，结果已保存。")

# 关闭 H2O
h2o.cluster().shutdown()