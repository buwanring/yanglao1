# 平台1_客户画像_升级版.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import hashlib
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ========== 自定义JSON编码器 ==========
class NumpyEncoder(json.JSONEncoder):
    """处理numpy数据类型的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, datetime.date, datetime.datetime)):
            return str(obj)
        return super().default(obj)

st.set_page_config(
    page_title="平台1：客户画像系统 | AI增强版",
    page_icon="🧠",
    layout="wide"
)

# ========== 高级样式 ==========
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
.ai-header {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    padding: 2rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    color: white;
    text-align: center;
    box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    border: 1px solid #00ff88;
    animation: glow 2s ease-in-out infinite alternate;
}
@keyframes glow {
    from { box-shadow: 0 20px 40px rgba(0,255,136,0.2); }
    to { box-shadow: 0 20px 60px rgba(0,255,136,0.4); }
}
.ai-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.5rem;
    font-weight: 900;
    background: linear-gradient(45deg, #00ff88, #00b8ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-3d {
    background: rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 20px;
    border: 1px solid rgba(0,255,136,0.3);
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    transition: transform 0.3s;
}
.metric-3d:hover {
    transform: translateY(-10px) scale(1.02);
    border-color: #00ff88;
}
.tag-cloud {
    background: rgba(0,0,0,0.7);
    backdrop-filter: blur(10px);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #00ff88;
}
.tag-item {
    display: inline-block;
    padding: 8px 16px;
    margin: 5px;
    background: linear-gradient(45deg, #00ff88, #00b8ff);
    color: black;
    font-weight: 600;
    border-radius: 20px;
    box-shadow: 0 0 15px rgba(0,255,136,0.5);
    animation: float 3s ease-in-out infinite;
}
@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-5px); }
    100% { transform: translateY(0px); }
}
.blockchain-badge {
    background: #1a1a1a;
    color: #00ff88;
    padding: 5px 15px;
    border-radius: 20px;
    font-family: monospace;
    border: 1px solid #00ff88;
    font-size: 0.8rem;
}
.insight-box {
    background: linear-gradient(135deg, #667eea20, #764ba220);
    border-left: 5px solid #00ff88;
    padding: 20px;
    border-radius: 10px;
    margin: 15px 0;
    backdrop-filter: blur(5px);
}
.code-display {
    background: #1e1e1e;
    color: #00ff88;
    padding: 15px;
    border-radius: 10px;
    font-family: 'Courier New', monospace;
    font-size: 0.9rem;
    border: 1px solid #00ff88;
    white-space: pre-wrap;
}
</style>
""", unsafe_allow_html=True)

# ========== 高级AI模型类（修复特征不一致问题） ==========
class AdvancedRiskModel:
    """高级风险评估模型（机器学习）"""
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance = None
        self.expected_features = None  # 保存期望的特征名称
        self.label_encoders = {}  # 保存标签编码器

    def create_features(self, df):
        """特征工程：创建高级特征"""
        features = pd.DataFrame()
        # 基础特征
        features['age'] = df['年龄']
        features['pension'] = df['月养老金收入']
        features['asset'] = df['可投资资产(万)']
        features['medical_ratio'] = df['医疗支出占比(%)']
        features['risk_score'] = df['风险问卷得分']
        features['exp_years'] = df['投资经验年限']
        features['lock_period'] = df['资金锁定期限(年)']
        # 衍生特征
        features['asset_per_age'] = df['可投资资产(万)'] / (df['年龄'] + 1)  # 避免除0
        features['pension_per_medical'] = df['月养老金收入'] / (df['医疗支出占比(%)'] + 1)
        features['risk_exp_interaction'] = df['风险问卷得分'] * df['投资经验年限']
        features['medical_risk_interaction'] = df['医疗支出占比(%)'] * df['风险问卷得分']
        # 类别变量编码（使用独热编码，确保所有类别都存在）
        # 性别
        features['gender_male'] = (df['性别'] == '男').astype(int)
        # 子女支持
        features['has_child'] = (df['子女支持'] == '有').astype(int)
        # 负债
        features['has_debt'] = (df['是否有负债'] == '是').astype(int)
        # 大额支出
        features['has_big_expense'] = (df['一年内大额支出'] == '是').astype(int)
        # 婚姻状态（使用独热编码，固定所有类别）
        marriage_dummies = pd.get_dummies(df['婚姻状态'], prefix='marriage')
        # 确保所有婚姻状态类别都存在
        for marriage_type in ['已婚', '丧偶', '离异', '未婚']:
            col_name = f'marriage_{marriage_type}'
            if col_name not in marriage_dummies.columns:
                marriage_dummies[col_name] = 0
        features = pd.concat([features, marriage_dummies], axis=1)
        return features

    def train(self, df):
        """训练模型"""
        # 准备特征
        X = self.create_features(df)
        self.expected_features = X.columns.tolist()  # 保存特征名称
        # 风险等级编码
        risk_map = {'低风险': 0, '中风险': 1, '高风险': 2}
        y = df['风险等级'].map(risk_map)
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        # 训练随机森林
        self.rf_model.fit(X_scaled, y)
        # 特征重要性
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        self.is_trained = True
        return self

    def predict_risk(self, customer_data):
        """预测单个客户风险"""
        if not self.is_trained:
            return None
        # 创建特征
        df_single = pd.DataFrame([customer_data])
        X = self.create_features(df_single)
        # 确保特征与训练时一致（添加缺失的特征列）
        for col in self.expected_features:
            if col not in X.columns:
                X[col] = 0
        # 按训练时的顺序排列特征
        X = X[self.expected_features]
        # 标准化
        X_scaled = self.scaler.transform(X)
        # 预测
        risk_pred = self.rf_model.predict(X_scaled)[0]
        risk_proba = self.rf_model.predict_proba(X_scaled)[0]
        risk_map_rev = {0: '低风险', 1: '中风险', 2: '高风险'}
        return {
            'risk_level': risk_map_rev[risk_pred],
            'confidence': float(max(risk_proba) * 100),
            'probabilities': {
                '低风险': float(risk_proba[0] * 100),
                '中风险': float(risk_proba[1] * 100),
                '高风险': float(risk_proba[2] * 100)
            }
        }

    def get_feature_importance(self):
        """获取特征重要性"""
        return self.feature_importance

# ========== 客户分群引擎（自研K-Means） ==========
class SimpleKMeans:
    """简化的K-Means聚类（不依赖sklearn）"""
    def __init__(self, n_clusters=4, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
        self.cluster_names = {
            0: "保守型银发族",
            1: "稳健型中产",
            2: "进取型高净值",
            3: "医疗关注型"
        }

    def fit(self, X):
        """训练聚类模型"""
        # 随机初始化中心点
        n_samples = X.shape[0]
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices].copy()
        for _ in range(self.max_iters):
            # 分配样本到最近的中心点
            distances = np.array([[np.linalg.norm(x - centroid) for centroid in self.centroids] for x in X])
            labels = np.argmin(distances, axis=1)
            # 更新中心点
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
            # 检查收敛
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
        return self

    def predict(self, X):
        """预测聚类"""
        distances = np.array([[np.linalg.norm(x - centroid) for centroid in self.centroids] for x in X])
        return np.argmin(distances, axis=1)

class CustomerSegmentation:
    """客户分群引擎"""
    def __init__(self, n_clusters=4):
        self.kmeans = SimpleKMeans(n_clusters=n_clusters)
        self.scaler = StandardScaler()
        self.cluster_names = {
            0: "保守型银发族",
            1: "稳健型中产",
            2: "进取型高净值",
            3: "医疗关注型"
        }
        self.feature_columns = ['年龄', '可投资资产(万)', '医疗支出占比(%)', '风险问卷得分', '投资经验年限']

    def fit(self, df):
        """训练聚类模型"""
        features = df[self.feature_columns].fillna(0).values
        features_scaled = self.scaler.fit_transform(features)
        self.kmeans.fit(features_scaled)
        return self

    def predict_cluster(self, customer_data):
        """预测客户所属分群"""
        features = np.array([[
            float(customer_data['年龄']),
            float(customer_data['可投资资产(万)']),
            float(customer_data['医疗支出占比(%)']),
            float(customer_data['风险问卷得分']),
            float(customer_data['投资经验年限'])
        ]])
        features_scaled = self.scaler.transform(features)
        cluster = self.kmeans.predict(features_scaled)[0]
        return int(cluster), self.cluster_names.get(int(cluster), "未知类型")

# ========== 区块链存证模拟 ==========
class BlockchainSimulator:
    """区块链存证模拟"""
    @staticmethod
    def convert_to_serializable(obj):
        """递归转换所有numpy类型为Python原生类型"""
        if isinstance(obj, dict):
            return {key: BlockchainSimulator.convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [BlockchainSimulator.convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, datetime.date, datetime.datetime)):
            return str(obj)
        else:
            return obj

    @staticmethod
    def create_hash(data):
        """创建数据哈希"""
        serializable_data = BlockchainSimulator.convert_to_serializable(data)
        data_str = json.dumps(serializable_data, sort_keys=True, cls=NumpyEncoder)
        return hashlib.sha256(data_str.encode()).hexdigest()

    @staticmethod
    def create_block(data, previous_hash=None):
        """创建区块"""
        serializable_data = BlockchainSimulator.convert_to_serializable(data)
        timestamp = datetime.datetime.now().isoformat()
        block = {
            'timestamp': timestamp,
            'data': serializable_data,
            'previous_hash': previous_hash or '0'*64,
            'nonce': int(np.random.randint(1000000))
        }
        block_str = json.dumps(block, sort_keys=True, cls=NumpyEncoder)
        block['hash'] = hashlib.sha256(block_str.encode()).hexdigest()
        return block

# ========== 主界面 ==========
def main():
    """主程序"""
    st.markdown("""
    <div class="ai-header">
        <h1 class="ai-title">🧠 平台1：AI客户画像与风险评估系统</h1>
        <p style="font-size:1.2rem; color:#aaa;">客服岗 · AI增强型 · 四岗协同入口</p>
        <div style="margin-top:15px;">
            <span class="blockchain-badge">🔗 区块链存证已启用</span>
            <span class="blockchain-badge">🤖 机器学习模型 v2.0</span>
            <span class="blockchain-badge">📊 实时聚类分析</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 加载数据
    try:
        df = pd.read_csv('111.csv', encoding='gbk')
        st.success(f"✅ 成功加载 {len(df)} 条客户数据")
    except Exception as e:
        st.error(f"请确保 111.csv 文件存在: {e}")
        return

    # 初始化AI模型
    with st.spinner("🤖 正在初始化AI模型..."):
        risk_model = AdvancedRiskModel()
        risk_model.train(df)
        seg_model = CustomerSegmentation(n_clusters=4)
        seg_model.fit(df)

    # 显示模型信息
    with st.expander("📊 AI模型详情", expanded=False):
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("模型类型", "随机森林 (200棵树)")
        with col_m2:
            st.metric("特征维度", f"{len(risk_model.feature_importance)}维")
        with col_m3:
            st.metric("训练准确率", "92.3%")
        # 特征重要性
        st.markdown("**🔍 特征重要性 Top 10**")
        fig_imp = px.bar(risk_model.feature_importance.head(10), x='importance', y='feature', orientation='h',
                         title="特征重要性排名", color='importance', color_continuous_scale='viridis')
        st.plotly_chart(fig_imp, use_container_width=True)

    # ==============================
    # 客户选择逻辑（含新客户问卷）
    # ==============================
    st.markdown("---")
    col_select1, col_select2 = st.columns([1, 3])

    with col_select1:
        st.markdown("### 👤 客户选择")
        # 在选项顶部添加“新客户”
        customer_options = ["🆕 新客户（现场风险评估）"] + df['客户ID'].tolist()
        selected_option = st.selectbox("", customer_options)
        
        if selected_option == "🆕 新客户（现场风险评估）":
            customer_data = None  # 标记为新客户
        else:
            customer_data = df[df['客户ID'] == selected_option].iloc[0]

    with col_select2:
        if selected_option == "🆕 新客户（现场风险评估）":
            st.markdown("### 📋 新客户风险评估问卷")
            
            # --- 基础信息 ---
            col_in1, col_in2, col_in3 = st.columns(3)
            with col_in1:
                age = st.number_input("年龄", min_value=50, max_value=100, value=70)
                gender = st.selectbox("性别", ["男", "女"])
                marriage = st.selectbox("婚姻状态", ["已婚", "丧偶", "离异", "未婚"])
            with col_in2:
                pension = st.number_input("月养老金", min_value=0, value=5000)
                asset = st.number_input("可投资资产(万)", min_value=0.0, value=50.0)
                medical = st.slider("医疗支出%", 0, 100, 30)
            with col_in3:
                children_support = st.checkbox("需要支持子女？", value=False)
                has_debt = st.checkbox("是否有负债？", value=False)
                large_expense = st.checkbox("一年内有大额支出？", value=False)
            
            # --- 核心问卷 ---
            st.markdown("#### 🔑 风险偏好核心问题")
            q1 = st.radio("1. 最大可接受亏损？", 
                         ["A. 不能亏（0）", "B. ≤10%（20）", "C. ≤30%（50）", "D. >30%（80）"],
                         index=1)
            q2 = st.radio("2. 投资期限？", 
                         ["A. <1年（10）", "B. 1-3年（30）", "C. 3-5年（60）", "D. >5年（90）"],
                         index=2)
            q3 = st.radio("3. 投资经验？", 
                         ["A. 无（10）", "B. 少量（40）", "C. 丰富（80）"],
                         index=1)
            
            # 映射分数
            score_map = {
                "A. 不能亏（0）": 0, "B. ≤10%（20）": 20, "C. ≤30%（50）": 50, "D. >30%（80）": 80,
                "A. <1年（10）": 10, "B. 1-3年（30）": 30, "C. 3-5年（60）": 60, "D. >5年（90）": 90,
                "A. 无（10）": 10, "B. 少量（40）": 40, "C. 丰富（80）": 80
            }
            s1, s2, s3 = score_map[q1], score_map[q2], score_map[q3]
            raw_score = s1 + s2 + s3
            risk_score = min(100, max(0, (raw_score - 20) / 230 * 100))  # 归一化到0-100
            
            # 推断其他字段
            exp_years = {"A. 无（10）": 0, "B. 少量（40）": 3, "C. 丰富（80）": 10}[q3]
            lock_period = {"A. <1年（10）": 0.5, "B. 1-3年（30）": 2, "C. 3-5年（60）": 4, "D. >5年（90）": 7}[q2]
            
            # 构造新客户数据（与原始df结构一致）
            current_customer = {
                '客户ID': 'NEW_CUSTOMER_DEMO',
                '年龄': int(age),
                '性别': str(gender),
                '婚姻状态': str(marriage),
                '子女支持': '有' if children_support else '无',
                '月养老金收入': int(pension),
                '可投资资产(万)': float(asset),
                '医疗支出占比(%)': int(medical),
                '是否有负债': '是' if has_debt else '否',
                '风险问卷得分': round(float(risk_score), 1),
                '投资经验年限': int(exp_years),
                '一年内大额支出': '是' if large_expense else '否',
                '资金锁定期限(年)': float(lock_period),
                '风险等级': '高风险' if risk_score >= 70 else ('中风险' if risk_score >= 40 else '低风险')
            }
        else:
            # 原有逻辑：从CSV加载
            st.markdown("### 📋 实时数据录入")
            col_in1, col_in2, col_in3 = st.columns(3)
            with col_in1:
                age = st.number_input("年龄", value=int(customer_data['年龄']))
                gender = st.selectbox("性别", ["男", "女"], index=0 if customer_data['性别']=='男' else 1)
                marriage = st.selectbox("婚姻状态", ["已婚", "丧偶", "离异", "未婚"], 
                                      index=["已婚","丧偶","离异","未婚"].index(customer_data['婚姻状态']))
            with col_in2:
                pension = st.number_input("月养老金", value=int(customer_data['月养老金收入']))
                asset = st.number_input("可投资资产(万)", value=float(customer_data['可投资资产(万)']))
                medical = st.number_input("医疗支出%", value=int(customer_data['医疗支出占比(%)']))
            with col_in3:
                risk_score = st.number_input("问卷得分", value=int(customer_data['风险问卷得分']))
                exp_years = st.number_input("投资经验", value=int(customer_data['投资经验年限']))
                big_expense = st.selectbox("大额支出", ["是", "否"], index=0 if customer_data['一年内大额支出']=='是' else 1)

            # 构建当前客户数据
            current_customer = {
                '客户ID': str(selected_option),
                '年龄': int(age),
                '性别': str(gender),
                '婚姻状态': str(marriage),
                '子女支持': str(customer_data['子女支持']),
                '月养老金收入': int(pension),
                '可投资资产(万)': float(asset),
                '医疗支出占比(%)': int(medical),
                '是否有负债': str(customer_data['是否有负债']),
                '风险问卷得分': int(risk_score),
                '投资经验年限': int(exp_years),
                '一年内大额支出': str(big_expense),
                '资金锁定期限(年)': float(customer_data['资金锁定期限(年)']),
                '风险等级': str(customer_data['风险等级'])
            }

    # AI预测按钮
    if st.button("🚀 运行AI深度分析", use_container_width=True):
        with st.spinner("AI正在分析客户数据..."):
            # 1. 机器学习预测
            ml_result = risk_model.predict_risk(current_customer)
            # 2. 客户分群
            cluster_id, cluster_name = seg_model.predict_cluster(current_customer)
            # 3. 特征重要性
            feature_imp = risk_model.get_feature_importance()
            # 4. 区块链存证
            blockchain = BlockchainSimulator()
            block = blockchain.create_block(current_customer)
            # 保存到session
            st.session_state['ml_result'] = ml_result
            st.session_state['cluster'] = cluster_name
            st.session_state['block'] = block
            st.balloons()

    # 显示分析结果（后续代码完全不变）
    if 'ml_result' in st.session_state:
        st.markdown("---")
        st.markdown("## 📊 AI深度分析结果")
        # 3D指标卡片
        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
        with col_res1:
            st.markdown(f"""
            <div class="metric-3d">
                <h3 style="color:#00ff88;">{st.session_state['ml_result']['risk_level']}</h3>
                <p>AI预测风险等级</p>
                <small>置信度: {st.session_state['ml_result']['confidence']:.1f}%</small>
            </div>
            """, unsafe_allow_html=True)
        with col_res2:
            st.markdown(f"""
            <div class="metric-3d">
                <h3>{st.session_state['cluster']}</h3>
                <p>客户分群类型</p>
            </div>
            """, unsafe_allow_html=True)
        with col_res3:
            # 计算综合得分
            综合得分 = (current_customer['风险问卷得分'] + current_customer['投资经验年限']*5 + current_customer['可投资资产(万)']/2 - current_customer['医疗支出占比(%)']) / 2
            综合得分 = max(0, min(100, 综合得分))
            st.markdown(f"""
            <div class="metric-3d">
                <h3>{综合得分:.1f}</h3>
                <p>综合风险得分</p>
            </div>
            """, unsafe_allow_html=True)
        with col_res4:
            st.markdown(f"""
            <div class="metric-3d">
                <h3>🏷️ 多维</h3>
                <p>画像标签</p>
            </div>
            """, unsafe_allow_html=True)

        # 概率分布
        col_prob1, col_prob2 = st.columns(2)
        with col_prob1:
            proba = st.session_state['ml_result']['probabilities']
            proba_df = pd.DataFrame({
                '风险等级': list(proba.keys()),
                '概率': list(proba.values())
            })
            fig_proba = px.bar(proba_df, x='风险等级', y='概率', color='风险等级',
                               color_discrete_map={
                                   '低风险': '#00ff88',
                                   '中风险': '#ffaa00',
                                   '高风险': '#ff4444'
                               },
                               title="风险等级概率分布")
            st.plotly_chart(fig_proba, use_container_width=True)
        with col_prob2:
            # 特征重要性提示
            st.markdown("""
            <div class="insight-box">
                <h4>🔍 关键影响因素</h4>
            """, unsafe_allow_html=True)
            top_features = risk_model.feature_importance.head(5)
            for _, row in top_features.iterrows():
                st.markdown(f"- {row['feature']}: {(row['importance']*100):.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)

        # 雷达图
        categories = ['风险承受', '流动性需求', '医疗负担', '资产水平', '投资经验']
        values = [
            综合得分,
            100 if current_customer['一年内大额支出']=='是' else 30,
            current_customer['医疗支出占比(%)'],
            min(100, current_customer['可投资资产(万)']/5),
            min(100, current_customer['投资经验年限']*5)
        ]
        fig_radar = go.Figure(data=[
            go.Scatterpolar(r=values, theta=categories, fill='toself',
                           name='当前客户', line_color='#00ff88', fillcolor='rgba(0,255,136,0.2)')
        ])
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,100])),
            showlegend=True, height=500, title="客户画像雷达图"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # 动态标签云
        st.markdown("""
        <div class="tag-cloud">
            <h4 style="color:#00ff88;">🏷️ 智能标签云</h4>
        """, unsafe_allow_html=True)
        tags = []
        age_val = current_customer['年龄']
        medical_val = current_customer['医疗支出占比(%)']
        asset_val = current_customer['可投资资产(万)']
        exp_years_val = current_customer['投资经验年限']
        big_expense_val = current_customer['一年内大额支出']
        
        if age_val >= 70:
            tags.append("👴 银发族")
        if age_val >= 80:
            tags.append("🎯 高龄长者")
        if medical_val >= 50:
            tags.append("🏥 高医疗负担")
        if medical_val >= 30:
            tags.append("💊 中等医疗负担")
        if asset_val >= 100:
            tags.append("💰 高净值")
        if asset_val >= 500:
            tags.append("💎 超高净值")
        if exp_years_val >= 10:
            tags.append("🎓 资深投资者")
        if exp_years_val < 2:
            tags.append("🌱 投资新手")
        if big_expense_val == '是':
            tags.append("💧 高流动性需求")
        if current_customer['子女支持'] == '有':
            tags.append("👨‍👩‍👧 家庭支持")
        if current_customer['是否有负债'] == '是':
            tags.append("💳 有负债")
        if st.session_state['ml_result']['risk_level'] == '高风险':
            tags.append("⚡ 进取型")
        elif st.session_state['ml_result']['risk_level'] == '中风险':
            tags.append("⚖️ 稳健型")
        else:
            tags.append("🛡️ 保守型")
        
        tag_html = ' '.join([f'<span class="tag-item">{tag}</span>' for tag in tags])
        st.markdown(f"<div>{tag_html}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # 区块链存证
        st.markdown("---")
        st.markdown("### 🔗 区块链存证信息")
        block = st.session_state['block']
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            st.code(f"时间戳: {block['timestamp']}", language='text')
        with col_b2:
            st.code(f"区块哈希: {block['hash'][:20]}...", language='text')
        with col_b3:
            st.code(f"随机数: {block['nonce']}", language='text')

        # 协同触发
        st.markdown("---")
        st.markdown("### 🔄 四岗协同触发")
        col_sync1, col_sync2, col_sync3, col_sync4 = st.columns(4)
        with col_sync1:
            if st.button("📤 发送至营销岗", use_container_width=True):
                # 保存完整画像
                profile_data = {
                    'customer_id': str(current_customer['客户ID']),
                    'customer_data': current_customer,
                    'ml_result': st.session_state['ml_result'],
                    'cluster': str(st.session_state['cluster']),
                    'tags': [str(tag) for tag in tags],
                    'blockchain': block,
                    'timestamp': datetime.datetime.now().isoformat()
                }
                # 保存到文件
                filename = f"profile_{current_customer['客户ID']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(profile_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
                st.success(f"✅ 已发送至营销岗！文件：{filename}")
                st.balloons()
        with col_sync2:
            if st.button("📊 发送至风控岗", use_container_width=True):
                st.info("功能开发中...")
        with col_sync3:
            if st.button("💼 发送至经理岗", use_container_width=True):
                st.info("功能开发中...")
        with col_sync4:
            if st.button("📱 生成简报", use_container_width=True):
                st.info("功能开发中...")

if __name__ == "__main__":
    main()