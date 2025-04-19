import streamlit as st
import joblib
import pandas as pd
import numpy as np
# noinspection PyUnresolvedReferences
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import os
import json
import logging
from typing import Optional, Tuple, List, Dict, Any

# -------------------------- 初始化配置 --------------------------
plt.style.use('ggplot')
sns.set_palette("husl")
st.set_page_config(page_title="糖尿病预测系统", layout="wide")

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------------- 自定义样式 --------------------------
st.markdown("""
<style>
/* 消息气泡 */
.message {
    max-width: 80%;
    margin: 10px 0;
    padding: 12px 16px;
    line-height: 1.6;
    font-size: 15px;
    border-radius: 12px;
    word-wrap: break-word;
    animation: fadeIn 0.3s ease-in-out;
}

.user-message {
    background: #f0f7ff;
    color: #0068c9;
    margin-left: auto;
    border: 1px solid #d0e3ff;
}

.ai-message {
    background: #f8f9fa;
    color: #333;
    margin-right: auto;
    border: 1px solid #eee;
}

/* 打字机效果 */
@keyframes cursor-blink {
    0% { opacity: 1; }
    50% { opacity: 0; }
    100% { opacity: 1; }
}

.typing-cursor::after {
    content: "▌";
    animation: cursor-blink 1s infinite;
    color: #666;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* 禁用状态 */
.stButton button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

/* 新增：推荐问题样式 */
.related-questions {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 12px 15px;
    margin: 10px 0;
    border: 1px solid #eee;
}

.related-questions h4 {
    color: #2F4F4F;
    margin-bottom: 8px;
    font-size: 14px;
}

.related-questions ul {
    padding-left: 20px;
    margin: 5px 0;
}

.related-questions li {
    margin: 5px 0;
    cursor: pointer;
    transition: all 0.2s;
}

.related-questions li:hover {
    color: #0068c9;
}

/* 健康卡片 */
.health-card {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
    border: 1px solid #eee;
}
</style>

<style>
/* 新增：推荐问题按钮样式 */
.stButton button {
    background: none !important;
    border: none !important;
    box-shadow: none !important;
    text-align: left !important;
    padding: 8px 12px !important;
    margin: 2px 0 !important;
    color: #444 !important;
    font-weight: normal !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
    justify-content: flex-start !important;
}

.stButton button:hover {
    background: #f0f7ff !important;
    color: #0068c9 !important;
    transform: translateX(5px);
}

.stButton button:active {
    transform: scale(0.98);
}

/* 问题容器样式 */
div[data-testid="stHorizontalBlock"] > div:first-child {
    padding-left: 15px !important;
}
</style>
""", unsafe_allow_html=True)


# -------------------------- 核心功能 --------------------------
@st.cache_data
def load_dataset() -> pd.DataFrame:
    """加载并预处理数据集"""
    try:
        df = pd.read_csv(r"糖尿病风险预测/逐月队-糖尿病风险预测系统/project/project_1/data-01.csv")
        df["风险等级"] = df["Outcome"].map({0: "低风险", 1: "高风险"})
        return df
    except Exception as e:
        logging.error(f"加载数据集失败: {str(e)}")
        st.error("无法加载数据集，请检查文件路径")
        return pd.DataFrame()


@st.cache_resource
def load_models() -> Tuple[Optional[Any], Optional[Any]]:
    """加载机器学习模型和预处理工具"""
    try:
        model = joblib.load(r"糖尿病风险预测/逐月队-糖尿病风险预测系统/project/project_1/random_forest_model.pkl")
        scaler = joblib.load(r"糖尿病风险预测/逐月队-糖尿病风险预测系统/project/project_1/scaler.pkl")
        return model, scaler
    except Exception as e:
        logging.error(f"加载模型失败: {str(e)}")
        st.error("无法加载模型文件，请检查文件路径")
        return None, None


def initialize_session_state():
    """初始化会话状态"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'prediction_data' not in st.session_state:
        st.session_state.prediction_data = None
    if 'is_responding' not in st.session_state:
        st.session_state.is_responding = False


# -------------------------- 流式聊天功能 --------------------------
def generate_related_questions(message: str) -> List[str]:
    """根据消息内容生成推荐问题"""
    question_map = {
        "症状": [
            "这些症状会自行消失吗？",
            "如何区分普通症状与糖尿病症状？",
            "出现症状后多久需要复查？"
        ],
        "饮食": [
            "糖尿病患者能吃水果吗？",
            "适合糖尿病患者的食谱有哪些？",
            "需要完全戒糖吗？"
        ],
        "运动": [
            "什么运动最适合糖尿病患者？",
            "运动前后需要注意什么？",
            "每天需要运动多长时间？"
        ],
        "检查": [
            "需要做哪些检查来确认？",
            "检查前需要空腹吗？",
            "多久需要复查一次？"
        ]
    }

    default_questions = [
        "早期发现后如何自我管理？",
        "这些症状会随着时间加重吗？",
        "家人需要一起做检查吗？"
    ]

    questions = []
    for keyword in question_map:
        if keyword in message:
            questions.extend(question_map[keyword][:2])  # 每个类别最多取2个问题

    # 保证至少有3个问题
    questions = (questions + default_questions)[:3]
    return questions


def render_chat_messages():
    """渲染聊天消息历史（已添加点击问题功能）"""
    for i, (role, msg) in enumerate(st.session_state.chat_history):
        div_class = "user-message" if role == "user" else "ai-message"
        if i == len(st.session_state.chat_history) - 1 and role == "ai" and st.session_state.is_responding:
            st.markdown(
                f'<div class="{div_class} message">{msg}<span class="typing-cursor"></span></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="{div_class} message">{msg}</div>',
                unsafe_allow_html=True
            )

        # 在最后一条AI消息后添加可点击的推荐问题
        if role == "ai" and i == len(st.session_state.chat_history) - 1:
            questions = generate_related_questions(msg)
            with st.container():
                st.markdown("""
                <div style="margin:15px 0 0 10px; padding-left:10px; border-left:2px solid #f0f7ff;">
                    <div style="color:#666; font-size:14px; margin-bottom:8px;">
                        🔍 相关问题推荐
                    </div>
                """, unsafe_allow_html=True)

                # 为每个问题创建可点击按钮
                for q in questions:
                    if st.button(
                            q,
                            key=f"related_q_{hash(q)}",  # 使用哈希值作为唯一键
                            help="点击自动提问",
                            use_container_width=True
                    ):
                        handle_chat_submission(q)
                        st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)


def stream_chat_response(prompt: str, placeholder):
    """处理流式API响应并实时更新界面"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        placeholder.markdown('<div class="ai-message message">⚠️ 系统配置错误：缺少API密钥</div>',
                             unsafe_allow_html=True)
        return

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    data = {
        "model": "deepseek-chat",
        "messages": [{
            "role": "system",
            "content": """你是一个专业的糖尿病健康管理助手，回答时请：
1. 使用简体中文
2. 先给出核心结论（1句话）
3. 分点说明关键要点（最多3点）
4. 每点不超过15字
5. 最后用👉提供行动建议
6. 使用简单易懂的表达方式"""
        }, {
            "role": "user",
            "content": prompt
        }],
        "stream": True,
        "temperature": 0.5
    }

    full_response = ""
    try:
        with requests.post(url, headers=headers, json=data, stream=True, timeout=10) as response:
            response.raise_for_status()

            for chunk in response.iter_lines():
                if chunk:
                    decoded = chunk.decode('utf-8')
                    if decoded.startswith("data: "):
                        json_str = decoded[6:].strip()
                        if json_str == '[DONE]':
                            break
                        try:
                            chunk_data = json.loads(json_str)
                            if 'content' in chunk_data['choices'][0]['delta']:
                                delta = chunk_data['choices'][0]['delta']['content']
                                full_response += delta
                                placeholder.markdown(
                                    f'<div class="ai-message message">{full_response}<span class="typing-cursor"></span></div>',
                                    unsafe_allow_html=True
                                )
                        except json.JSONDecodeError:
                            continue

        placeholder.markdown(
            f'<div class="ai-message message">{full_response}</div>',
            unsafe_allow_html=True
        )
        return full_response

    except requests.exceptions.RequestException as e:
        error_msg = f"⚠️ 网络错误：{str(e)}" if "HTTPSConnectionPool" in str(e) else f"⚠️ 请求失败：{str(e)}"
        placeholder.markdown(f'<div class="ai-message message">{error_msg}</div>', unsafe_allow_html=True)
        return error_msg
    except Exception as e:
        error_msg = f"⚠️ 处理错误：{str(e)}"
        placeholder.markdown(f'<div class="ai-message message">{error_msg}</div>', unsafe_allow_html=True)
        return error_msg


def handle_chat_submission(user_input: str):
    """处理用户提交的聊天消息"""
    if not user_input.strip():
        return

    st.session_state.chat_history.append(("user", user_input.strip()))
    st.session_state.chat_history.append(("ai", "思考中..."))
    st.session_state.is_responding = True

    message_placeholder = st.empty()
    message_placeholder.markdown(
        '<div class="ai-message message">思考中...<span class="typing-cursor"></span></div>',
        unsafe_allow_html=True
    )

    response = stream_chat_response(user_input.strip(), message_placeholder)
    st.session_state.chat_history[-1] = ("ai", response if response else "⚠️ 未能获取响应")
    st.session_state.is_responding = False
    st.rerun()


# -------------------------- 主界面组件 --------------------------
def render_sidebar():
    """渲染侧边栏内容"""
    with st.sidebar:
        st.header("🔍 项目看板")
        st.markdown("""
        **数据集**: 📊 Kaggle糖尿病数据集  
        **核心算法**: 🤖 随机森林分类器  
        **版本信息**: 🚀 v3.4 | 开发者: 逐月队  
        """)
        st.markdown("---")

        st.subheader("📈 数据洞察分析")
        diabetes_data = load_dataset()

        if not diabetes_data.empty:
            col1, col2 = st.columns(2)
            with col1:
                dist_feature = st.selectbox("分析特征", options=["血糖值", "BMI", "年龄", "胰岛素"])
            with col2:
                chart_type = st.selectbox("图表类型", options=["分布图", "箱线图"])

            fig, ax = plt.subplots(figsize=(8, 4))
            try:
                if dist_feature == "血糖值":
                    if chart_type == "分布图":
                        sns.histplot(data=diabetes_data, x="Glucose", hue="风险等级", kde=True, palette="viridis",
                                     ax=ax)
                    else:
                        sns.boxplot(data=diabetes_data, x="风险等级", y="Glucose", palette="coolwarm", ax=ax)
                    ax.set_xlabel("血糖值 (mg/dL)")
                elif dist_feature == "BMI":
                    sns.violinplot(data=diabetes_data, x="风险等级", y="BMI", palette="Spectral", ax=ax)
                    ax.set_ylabel("BMI 指数")
                elif dist_feature == "年龄":
                    sns.countplot(data=diabetes_data, x="Age", hue="风险等级", palette="rocket", ax=ax)
                    ax.set_xlabel("年龄分布")
                elif dist_feature == "胰岛素":
                    sns.scatterplot(data=diabetes_data, x="Insulin", y="Glucose", hue="风险等级", palette="mako", ax=ax)
                    ax.set_xlabel("胰岛素 (μU/mL)")

                plt.title(f"{dist_feature} - 风险分布分析", fontsize=12)
                ax.grid(alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"图表生成失败: {str(e)}")

            with st.expander("📊 数据集统计摘要"):
                st.dataframe(diabetes_data.describe().T.style.background_gradient(cmap="Blues"))


def render_prediction_form():
    """渲染预测表单"""
    st.title("🩸 糖尿病风险智能预测系统")
    st.caption("基于机器学习的临床辅助决策工具")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            glucose = st.number_input("血糖值（mg/dL）", 0, 300, 100)
            blood_pressure = st.number_input("血压（mmHg）", 40, 200, 70)
        with col2:
            insulin = st.number_input("胰岛素（μU/mL）", 0, 1000, 80)
            bmi = st.number_input("BMI指数", 10.0, 50.0, 25.0, step=0.1)
        with col3:
            dpf = st.number_input("遗传风险系数", 0.0, 2.5, 0.5, step=0.001)
            age = st.number_input("患者年龄", 0, 100, 30)

        if st.form_submit_button("✨ 开始智能分析"):
            with st.spinner('🔍 正在分析数据...'):
                model, scaler = load_models()
                if model is None or scaler is None:
                    st.error("模型加载失败，无法进行分析")
                    return

                input_data = pd.DataFrame([[glucose, blood_pressure, insulin, bmi, dpf, age]],
                                          columns=["Glucose", "BloodPressure", "Insulin", "BMI",
                                                   "DiabetesPedigreeFunction", "Age"])

                try:
                    scaled_data = scaler.transform(input_data)
                    prediction = model.predict(scaled_data)[0]
                    proba = model.predict_proba(scaled_data)[0][1] * 100

                    st.session_state.prediction_data = {
                        'proba': proba,
                        'risk_level': "高风险" if prediction == 1 else "低风险",
                        'glucose': glucose,
                        'blood_pressure': blood_pressure,
                        'bmi': bmi,
                        'insulin': insulin,
                        'age': age
                    }

                    if 'auto_reply_sent' not in st.session_state:
                        st.session_state.auto_reply_sent = False

                    if not st.session_state.auto_reply_sent:
                        st.session_state.auto_reply_sent = True
                        handle_chat_submission(f"我的糖尿病风险是{proba:.1f}%，请给我健康建议")

                    risk_color = "#FF6B6B" if prediction == 1 else "#20B2AA"
                    st.markdown(f"""
                    <div class="result-card">
                        <h3 style="color:#2F4F4F; margin-bottom:1rem">📊 分析结果</h3>
                        <div style="font-size:1.2rem; margin-bottom:1rem">
                            患病概率: <b style="color:{risk_color}">{proba:.1f}%</b>
                        </div>
                        <div style="display: flex; align-items: center; gap: 1rem">
                            {"⚠️" if prediction == 1 else "✅"}
                            <span style="font-size:1.3rem; color:{risk_color}">
                                {"高风险！建议立即就医检查" if prediction == 1 else "低风险，请保持健康习惯"}
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.success("分析完成！已自动生成健康建议")
                except Exception as e:
                    logging.error(f"预测失败: {str(e)}")
                    st.error(f"分析失败: {str(e)}")


def render_quick_tools():
    """渲染快捷工具面板"""
    st.markdown("### 🚀 快捷工具")

    if st.session_state.get('prediction_data'):
        if st.button("获取个性化建议",
                     key="personal_advice",
                     disabled=st.session_state.is_responding):
            handle_chat_submission(
                f"根据我的检测结果（风险{st.session_state.prediction_data['proba']:.1f}%），请给我个性化建议"
            )

    preset_questions = {
        "糖尿病早期症状": "糖尿病早期有哪些典型症状？",
        "饮食建议": "糖尿病患者适合吃什么食物？",
        "运动方案": "糖尿病患者推荐的运动方式和频率？",
        "检测建议": "糖尿病患者应该定期做哪些检查？"
    }

    for btn_text, question in preset_questions.items():
        if st.button(btn_text,
                     key=f"preset_{btn_text}",
                     disabled=st.session_state.is_responding):
            handle_chat_submission(question)

    st.markdown("---")
    st.markdown("### 📊 健康标准")
    st.markdown("""
    <div class="health-card">
    <b>血糖指标</b>
    - 空腹: 4.4-6.1 mmol/L
    - 餐后: <7.8 mmol/L<br>
    <b>BMI指数</b>
    - 正常范围: 18.5-24.9
    </div>
    """, unsafe_allow_html=True)


def diabetes_chatbot():
    """糖尿病聊天机器人主组件"""
    with st.container():
        cols = st.columns([0.75, 0.25])

        with cols[0]:
            st.markdown("## 📝 健康咨询对话")
            render_chat_messages()

            with st.form("chat_form"):
                user_input = st.text_input(
                    "输入您的问题...",
                    key="user_input",
                    label_visibility="collapsed",
                    placeholder="例如：糖尿病早期有哪些症状？"
                )

                submit_button = st.form_submit_button(
                    "发送",
                    disabled=st.session_state.is_responding
                )

                if submit_button and user_input.strip():
                    handle_chat_submission(user_input.strip())

        with cols[1]:
            render_quick_tools()


# -------------------------- 主函数 --------------------------
def main():
    """主入口函数"""
    initialize_session_state()
    render_sidebar()
    render_prediction_form()
    diabetes_chatbot()


if __name__ == "__main__":
    main()