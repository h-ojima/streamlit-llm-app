from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# シークレットキーの取得
load_dotenv()

# --- システムメッセージ定義 ---
SYSTEM_MESSAGES = {
    "適応障害": (
        "あなたは適応障害の専門家です。"
        "適応障害の症状・原因・治療法・日常生活でのケアについて、"
        "正確かつ共感的に答えてください。"
        "医学的なアドバイスはあくまで参考情報であり、"
        "具体的な診断や治療は医師に相談するよう促してください。"
    ),
    "躁鬱（双極性障害）": (
        "あなたは双極性障害（躁鬱病）の専門家です。"
        "躁状態・鬱状態それぞれの特徴、治療薬、生活管理のポイントについて、"
        "正確かつ共感的に答えてください。"
        "医学的なアドバイスはあくまで参考情報であり、"
        "具体的な診断や治療は医師に相談するよう促してください。"
    ),
}


def get_llm_response(user_input: str, expert_type: str) -> str:
    """
    入力テキストと専門家の種類を受け取り、LLMからの回答を返す関数。

    Args:
        user_input: ユーザーが入力したテキスト
        expert_type: ラジオボタンで選択された専門家の種類

    Returns:
        LLMからの回答文字列
    """
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    messages = [
        SystemMessage(content=SYSTEM_MESSAGES[expert_type]),
        HumanMessage(content=user_input),
    ]

    result = llm(messages)
    return result.content


# --- Streamlit UI ---
st.title("🧠 精神疾患 専門家AIチャット")

st.markdown(
    """
## このアプリについて
精神疾患に関する疑問を、専門家AIに質問できるアプリです。

### 操作方法
1. **専門家を選択**してください（適応障害 or 躁鬱）
2. **質問を入力**してください
3. **「回答を得る」ボタン**を押すと、選択した専門家の視点でAIが回答します

> ⚠️ このアプリの回答はAIによる参考情報です。実際の診断・治療は必ず医師にご相談ください。
"""
)

st.divider()

# 専門家の選択（ラジオボタン）
expert_type = st.radio(
    "相談したい専門家を選択してください",
    options=list(SYSTEM_MESSAGES.keys()),
    horizontal=True,
)

# テキスト入力フォーム
user_input = st.text_area(
    "質問を入力してください",
    placeholder="例：適応障害になったときの職場への伝え方を教えてください。",
    height=150,
)

# 送信ボタン
if st.button("回答を得る", type="primary"):
    if not user_input.strip():
        st.error("⚠️ 質問が入力されていません。テキストを入力してから送信してください。")
    else:
        try:
            with st.spinner("回答を生成中..."):
                response = get_llm_response(user_input, expert_type)
            st.subheader(f"💬 {expert_type}専門家からの回答")
            st.markdown(response)
        except Exception as e:
            st.error(f"⚠️ 回答の生成中にエラーが発生しました。しばらく待ってから再度お試しください。\n\n（エラー詳細: {e}）")