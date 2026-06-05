import os
import json
from datetime import datetime
from typing import Optional
from uuid import uuid4

import streamlit as st
from openai import OpenAI


LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_results.jsonl")


# =========================
# 系统提示词
# 说明：根据你的新需求，输出改为“单段自然语言”，不再使用1/2/3小标题。
# =========================
SYSTEM_PROMPT_1 = """你是一个人际行为分析工具，不是情感陪伴。原则：

1. 不替对方补动机。任何动机都必须写成“可能性 + 证据强度”，
   不允许输出“他其实是在乎你的”这类替对方兜底的结论。

2. 区分两类行为：
   - [针对性行为] 仅对用户表现
   - [固有模式] 对所有人都这样
   没有对照数据时，必须标注“无法判断：缺少他对他人的行为样本”。

3. 证据强度分级：
   - 强：多次出现 / 有明确对照
   - 中：单次但信息量高（如主动修正、主动暴露）
   - 弱：单次且常见行为（如忙、失联、延迟）

4. 安抚最多一句，且不得包含正向结论。

5. 样本不足时直接说“画像可信度低”，不许凑结论。

请按以下结构输出：
【情绪校准】最多一句，不下结论。
【行为拆解】列出2-4条，每条都要写“观察到的行为 -> 可能性 -> 证据强度（强/中/弱）”。
【针对性 vs 固有模式】逐条标注；若缺少对照样本，必须写“无法判断：缺少他对他人的行为样本”。
【当前画像可信度】明确写“高/中/低”，并说明为什么。
【下一步取证】给2条具体可观察信号，格式为“看到A，更支持X；看到B，更支持Y”。
【可选观察维度】可给“大五/依恋”等仅作观察方向，必须标注“不可用于判定人格类型”。

若用户流露严重心理危机（如伤害自己念头），温柔建议其尽快联系专业帮助或可信任的人。"""

SYSTEM_PROMPT_2 = """你是同一个人际行为分析工具。现在要根据新的互动记录，更新对这个人的理解。
你会收到：之前的画像、历史记录、这次的新记录。

更新原则：
1. 不替对方补动机；任何动机都写成“可能性 + 证据强度”。
2. 区分[针对性行为]/[固有模式]；没有对照样本就写“无法判断：缺少他对他人的行为样本”。
3. 证据强度分级：强/中/弱（定义同上）。
4. 安抚最多一句，且不得包含正向结论。
5. 样本不足直接写“画像可信度低”，不凑结论。

请输出：
【情绪校准】最多一句。
【新增证据如何改变判断】哪些被加强，哪些被削弱，哪些转为“待定”。
【仅更新变化部分】用“观察 -> 可能性 -> 证据强度”写出变化，不要全文重写。
【针对性 vs 固有模式】本次新增内容逐条标注；无对照样本就明确“无法判断：缺少他对他人的行为样本”。
【当前画像可信度】高/中/低，并给理由。
【下一步取证】给2条“看到A支持X，看到B支持Y”的可观察信号。

可选给“大五/依恋”等观察维度，但只能作为观察方向，禁止当作定型结论；禁止MBTI、星座和标签化判定。
若用户流露严重心理危机，温柔建议其尽快联系专业帮助或可信任的人。"""

SYSTEM_PROMPT_3 = """你是人际行为分析工具。你要基于用户给出的全部信息（初始建档、历史记录、当前画像），输出一个可用于展示的全量总结。
要求：
1) 不补动机，不下绝对结论；任何判断都要体现证据强弱。
2) 如果样本不足，明确写“画像可信度低”。
3) 人格分类模型使用“大五（Big Five）+ 依恋线索（仅观察方向）”。禁止MBTI、星座。
4) 你必须只输出 JSON，不要输出任何额外文字。

JSON 格式严格如下：
{
  "person_summary": "基于所有记录对这个人的重述画像（150字内）",
  "confidence": "高/中/低",
  "personality_model": {
    "name": "Big Five + 依恋线索",
    "dimensions": [
      {"name": "开放性", "score": 0-100, "evidence": "强/中/弱", "note": "一句依据"},
      {"name": "尽责性", "score": 0-100, "evidence": "强/中/弱", "note": "一句依据"},
      {"name": "外向性", "score": 0-100, "evidence": "强/中/弱", "note": "一句依据"},
      {"name": "宜人性", "score": 0-100, "evidence": "强/中/弱", "note": "一句依据"},
      {"name": "情绪稳定性", "score": 0-100, "evidence": "强/中/弱", "note": "一句依据"}
    ],
    "attachment_clues": "依恋相关线索，必须写清样本是否不足"
  },
  "diagram_mermaid": "一段 mermaid graph TD 代码，用于展示上述维度高低和证据强弱",
  "me_pattern": "总结用户在这段关系里的模式：偏主动/偏被动/摇摆，并给一句依据",
  "next_observation": "下一步最关键取证点（1句话）"
}
"""


def init_session_state() -> None:
    """初始化所有会在内存中使用的数据结构。"""
    if "profile" not in st.session_state:
        # 建档信息：谁、困惑点、特征、印象事件
        st.session_state.profile = {}

    if "portrait" not in st.session_state:
        # 当前画像文本（首次生成后会有内容）
        st.session_state.portrait = ""

    if "records" not in st.session_state:
        # 历史互动记录列表（按追加顺序即时间顺序）
        st.session_state.records = []

    if "global_report" not in st.session_state:
        # 全量总结（对象画像 + 人格维度 + 我方关系模式）
        st.session_state.global_report = None

    if "session_id" not in st.session_state:
        # 区分不同访问者会话，便于回看测试数据
        st.session_state.session_id = str(uuid4())

    if "tester_name" not in st.session_state:
        # 测试昵称（可选）
        st.session_state.tester_name = ""


def get_openai_client() -> Optional[OpenAI]:
    """从环境变量读取 API Key，并创建 OpenAI 客户端。"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def call_gpt4o(system_prompt: str, user_content: str) -> str:
    """统一封装一次 GPT-4o 调用，返回模型文本。"""
    client = get_openai_client()
    if client is None:
        raise RuntimeError("未检测到 OPENAI_API_KEY，请先设置环境变量。")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.7,
    )

    # 常规情况下第一条 choice 即可满足 MVP 需求
    return response.choices[0].message.content or "(模型没有返回文本)"


def parse_json_response(raw_text: str) -> dict:
    """兼容模型偶发返回 ```json 包裹内容的情况。"""
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    return json.loads(text)


def generate_global_report(profile: dict, portrait: str, records: list) -> dict:
    """基于全部记录生成全量总结（结构化 JSON）。"""
    history_lines = []
    for i, r in enumerate(records, start=1):
        history_lines.append(
            f"[{i}] 时间：{r['time']} | 感受：{r['feeling']} | 事件：{r['event']} | 关键一句：{r['key_point']}"
        )
    history_text = "\n".join(history_lines) if history_lines else "（暂无历史记录）"

    user_content = (
        "请根据以下完整样本生成 JSON：\n"
        f"【建档信息】\n{json.dumps(profile, ensure_ascii=False)}\n\n"
        f"【当前画像】\n{portrait}\n\n"
        f"【历史记录】\n{history_text}\n"
    )

    raw = call_gpt4o(SYSTEM_PROMPT_3, user_content)
    return parse_json_response(raw)


def append_test_log(event_type: str, payload: dict) -> None:
    """将测试行为写入本机 JSONL，便于你回看朋友的使用结果。"""
    log_item = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": st.session_state.get("session_id", "unknown"),
        "tester_name": st.session_state.get("tester_name", "") or "未填写",
        "event_type": event_type,
        "payload": payload,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_item, ensure_ascii=False) + "\n")


def read_test_logs(limit: int = 200) -> list:
    """读取最近若干条测试日志。"""
    if not os.path.exists(LOG_FILE):
        return []

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    logs = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            logs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return logs


def require_access_password() -> None:
    """可选访问密码保护：设置 APP_ACCESS_PASSWORD 后生效。"""
    expected_password = os.getenv("APP_ACCESS_PASSWORD")
    if not expected_password:
        st.caption("当前未启用访问密码（可设置 APP_ACCESS_PASSWORD 启用）。")
        return

    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False

    if st.session_state.auth_ok:
        return

    st.warning("此测试链接已启用访问密码。")
    pwd = st.text_input("请输入访问密码", type="password")
    if st.button("进入应用"):
        if pwd == expected_password:
            st.session_state.auth_ok = True
            st.rerun()
        else:
            st.error("密码不正确。")
    st.stop()


def render_history() -> None:
    """展示历史互动记录（时间顺序）。"""
    st.sidebar.header("历史互动记录")
    if not st.session_state.records:
        st.sidebar.info("还没有记录。")
        return

    for idx, record in enumerate(st.session_state.records, start=1):
        st.sidebar.markdown(
            f"**{idx}. {record['time']}**\n"
            f"- 感受：{record['feeling']}\n"
            f"- 发生了什么：{record['event']}\n"
            f"- TA关键一句：{record['key_point']}"
        )
        st.sidebar.markdown("---")


def render_test_logs_panel() -> None:
    """在侧边栏展示回看入口。"""
    st.sidebar.header("测试回看")
    with st.sidebar.expander("查看最近测试结果", expanded=False):
        logs = read_test_logs(limit=200)
        if not logs:
            st.info("还没有日志。")
            return

        st.caption(f"本机已记录 {len(logs)} 条（最多显示最近200条）")
        for item in reversed(logs[-20:]):
            st.markdown(
                f"**{item.get('time', '')}** | "
                f"{item.get('tester_name', '未填写')} | "
                f"{item.get('event_type', '')}"
            )

        with open(LOG_FILE, "r", encoding="utf-8") as f:
            raw = f.read()
        st.download_button(
            label="下载全部日志(JSONL)",
            data=raw,
            file_name="test_results.jsonl",
            mime="application/json",
        )


def is_admin_view() -> bool:
    """仅当 URL 带 ?admin=1/true 时显示管理面板。"""
    admin_value = ""

    # Streamlit 新版
    try:
        if hasattr(st, "query_params"):
            admin_value = str(st.query_params.get("admin", ""))
    except Exception:
        admin_value = ""

    # 兼容旧版 Streamlit
    if not admin_value:
        try:
            params = st.experimental_get_query_params()
            raw_value = params.get("admin", [""])
            admin_value = raw_value[0] if isinstance(raw_value, list) and raw_value else str(raw_value)
        except Exception:
            admin_value = ""

    return admin_value.lower() in {"1", "true", "yes", "admin"}


def main() -> None:
    st.set_page_config(page_title="人际关系日记本 MVP", page_icon="📝", layout="wide")
    st.title("📝 人际关系日记本（MVP）")
    st.caption("目标：帮你理解一个让你困惑的人，并随着记录不断更新画像。")

    init_session_state()
    require_access_password()

    st.text_input("你的测试昵称（可选）", key="tester_name", placeholder="例如：小王")

    render_history()
    if is_admin_view():
        render_test_logs_panel()

    if not os.getenv("OPENAI_API_KEY"):
        st.warning("未检测到 OPENAI_API_KEY。请先设置环境变量后再调用模型。")

    # =========================
    # 功能1：初次建档
    # =========================
    st.subheader("功能1：初次建档")
    with st.form("create_profile_form"):
        who = st.text_input("这个人是谁（关系/认识多久）", placeholder="例如：同事，认识两年")
        why_hard = st.text_area("为什么觉得 TA 难懂", placeholder="例如：忽冷忽热，不太表达真实想法")
        traits = st.text_area("TA 的一些特征", placeholder="例如：做事利落、话不多、回消息慢")
        deep_event = st.text_area("最近一件印象深的事", placeholder="例如：上周我主动约聊，TA临时取消但后来又送了咖啡")
        submitted_profile = st.form_submit_button("生成初始画像")

    if submitted_profile:
        if not (who and why_hard and traits and deep_event):
            st.error("请先完整填写四项建档信息。")
        else:
            st.session_state.profile = {
                "who": who,
                "why_hard": why_hard,
                "traits": traits,
                "deep_event": deep_event,
            }

            user_content = (
                "请基于以下信息给出初始画像：\n"
                f"- 这个人是谁（关系/认识多久）：{who}\n"
                f"- 为什么觉得 TA 难懂：{why_hard}\n"
                f"- TA 的一些特征：{traits}\n"
                f"- 最近一件印象深的事：{deep_event}\n"
            )

            with st.spinner("正在生成初始画像..."):
                try:
                    portrait_text = call_gpt4o(SYSTEM_PROMPT_1, user_content)
                    st.session_state.portrait = portrait_text

                    append_test_log(
                        event_type="initial_portrait_generated",
                        payload={
                            "profile": st.session_state.profile,
                            "portrait": portrait_text,
                        },
                    )

                    st.success("初始画像已生成。")
                except Exception as e:
                    st.error(f"生成失败：{e}")

    if st.session_state.portrait:
        st.markdown("### 当前画像")
        st.write(st.session_state.portrait)

    # =========================
    # 功能2：互动记录与画像更新
    # =========================
    st.subheader("功能2：记录新互动并更新画像")

    if not st.session_state.portrait:
        st.info("请先完成建档并生成初始画像，再进行互动记录更新。")
    else:
        with st.form("new_record_form"):
            feeling = st.radio("这次的感受", ["开心", "平淡", "不爽", "困惑"], horizontal=True)
            event = st.text_area("发生了什么", placeholder="描述这次互动经过")
            key_point = st.text_area("TA 说的关键一点", placeholder="尽量写原话或最核心的一句")
            submitted_record = st.form_submit_button("更新画像")

        if submitted_record:
            if not (event and key_point):
                st.error("请填写“发生了什么”和“TA 说的关键一点”。")
            else:
                new_record = {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "feeling": feeling,
                    "event": event,
                    "key_point": key_point,
                }

                # 将历史记录组织成清晰文本，一次性喂给模型
                history_lines = []
                for i, r in enumerate(st.session_state.records, start=1):
                    history_lines.append(
                        f"[{i}] 时间：{r['time']} | 感受：{r['feeling']} | 发生了什么：{r['event']} | TA关键一句：{r['key_point']}"
                    )
                history_text = "\n".join(history_lines) if history_lines else "（暂无历史记录）"

                user_content = (
                    "请根据以下信息做渐进式画像更新：\n"
                    f"【之前的画像】\n{st.session_state.portrait}\n\n"
                    f"【历史所有记录】\n{history_text}\n\n"
                    f"【这次新记录】\n"
                    f"- 时间：{new_record['time']}\n"
                    f"- 感受：{new_record['feeling']}\n"
                    f"- 发生了什么：{new_record['event']}\n"
                    f"- TA关键一句：{new_record['key_point']}\n"
                )

                with st.spinner("正在更新画像..."):
                    try:
                        updated_portrait = call_gpt4o(SYSTEM_PROMPT_2, user_content)

                        # 成功后再写入内存：更新画像 + 追加历史记录
                        st.session_state.portrait = updated_portrait
                        st.session_state.records.append(new_record)

                        # 在“更新画像”后，自动补充全量对象总结与关系模式总结
                        try:
                            st.session_state.global_report = generate_global_report(
                                st.session_state.profile,
                                st.session_state.portrait,
                                st.session_state.records,
                            )
                        except Exception as report_error:
                            st.warning(f"全量总结生成失败：{report_error}")

                        append_test_log(
                            event_type="portrait_updated",
                            payload={
                                "new_record": new_record,
                                "updated_portrait": st.session_state.portrait,
                                "global_report": st.session_state.global_report,
                            },
                        )

                        st.success("画像已更新，并已保存这次互动记录。")
                    except Exception as e:
                        st.error(f"更新失败：{e}")

        # 只有真正发生过“互动更新”后，才展示“更新后的画像”，避免初次建档时误导用户。
        if st.session_state.portrait and st.session_state.records:
            st.markdown("### 更新后的画像")
            st.write(st.session_state.portrait)

            st.markdown("### 当前对象全量总结（基于全部记录）")

            if st.button("重新生成全量总结"):
                with st.spinner("正在重算全量总结..."):
                    try:
                        st.session_state.global_report = generate_global_report(
                            st.session_state.profile,
                            st.session_state.portrait,
                            st.session_state.records,
                        )

                        append_test_log(
                            event_type="global_report_regenerated",
                            payload={
                                "global_report": st.session_state.global_report,
                            },
                        )

                        st.success("全量总结已更新。")
                    except Exception as e:
                        st.error(f"重算失败：{e}")

            report = st.session_state.global_report
            if report:
                st.write(report.get("person_summary", ""))
                st.caption(f"当前画像可信度：{report.get('confidence', '未知')}")

                model = report.get("personality_model", {})
                st.markdown(f"**人格分类模型：{model.get('name', 'Big Five + 依恋线索')}**")

                for dim in model.get("dimensions", []):
                    name = dim.get("name", "未命名维度")
                    score = int(dim.get("score", 0))
                    score = max(0, min(100, score))
                    evidence = dim.get("evidence", "未知")
                    note = dim.get("note", "")

                    st.write(f"{name}：{score}/100（证据：{evidence}）")
                    st.progress(score)
                    if note:
                        st.caption(note)

                attachment = model.get("attachment_clues", "")
                if attachment:
                    st.write(f"依恋线索（仅观察方向）：{attachment}")

                diagram = report.get("diagram_mermaid", "")
                if diagram:
                    st.markdown("**示意图代码（Mermaid）**")
                    st.code(diagram, language="mermaid")

                me_pattern = report.get("me_pattern", "")
                if me_pattern:
                    st.markdown("**我在这段关系里的模式**")
                    st.write(me_pattern)

                next_obs = report.get("next_observation", "")
                if next_obs:
                    st.markdown("**下一步关键取证点**")
                    st.write(next_obs)
            else:
                st.info("你更新一次互动后，这里会自动生成“对象全量总结”和“我方关系模式”。")


if __name__ == "__main__":
    main()
