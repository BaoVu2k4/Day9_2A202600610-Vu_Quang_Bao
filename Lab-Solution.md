# Lab Solution — Day 09: Multi-Agent A2A System

**MSSV:** 2A202600610  
**Họ tên:** Vũ Quang Bảo

---

## Stage 1: Direct LLM Calling

### Bài Tập 1.1 — Thay đổi câu hỏi

Sửa biến `QUESTION` trong `stages/stage_1_direct_llm/main.py` thành câu hỏi pháp lý tiếng Việt:

```python
QUESTION = "Theo pháp luật Việt Nam, hậu quả pháp lý khi một công ty vi phạm hợp đồng bảo mật thông tin là gì?"
```

**Kết quả:** LLM trả lời dựa thuần vào training data, không có grounding từ database thực tế — có thể thiếu chính xác về điều khoản cụ thể của Luật Việt Nam.

---

### Bài Tập 1.2 — Thêm temperature control

Sửa `common/llm.py`, thêm `temperature=0.3`:

```python
def get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at OpenRouter."""
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "500")),
        temperature=0.3,  # <-- thêm dòng này
    )
```

**Tại sao `0.3`?** Với câu hỏi pháp lý cần độ chính xác cao, temperature thấp giảm "sáng tạo" và giúp output nhất quán hơn giữa các lần chạy.

---

## Stage 2: LLM + RAG & Tools

### Bài Tập 2.1 — Thêm knowledge base entry

Thêm entry về luật lao động vào `LEGAL_KNOWLEDGE` trong `stages/stage_2_rag_tools/main.py`:

```python
{
    "id": "labor_law",
    "keywords": ["lao động", "sa thải", "hợp đồng lao động", "labor", "termination"],
    "text": (
        "Theo Bộ luật Lao động Việt Nam 2019, người sử dụng lao động có thể "
        "đơn phương chấm dứt hợp đồng trong các trường hợp: (1) người lao động "
        "thường xuyên không hoàn thành công việc; (2) bị ốm đau, tai nạn đã điều trị "
        "12 tháng chưa khỏi; (3) thiên tai, hỏa hoạn; (4) người lao động đủ tuổi nghỉ hưu."
    ),
},
```

**Kiểm tra:** Gọi `search_legal_database("sa thải lao động")` → tool trả về entry này.

---

### Bài Tập 2.2 — Tạo tool mới `check_statute_of_limitations`

Thêm vào `stages/stage_2_rag_tools/main.py`:

```python
@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Kiểm tra thời hiệu khởi kiện theo loại vụ án.

    Args:
        case_type: Loại vụ án (contract, tort, property)
    """
    limits = {
        "contract": "4 năm (UCC § 2-725)",
        "tort": "2-3 năm tùy bang",
        "property": "5 năm",
    }
    return limits.get(case_type.lower(), "Không xác định — cần tra cứu thêm")
```

Cập nhật `TOOLS`:
```python
TOOLS = [search_legal_database, calculate_damages, check_statute_of_limitations]
```

**Test:** LLM tự gọi tool khi câu hỏi đề cập đến "thời hiệu" hoặc "statute of limitations".

---

## Stage 3: Single Agent với ReAct

### Bài Tập 3.1 — Thêm tool tra cứu án lệ

Tool `search_case_law` đã được implement trong `stages/stage_3_single_agent/main.py`:

```python
@tool
def search_case_law(keywords: str) -> str:
    """Tìm kiếm án lệ theo từ khóa.

    Args:
        keywords: Từ khóa tìm kiếm (ví dụ: breach, negligence, contract)
    """
    cases = {
        "breach": "Hadley v. Baxendale (1854) — Consequential damages must be foreseeable at contract formation",
        "negligence": "Donoghue v. Stevenson (1932) — Duty of care; manufacturer liable to end consumer",
        "contract": "Carlill v. Carbolic Smoke Ball Co (1893) — Unilateral contract; advertisement = binding offer",
        "fraud": "Derry v. Peek (1889) — Fraud requires knowledge of falsity or reckless disregard for truth",
        "privacy": "Griswold v. Connecticut (1965) — Constitutional right to privacy; foundational for data privacy law",
    }
    keywords_lower = keywords.lower()
    found = [case for kw, case in cases.items() if kw in keywords_lower]
    if found:
        return "\n".join(found)
    return "Không tìm thấy án lệ phù hợp. Thử: breach, negligence, contract, fraud, privacy"
```

Tool này đã có trong `TOOLS` list và agent tự động gọi khi cần tìm án lệ.

---

### Bài Tập 3.2 — Debug agent reasoning

`create_react_agent` không có tham số `verbose`. Thay vào đó, dùng `astream` với `stream_mode="updates"` để xem từng bước (đã implement sẵn trong `main.py`):

```python
async for chunk in graph.astream(inputs, stream_mode="updates"):
    for node_name, update in chunk.items():
        step += 1
        messages = update.get("messages", [])
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                print(f"\n[Step {step}] THINK + ACT (node: {node_name})")
                for tc in msg.tool_calls:
                    print(f"  Tool: {tc['name']}")
                    print(f"  Args: {tc['args']}")
            elif msg.type == "tool":
                print(f"\n[Step {step}] OBSERVE (node: {node_name})")
```

Output streaming này thay thế `verbose=True` — hiển thị từng bước Think-Act-Observe của agent.

---

## Stage 4: Multi-Agent In-Process

### Bài Tập 4.1 — Thêm `privacy_agent`

Thêm vào `stages/stage_4_milti_agent/main.py`:

```python
@tool
def search_privacy_law(query: str) -> str:
    """Search privacy and data protection law knowledge base.

    Args:
        query: Natural language query about privacy or data law.
    """
    knowledge = [
        (
            ["data", "privacy", "gdpr", "ccpa", "consent", "user"],
            "CCPA: fines up to $7,500 per intentional violation. GDPR: up to 4% of global "
            "revenue or EUR 20M. FTC Act Section 5 for unfair/deceptive practices.",
        ),
        (
            ["breach", "notification", "security"],
            "GDPR Art. 33: notify supervisory authority within 72 hours of discovering a "
            "personal data breach. Art. 34: notify affected individuals without undue delay.",
        ),
    ]
    query_lower = query.lower()
    results = []
    for keywords, text in knowledge:
        if any(kw in query_lower for kw in keywords):
            results.append(text)
    return "\n\n".join(results) if results else "No privacy law matches found."


async def call_privacy_specialist(state: LegalState) -> dict:
    """Privacy specialist sub-agent chuyên về GDPR và luật bảo vệ dữ liệu."""
    from langgraph.prebuilt import create_react_agent

    print("\n  [Node: call_privacy_specialist] Privacy specialist agent starting...")

    privacy_prompt = (
        "Bạn là chuyên gia về GDPR và luật bảo vệ dữ liệu cá nhân. "
        "Phân tích các vấn đề về privacy, data protection, consent. "
        "Sử dụng tool search_privacy_law để grounding câu trả lời. "
        "Giới hạn 200 từ."
    )

    llm = get_llm()
    agent = create_react_agent(model=llm, tools=[search_privacy_law], prompt=privacy_prompt)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": state["question"]}]})

    final_msg = result["messages"][-1].content
    print(f"  [Node: call_privacy_specialist] Done ({len(final_msg)} chars)")
    return {"privacy_result": final_msg}
```

Cập nhật `LegalState` để thêm `privacy_result`:
```python
class LegalState(TypedDict):
    ...
    privacy_result: Annotated[str, _last_wins]
    needs_privacy: bool
```

Thêm vào graph:
```python
graph.add_node("call_privacy_specialist", call_privacy_specialist)
graph.add_edge("call_privacy_specialist", "aggregate")
```

---

### Bài Tập 4.2 — Implement conditional routing

Cập nhật `route_to_specialists` để check keyword cho privacy:

```python
def route_to_specialists(state: LegalState) -> list[Send]:
    """Routing function: dispatch parallel Send objects to specialist nodes."""
    question_lower = state["question"].lower()
    sends: list[Send] = []

    if state.get("needs_tax"):
        sends.append(Send("call_tax_specialist", state))

    if state.get("needs_compliance"):
        sends.append(Send("call_compliance_specialist", state))

    if state.get("needs_privacy") or any(
        kw in question_lower
        for kw in ["data", "privacy", "gdpr", "ccpa", "dữ liệu", "riêng tư"]
    ):
        sends.append(Send("call_privacy_specialist", state))

    if not sends:
        sends.append(Send("aggregate", state))
    return sends
```

Cập nhật `check_routing` node để cũng phân tích `needs_privacy`:
```python
# Trong check_routing, thêm needs_privacy vào JSON prompt:
'{"needs_tax": <true|false>, "needs_compliance": <true|false>, "needs_privacy": <true|false>}'
```

**Kết quả:** Agent chỉ gọi `privacy_specialist` khi câu hỏi liên quan đến data/privacy/GDPR — tiết kiệm tokens và thời gian.

---

## Stage 5: Distributed A2A System

### Bài Tập 5.1 — Trace request flow

Chạy `./start_all.sh` và `uv run python test_client.py`, quan sát `trace_id` trong logs.

**Sequence diagram của request flow:**

```
Client          Customer(10100)    Law(10101)    Tax(10102)  Compliance(10103)
  │                   │                │              │             │
  │──POST /a2a────→   │                │              │             │
  │                   │──delegate──→   │              │             │
  │                   │               │──Send(tax)──→ │             │
  │                   │               │──Send(comp)──────────────→  │
  │                   │               │              │ (parallel)   │
  │                   │               │←─result──────│             │
  │                   │               │←──────result──────────────  │
  │                   │               │─aggregate─→  │             │
  │                   │←──final────   │              │             │
  │←─response─────    │                │              │             │
```

`trace_id` giống nhau xuất hiện trong tất cả 5 terminal logs → cho phép correlated tracing.

---

### Bài Tập 5.2 — Test dynamic discovery

1. Dừng Tax Agent: `Ctrl+C` ở terminal Tax Agent (port 10102)
2. Chạy lại: `uv run python test_client.py`

**Kết quả quan sát:**
- Registry vẫn có `tax_agent` trong danh sách (vì agent chưa gọi deregister)
- Law Agent gửi request đến Tax Agent → nhận `ConnectionRefusedError`
- Lỗi truyền về Customer Agent → trả về partial result (chỉ có compliance analysis)
- **Bài học:** Cần implement health check trong Registry để tự động xóa dead agents.

---

### Bài Tập 5.3 — Modify agent behavior

Sửa `tax_agent/graph.py`, thay đổi system prompt để trả lời ngắn gọn hơn:

```python
# Tìm trong tax_agent/graph.py, sửa SYSTEM_PROMPT:
SYSTEM_PROMPT = (
    "You are a tax attorney. Answer in BULLET POINTS only — max 3 bullets, "
    "each under 20 words. Cite the statute code. No introductions."
)
```

Restart tax agent:
```bash
# Dừng tax agent cũ (Ctrl+C), chạy lại:
uv run python -m tax_agent
```

**Kết quả:** Response ngắn hơn → giảm latency tổng thể của hệ thống.

---

## Tổng Kết — So Sánh 5 Stages

| Stage | Pattern | Key Insight |
|-------|---------|-------------|
| 1 | Direct LLM | Stateless, pure training data, dễ bị hallucinate |
| 2 | LLM + Tools | Grounding qua retrieval, nhưng manual tool loop |
| 3 | ReAct Agent | Autonomous multi-step reasoning, sequential |
| 4 | Multi-Agent In-Process | Parallel specialists, LangGraph Send API |
| 5 | Distributed A2A | HTTP-based, independent scaling, service discovery |

**Câu hỏi ôn tập:**

1. **Single vs Multi-agent:** Dùng single agent khi domain đơn giản và không cần parallel; dùng multi-agent khi cần domain expertise riêng hoặc parallel execution.

2. **A2A vs gRPC/REST:** A2A có chuẩn discovery qua Registry và built-in agent card (capability description) — giúp dynamic composition. gRPC/REST thông thường cần hardcode endpoints.

3. **Prevent infinite delegation:** Thêm `max_delegation_depth` counter trong state, reject nếu vượt ngưỡng (vd: 3). Kiểm tra trong supervisor trước khi delegate.

4. **Tại sao cần Registry:** Registry cho phép dynamic discovery — agent mới register lên, các agent khác tự biết. Không cần hardcode URLs, dễ scale và thêm/bớt agents.
