# Báo Cáo Thực Hành Day 9: Multi-Agent System với A2A Protocol

**Sinh viên:** Vu Quang Bao  
**MSSV:** 2A202600610  
**Ngày:** 2026-06-09

---

## 1. Tổng Quan Những Gì Đã Học

Buổi thực hành hôm nay đi theo lộ trình 5 giai đoạn, từ cách dùng LLM đơn giản nhất đến hệ thống multi-agent phân tán:

```
Stage 1: Gọi LLM trực tiếp
    ↓  (thêm khả năng tra cứu dữ liệu)
Stage 2: LLM + RAG & Tools
    ↓  (tự động hóa vòng lặp)
Stage 3: Single Agent (ReAct)
    ↓  (chuyên môn hóa, song song)
Stage 4: Multi-Agent trong 1 process
    ↓  (tách thành services độc lập)
Stage 5: Distributed A2A System
```

---

## 2. Giải Thích Từng Stage

### Stage 1 — Direct LLM Calling

**Cách hoạt động:** Gửi câu hỏi thẳng đến LLM, nhận câu trả lời.

```python
llm = get_llm()
messages = [SystemMessage(content="Bạn là chuyên gia pháp lý..."),
            HumanMessage(content=QUESTION)]
response = await llm.ainvoke(messages)
```

**Hạn chế:** LLM chỉ dựa vào dữ liệu huấn luyện, không tra cứu được thông tin cụ thể, không tính toán được.

---

### Stage 2 — LLM + Tools (Function Calling)

**Cách hoạt động:** LLM được trang bị "công cụ" — các hàm Python mà nó có thể yêu cầu chạy.

```
LLM nhận câu hỏi + danh sách tools
    → LLM quyết định gọi tool nào
    → Code thực thi tool, trả kết quả
    → LLM tổng hợp câu trả lời cuối
```

**Ví dụ tool:**
```python
@tool
def search_legal_database(query: str) -> str:
    """Tìm trong knowledge base pháp lý."""
    # tìm kiếm và trả về kết quả
```

**Cải thiện:** Câu trả lời được grounded (có trích dẫn cụ thể), giảm hallucination.

---

### Stage 3 — Single ReAct Agent

**ReAct = Reasoning + Acting**

Thay vì gọi tool 1 lần thủ công, agent tự lặp lại:

```
Think → "Tôi cần tra luật về NDA trước"
Act   → gọi search_legal_database("NDA breach")
Observe → nhận kết quả
Think → "Cần tính thêm thiệt hại"
Act   → gọi calculate_penalty(...)
Observe → nhận kết quả
Think → "Đủ thông tin rồi, tổng hợp trả lời"
→ Final Answer
```

**Dùng `create_react_agent` của LangGraph** để tự động hóa toàn bộ vòng lặp này.

---

### Stage 4 — Multi-Agent (In-Process)

**Vấn đề của Stage 3:** Một agent phải xử lý mọi domain (luật, thuế, compliance) → không chuyên sâu.

**Giải pháp:** Tách thành nhiều agents chuyên môn, chạy **song song** bằng LangGraph `Send` API:

```
analyze_law (Lead Attorney)
      ↓
check_routing → [needs_tax=True, needs_compliance=True]
      ↓
┌─────┴─────┐
tax_agent   compliance_agent   ← chạy SONG SONG
└─────┬─────┘
      ↓
  aggregate (tổng hợp)
```

**Key concept — `Send` API:**
```python
def route_to_specialists(state) -> list[Send]:
    sends = []
    if state["needs_tax"]:
        sends.append(Send("tax_agent", state))      # dispatch song song
    if state["needs_compliance"]:
        sends.append(Send("compliance_agent", state))
    return sends
```

**Annotated reducer** để xử lý parallel writes:
```python
tax_result: Annotated[str, _last_wins]  # 2 agents cùng ghi → không conflict
```

---

### Stage 5 — Distributed A2A System

**Vấn đề của Stage 4:** Tất cả agents trong 1 process → không scale, 1 agent crash = cả hệ thống crash.

**Giải pháp:** Mỗi agent là một **HTTP service độc lập**, giao tiếp qua **A2A Protocol**.

**Kiến trúc:**
```
Registry :10000          ← trung tâm đăng ký & tìm kiếm
     ↑ (register)
Customer Agent :10100    ← nhận câu hỏi từ user
     ↓ (A2A HTTP)
Law Agent :10101         ← phân tích pháp lý + điều phối
     ↓ (A2A HTTP, song song)
Tax Agent :10102         ← chuyên gia thuế
Compliance Agent :10103  ← chuyên gia tuân thủ
```

**Dynamic Discovery — không hardcode URL:**
```python
# Thay vì: endpoint = "http://localhost:10101"  ← cứng nhắc
endpoint = await discover("legal_question")      # ← linh hoạt, qua Registry
```

**Trace Propagation — theo dõi request xuyên suốt:**
```python
# trace_id được truyền qua mọi hop A2A
metadata={"trace_id": trace_id, "delegation_depth": depth}
```

---

## 3. Bài Tập Đã Hoàn Thành

### Exercise 2 — Thêm Tools và Knowledge Base

**Bài 2.1:** Thêm entry về luật lao động Việt Nam vào knowledge base.

```python
{
    "id": "labor_law",
    "keywords": ["lao động", "sa thải", "hợp đồng lao động", ...],
    "text": "Theo Bộ luật Lao động Việt Nam 2019, người sử dụng lao động
             có thể đơn phương chấm dứt hợp đồng trong các trường hợp: ..."
}
```

**Bài 2.2:** Tạo tool `check_statute_of_limitations`.

```python
@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Kiểm tra thời hiệu khởi kiện theo loại vụ án."""
    limits = {
        "contract": "4 năm (UCC § 2-725)",
        "tort":     "2-3 năm tùy bang",
        "labor":    "1 năm (BLLĐ 2019 Việt Nam)",
        "nda":      "3 năm (DTSA, 18 U.S.C. § 1836)",
    }
    return limits.get(case_type.lower(), "Không xác định")
```

**Kết quả chạy:** LLM tự nhận ra câu hỏi về thời hiệu → tự gọi đúng tool → trả lời chính xác với trích dẫn pháp luật.

---

### Exercise 4 — Thêm Privacy Agent

**Nhiệm vụ:** Mở rộng hệ thống multi-agent với agent chuyên về GDPR/bảo vệ dữ liệu.

**Implement `privacy_agent`:**
```python
def privacy_agent(state: State) -> dict:
    """Agent chuyên về bảo vệ dữ liệu cá nhân và GDPR."""
    llm = get_llm()
    prompt = f"""Bạn là chuyên gia về GDPR và privacy law.
    Tập trung: GDPR (phạt 4% doanh thu toàn cầu hoặc EUR 20M),
    CCPA (phạt $7,500/vi phạm), data breach notification obligations..."""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"privacy_analysis": response.content}
```

**Routing cho privacy_agent:**
```python
if any(kw in question_lower for kw in ["data", "privacy", "gdpr", "rò rỉ", "dữ liệu"]):
    tasks.append(Send("privacy_agent", state))
```

**Bug phát hiện trong skeleton:** `check_routing` được định nghĩa là node nhưng trả về `list[Send]` — LangGraph 0.4.x không cho node làm vậy.

**Fix:** Dùng `check_routing` làm **routing function** của `add_conditional_edges` thay vì node:
```python
# Sai (skeleton gốc):
graph.add_node("check_routing", check_routing)
graph.add_conditional_edges("check_routing", lambda x: x)

# Đúng (sau fix):
graph.add_conditional_edges("law_agent", check_routing)
# check_routing() → list[Send] là đúng signature của routing function
```

---

## 4. Bài Tập Nâng Cao

### Exercise 3.1 — Tool tra cứu án lệ

Thêm `search_case_law` vào Stage 3, giúp agent tra cứu các án lệ quan trọng:

| Keyword | Án lệ |
|---|---|
| breach | Hadley v. Baxendale (1854) — Consequential damages |
| negligence | Donoghue v. Stevenson (1932) — Duty of care |
| contract | Carlill v. Carbolic Smoke Ball (1893) — Unilateral contract |

### Exercise 5.2 — Fault Tolerance

**Thử nghiệm:** Tắt Tax Agent trong khi hệ thống đang chạy.

**Kết quả:** Hệ thống **không crash**. Law Agent bắt exception và trả về:
```
[Tax analysis unavailable: Connection refused]
```
Compliance Agent vẫn chạy bình thường → đây là **graceful degradation**.

**Tại sao được?** Vì `call_tax` trong `law_agent/graph.py` có `try/except`:
```python
except Exception as exc:
    return {"tax_result": f"[Tax analysis unavailable: {exc}]"}
```

### Exercise 5.3 — Modify Agent Behavior

Sửa system prompt của Tax Agent để trả lời ngắn gọn hơn:
```python
# Thêm vào cuối TAX_SYSTEM_PROMPT:
"Keep your response concise — use bullet points, limit to 300 words."
```
Sau khi restart, Tax Agent output có cấu trúc và ngắn gọn hơn rõ rệt.

---

## 5. Trace Request Flow — Stage 5

### Sequence Diagram

```
User → Customer Agent → Registry → Law Agent → Tax Agent    (song song)
                                             → Compliance Agent
                                  Law Agent (aggregate) → Customer Agent → User
```

### Dữ Liệu Thực Tế

**trace_id:** `965fd8e5-93eb-4736-a307-f9a7e32a6ee4`

| Bước | Sự kiện | Thời gian |
|---|---|---|
| 1 | User gửi câu hỏi đến Customer Agent | 11:05:34 |
| 2 | Customer discover Law Agent qua Registry | 11:05:34 |
| 3 | Customer → Law Agent (depth=1) qua A2A | 11:05:34 |
| 4 | Law Agent: LLM call `analyze_law` | 11:05:39 |
| 5 | Law Agent: LLM call `check_routing` | 11:05:48 |
| 6 | Routing: `needs_tax=True, needs_compliance=True` | 11:05:49 |
| 7 | Law Agent discover Tax + Compliance qua Registry | 11:05:50–51 |
| 8 | **PARALLEL:** Law → Tax Agent (depth=2) | 11:05:52 |
| 8 | **PARALLEL:** Law → Compliance Agent (depth=2) | 11:05:52 |
| 9 | Tax Agent phản hồi (1732 chars) | 11:06:06 |
| 10 | Compliance Agent phản hồi | 11:06:07 |
| 11 | Law Agent: LLM call `aggregate` | 11:06:07 |
| 12 | Law Agent → Customer Agent → User | ~11:06:20 |

**Tổng:** 3 delegation hops, 5–6 LLM calls, ~46 giây end-to-end.

---

## 6. Câu Hỏi Ôn Tập

**1. Khi nào dùng single agent, khi nào dùng multi-agent?**

Single agent phù hợp khi bài toán đơn giản, một domain, không cần song song. Multi-agent cần thiết khi bài toán đòi hỏi nhiều chuyên môn khác nhau, cần xử lý song song, hoặc mỗi phần có logic độc lập.

**2. Ưu điểm A2A so với REST/gRPC thông thường?**

A2A chuẩn hóa: Agent Card (metadata), Message format, Task lifecycle, trace propagation. Quan trọng nhất là **dynamic discovery** qua Registry — agents tìm nhau qua tên task, không hardcode URL.

**3. Làm thế nào prevent infinite delegation loops?**

`MAX_DELEGATION_DEPTH = 3` trong `law_agent/graph.py`. Mỗi A2A message mang `delegation_depth` trong metadata, agent kiểm tra trước khi delegate tiếp.

**4. Tại sao cần Registry?**

Nếu hardcode URL thì khi agent đổi port/server phải sửa code ở mọi nơi. Registry cho phép **loose coupling** — agent A chỉ cần biết "tìm agent xử lý task X", không cần biết agent đó ở đâu.
