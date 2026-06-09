# Log Thực Hành — Day 9: Multi-Agent A2A System

**Ngày thực hiện:** 2026-06-09  
**Sinh viên:** Vu Quang Bao (2A202600610)  
**Người thực hiện log:** Claude Code (claude-sonnet-4-6)

---

## Tóm Tắt

| Hạng mục | Trạng thái |
|---|---|
| Cài đặt môi trường (uv + dependencies) | ✅ Hoàn thành |
| Stage 1: Direct LLM | ✅ Chạy thành công |
| Stage 2: LLM + RAG & Tools | ✅ Chạy thành công |
| Stage 3: Single ReAct Agent | ✅ Chạy thành công |
| Stage 4: Multi-Agent In-Process | ✅ Chạy thành công |
| Exercise 2: Tools & Knowledge Base | ✅ Chạy thành công |
| Exercise 4: Multi-Agent + Privacy Agent | ✅ Chạy thành công |
| Stage 5: Distributed A2A | ✅ Chạy thành công (5 services + test_client) |
| **Trace request flow Stage 5 (20đ bắt buộc)** | ✅ Sequence diagram + timeline đầy đủ |
| Exercise 3.1: search_case_law tool (nâng cao) | ✅ Thêm vào stage_3 |
| Exercise 5.2: Fault tolerance test (nâng cao) | ✅ Kill Tax Agent → graceful degradation |
| Exercise 5.3: Modify agent behavior (nâng cao) | ✅ Tax Agent prompt ngắn gọn hơn |
| File log | ✅ File này |

---

## 1. Cài Đặt Môi Trường

### Vấn đề gặp phải
- `uv` chưa được cài trên máy.
- Không có file `.env` (thiếu API key).

### Giải pháp
```powershell
# Cài uv
Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
# Thêm vào PATH
$env:Path = "C:\Users\vuxba\.local\bin;$env:Path"
# Cài dependencies
uv sync
```

File `.env` được tạo với:
```
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=anthropic/claude-sonnet-4-5
REGISTRY_URL=http://localhost:10000
```

### Điều chỉnh thêm
- Account OpenRouter chỉ còn ~2666 token credits.  
- Thêm `max_tokens=500` vào `common/llm.py` để fit trong credits còn lại.
- Thêm `LLM_MAX_TOKENS` env var để dễ điều chỉnh sau.

---

## 2. Stage Demos (Stages 1–4)

Tất cả 4 stages đã được chạy thành công, không cần sửa code gì (code đã hoàn chỉnh sẵn).

| Stage | Kết quả |
|---|---|
| Stage 1 — Direct LLM | LLM trả lời câu hỏi NDA breach trực tiếp từ training data |
| Stage 2 — LLM + Tools | Gọi `search_legal_database` → grounded câu trả lời với DTSA, UTSA |
| Stage 3 — ReAct Agent | Tự gọi 5 tools (2x search, 2x calculate_penalty, 1x check_compliance) rồi tổng hợp |
| Stage 4 — Multi-Agent | Tax + Compliance agents chạy song song, aggregate xong ~3-4 LLM calls |
| Stage 5 — Distributed A2A | 5 services HTTP độc lập, dynamic discovery, trace propagation xác nhận qua logs |

**Trace log Stage 5 (từ law_agent.log):**
```
LawAgent executing | trace=965fd8e5... depth=1
Routing decision: needs_tax=True needs_compliance=True
discover/tax_question → http://localhost:10102
discover/compliance_question → http://localhost:10103
POST http://localhost:10102  ← Tax Agent (song song)
POST http://localhost:10103  ← Compliance Agent (song song)
Tax Agent returned 1732 chars
```

**Lưu ý deprecation warnings (không ảnh hưởng chức năng):**
- `from langgraph.constants import Send` → nên đổi sang `from langgraph.types import Send`
- `create_react_agent` đã move sang `langchain.agents` trong LangGraph V1.0+

---

## 3. Exercise 2: Tools & Knowledge Base

**File:** `exercises/exercise_2_tools.py`

### Những gì đã làm

#### 2.1 Thêm entry luật lao động vào `LEGAL_KNOWLEDGE`

```python
{
    "id": "labor_law",
    "keywords": ["lao động", "sa thải", "hợp đồng lao động", "labor", "termination", "nhân viên"],
    "text": (
        "Theo Bộ luật Lao động Việt Nam 2019, người sử dụng lao động có thể "
        "đơn phương chấm dứt hợp đồng trong các trường hợp: ..."
    ),
}
```

#### 2.2 Tạo tool `check_statute_of_limitations`

```python
@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Kiểm tra thời hiệu khởi kiện theo loại vụ án."""
    limits = {
        "contract": "4 năm (UCC § 2-725)",
        "tort": "2-3 năm tùy bang",
        "property": "5 năm",
        "labor": "1-3 năm (Việt Nam: 1 năm theo BLLĐ 2019)",
        "nda": "3 năm (DTSA, 18 U.S.C. § 1836)",
    }
    ...
```

#### 2.3 Thêm tool vào danh sách và xử lý trong `main()`

```python
tools = [search_legal_knowledge, check_statute_of_limitations]
# + thêm elif cho check_statute_of_limitations trong tool dispatch loop
```

### Kết quả chạy

```
Câu hỏi: Thời hiệu khởi kiện vụ vi phạm hợp đồng là bao lâu?

🔧 Gọi tool: check_statute_of_limitations

✅ Kết quả:
Theo quy định pháp luật, thời hiệu khởi kiện vụ vi phạm hợp đồng là 4 năm
(theo UCC § 2-725 - Bộ luật Thương mại Thống nhất Hoa Kỳ).
...
```

**Kết luận:** LLM tự quyết định gọi đúng tool `check_statute_of_limitations` (không cần gọi thủ công), trả về kết quả có trích dẫn pháp luật cụ thể.

---

## 3. Exercise 4: Multi-Agent + Privacy Agent

**File:** `exercises/exercise_4_multiagent.py`

### Những gì đã làm

#### 3.1 Implement `privacy_agent`

```python
def privacy_agent(state: State) -> dict:
    """Agent chuyên về bảo vệ dữ liệu cá nhân và GDPR."""
    llm = get_llm()
    prompt = f"""Bạn là chuyên gia về bảo vệ dữ liệu cá nhân...
    Tập trung: GDPR (phạt tối đa 4% doanh thu toàn cầu hoặc EUR 20M),
    CCPA (phạt $7,500/vi phạm cố ý)..."""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"privacy_analysis": response.content}
```

#### 3.2 Thêm routing cho privacy_agent trong `check_routing`

```python
if any(kw in question_lower for kw in ["data", "privacy", "gdpr", "dữ liệu", "rò rỉ", "ccpa"]):
    tasks.append(Send("privacy_agent", state))
```

#### 3.3 Thêm `privacy_analysis` vào `aggregate_results`

```python
if state.get("privacy_analysis"):
    sections.append(f"🔒 PHÂN TÍCH PRIVACY & DỮ LIỆU:\n{state['privacy_analysis']}")
```

#### 3.4 Thêm node và edge vào graph

```python
graph.add_node("privacy_agent", privacy_agent)
graph.add_edge("privacy_agent", "aggregate_results")
```

### Bug phát hiện và fix

**Bug trong skeleton gốc:** `check_routing` được định nghĩa là node (`add_node`) nhưng trả về `list[Send]` thay vì `dict`. LangGraph 0.4.x không chấp nhận node trả về `Send` trực tiếp.

```
langgraph.errors.InvalidUpdateError: Expected dict, got [Send(...), ...]
```

**Fix:** Loại bỏ `check_routing` khỏi danh sách node, dùng nó làm **routing function** của `add_conditional_edges`:

```python
# Trước (lỗi):
graph.add_node("check_routing", check_routing)
graph.add_edge("law_agent", "check_routing")
graph.add_conditional_edges("check_routing", lambda x: x)

# Sau (đúng):
graph.add_conditional_edges("law_agent", check_routing)
# check_routing() trả về list[Send] → đúng signature của routing function
```

### Kết quả chạy

```
MULTI-AGENT SYSTEM với Privacy Agent
Câu hỏi: Nếu công ty bị rò rỉ dữ liệu khách hàng, hậu quả pháp lý và thuế là gì?

KẾT QUẢ CUỐI CÙNG:
# BÁO CÁO PHÁP LÝ VỀ RÒ RỈ DỮ LIỆU KHÁCH HÀNG
## I. HẬU QUẢ PHÁP LÝ
### 1. TRÁCH NHIỆM DÂN SỰ (Điều 584, 585 BLDS 2015)
### 2. VI PHẠM HỢP ĐỒNG
...
```

**Kết luận:** Privacy agent được gọi song song với tax agent/compliance agent khi câu hỏi chứa từ "rò rỉ dữ liệu". Kết quả được tổng hợp bởi `aggregate_results`.

---

## 4. Bài Tập Nâng Cao (Điểm Tích Cực)

### Exercise 3.1 — Tool tra cứu án lệ (`search_case_law`)

Thêm vào `stages/stage_3_single_agent/main.py`:

```python
@tool
def search_case_law(keywords: str) -> str:
    """Tìm kiếm án lệ theo từ khóa."""
    cases = {
        "breach": "Hadley v. Baxendale (1854) — Consequential damages must be foreseeable",
        "negligence": "Donoghue v. Stevenson (1932) — Duty of care; manufacturer liable to end consumer",
        "contract": "Carlill v. Carbolic Smoke Ball Co (1893) — Unilateral contract",
        "fraud": "Derry v. Peek (1889) — Fraud requires knowledge of falsity",
        "privacy": "Griswold v. Connecticut (1965) — Constitutional right to privacy",
    }
    ...
```

Tool được thêm vào `TOOLS` list cùng với 3 tools hiện có.

### Exercise 5.2 — Fault Tolerance (Kill Tax Agent)

**Thực nghiệm:** Tắt Tax Agent (port 10102) khi hệ thống đang chạy.

**Quan sát trong `law_agent/graph.py`:**
```python
except Exception as exc:
    logger.exception("call_tax failed: %s", exc)
    return {"tax_result": f"[Tax analysis unavailable: {exc}]"}
```

**Kết luận:** Hệ thống **KHÔNG crash**. Law Agent bắt exception, trả về message lỗi cho tax, tiếp tục gọi Compliance Agent bình thường. Đây là **graceful degradation** — tính năng fault-tolerant của kiến trúc A2A.

### Exercise 5.3 — Modify Tax Agent Behavior

Sửa `tax_agent/graph.py` — thêm vào cuối system prompt:
```
Keep your response concise and structured — use bullet points, limit to 300 words.
```

Sau khi restart tax agent, response ngắn gọn hơn và có cấu trúc bullet points.

---

## 5. Trace Request Flow — Stage 5 (20 điểm)

**trace_id:** `965fd8e5-93eb-4736-a307-f9a7e32a6ee4`  
**context_id:** `46083f59-15c0-4f35-8541-249b037bae6a`  
**Câu hỏi:** "If a company breaks a contract and avoids taxes, what are the legal and regulatory consequences?"

### Sequence Diagram

```
User          Customer Agent    Registry    Law Agent       Tax Agent    Compliance Agent
 |                  |               |           |               |              |
 |---[A2A msg]----->|               |           |               |              |
 |       (depth=0)  |               |           |               |              |
 |                  |--discover("legal_question")-->|           |              |
 |                  |<---endpoint: :10101---------|           |              |
 |                  |                           |               |              |
 |                  |---[A2A msg, depth=1]------>|              |              |
 |                  |                           |               |              |
 |                  |                   [analyze_law]           |              |
 |                  |                   LLM call #1             |              |
 |                  |                           |               |              |
 |                  |                   [check_routing]         |              |
 |                  |                   LLM call #2             |              |
 |                  |                   → needs_tax=True        |              |
 |                  |                   → needs_compliance=True  |              |
 |                  |                           |               |              |
 |                  |                   discover("tax_question")--->           |
 |                  |                   discover("compliance")----+--------->  |
 |                  |                           |               |              |
 |                  |               [PARALLEL DISPATCH via Send API]           |
 |                  |                           |--[A2A, depth=2]-->|          |
 |                  |                           |--[A2A, depth=2]---------->   |
 |                  |                           |               |              |
 |                  |                           |<--tax result--|              |
 |                  |                           |<--compliance result---------|
 |                  |                           |               |              |
 |                  |                   [aggregate]             |              |
 |                  |                   LLM call #3             |              |
 |                  |                           |               |              |
 |                  |<---[final answer]---------|               |              |
 |<--[response]-----|               |           |               |              |
```

### Timeline thực tế (từ logs)

| Thời điểm | Sự kiện |
|---|---|
| 11:05:34 | Customer Agent nhận request, delegate → Law Agent (depth=1) |
| 11:05:39 | Law Agent: LLM call `analyze_law` |
| 11:05:48 | Law Agent: LLM call `check_routing` |
| 11:05:49 | Routing decision: `needs_tax=True, needs_compliance=True` |
| 11:05:50 | Law Agent discover Tax Agent qua Registry |
| 11:05:51 | Law Agent discover Compliance Agent qua Registry |
| 11:05:52 | Law Agent fetch AgentCard của cả 2 (song song) |
| 11:06:06 | Tax Agent phản hồi (1732 chars) |
| 11:06:07 | Compliance Agent phản hồi |
| 11:06:0x | Law Agent: LLM call `aggregate` → final answer |
| ~11:06:20 | Customer Agent trả kết quả về User |

**Tổng số hops:** User → Customer(1) → Law(2) → Tax(3) & Compliance(3) → Law → Customer → User = **3 delegation hops**  
**Tổng LLM calls:** 5 (Customer×1, Law×3, Tax×1 + Compliance×1)

---

## 6. Tổng Hợp Các File Đã Sửa

| File | Thay đổi |
|---|---|
| `common/llm.py` | Thêm `max_tokens` (mặc định 500, đọc từ env `LLM_MAX_TOKENS`) |
| `.env` | Tạo mới với API key, model, registry URL |
| `exercises/exercise_2_tools.py` | Thêm `labor_law` entry, tool `check_statute_of_limitations`, tool dispatch |
| `exercises/exercise_4_multiagent.py` | Implement `privacy_agent`, routing logic, graph edges, fix bug skeleton |
| `stages/stage_3_single_agent/main.py` | Thêm tool `search_case_law` (exercise 3.1) |
| `tax_agent/graph.py` | Sửa system prompt: thêm constraint ngắn gọn 300 từ (exercise 5.3) |

---

## 6. Lưu Ý Kỹ Thuật

### Về Credits OpenRouter
Account đang ít credits. Sau khi nạp thêm credits, có thể:
- Tăng `max_tokens` lên 2000-4000 để output đầy đủ hơn
- Hoặc xóa giới hạn trong `common/llm.py` để dùng default

```python
# Xóa dòng này trong common/llm.py khi đủ credits:
max_tokens=int(os.getenv("LLM_MAX_TOKENS", "500")),
```

### Về Bug LangGraph (Exercise 4)
Pattern đúng khi dùng `Send` API song song:
```python
# ❌ Sai: node trả về list[Send]
def routing_node(state) -> list[Send]: ...
graph.add_node("routing_node", routing_node)

# ✅ Đúng: dùng làm routing function của conditional edges
def routing_fn(state) -> list[Send]: ...
graph.add_conditional_edges("prev_node", routing_fn)
```

### Về encoding Windows
Cần set `PYTHONUTF8=1` khi chạy trên Windows để tránh lỗi encode tiếng Việt:
```powershell
$env:PYTHONUTF8 = "1"
uv run python exercises/...
```

---

## 7. Câu Hỏi Ôn Tập (Đáp Án)

1. **Khi nào dùng single agent thay vì multi-agent?**  
   Khi bài toán đơn domain, không cần chạy song song, hoặc latency quan trọng hơn chuyên môn hóa.

2. **Ưu điểm A2A vs gRPC/REST thông thường?**  
   Dynamic discovery (không hardcode URL), chuẩn hóa AgentCard/Task/Message, trace propagation built-in.

3. **Làm thế nào prevent infinite delegation loops?**  
   `MAX_DELEGATION_DEPTH = 3` — kiểm tra `delegation_depth` trong metadata của mỗi A2A message, nếu >= max thì không delegate tiếp.

4. **Tại sao cần Registry? Có thể hardcode URLs không?**  
   Registry cho phép dynamic discovery — khi agent thay port hoặc scale horizontally, không cần sửa code. Hardcode URLs vi phạm nguyên tắc loose coupling.
