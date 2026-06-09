"""
Lab Assignment — Day 09: Cải tiến Agent Day 08 với Supervisor-Workers Pattern
MSSV: 2A202600610 — Vũ Quang Bảo

============================================================
Day 08 (Original):  Linear Pipeline
    User Question
        → retrieve()       [Task 9: semantic + lexical + rerank + PageIndex]
        → reorder_for_llm()
        → format_context()
        → generate_with_citation()  [Groq LLM]
        → Final Answer

Day 09 (Improved):  Supervisor-Workers Pattern (LangGraph StateGraph)
    User Question
        → [SUPERVISOR] phân tích và khởi tạo routing strategy
              ↓
        → [WORKER 1 — QueryAnalyzer]
            Phân tích câu hỏi, xác định intent (legal/news/general),
            trích xuất keywords để tối ưu retrieval
              ↓
        → [WORKER 2 — RAGRetriever]
            Truy xuất documents bằng keyword matching + semantic scoring,
            rerank và reorder (tránh lost-in-the-middle)
              ↓
        → [WORKER 3 — AnswerGenerator]
            Sinh câu trả lời có citation dựa trên context,
            trả về "không thể xác minh" nếu evidence không đủ
              ↓
        → Final Answer with Citations

============================================================
Ưu điểm so với Day 08:
1. Mỗi Worker độc lập → dễ test, debug, swap component
2. Supervisor có thể route khác nhau (vd: skip RAG nếu câu hỏi quá chung)
3. Workers có thể mở rộng thêm (vd: Worker 4 QualityChecker, Worker 5 Translator)
4. State rõ ràng → dễ monitor từng bước xử lý
5. Dễ parallelise workers nếu independent (dùng LangGraph Send API)

Chạy:
    uv run python Lab_Assignment/main.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.llm import get_llm

load_dotenv()


# ============================================================
# KNOWLEDGE BASE (Drug Law + Vietnamese Legal — same domain as Day 08)
# ============================================================

DRUG_LAW_KNOWLEDGE = [
    {
        "id": "drug_law_2021_basic",
        "keywords": ["ma túy", "phòng chống", "luật", "2021", "cơ bản", "drug", "law"],
        "text": (
            "Luật Phòng chống ma túy 2021 (Luật số 73/2021/QH14) có hiệu lực từ 1/1/2022. "
            "Luật quy định về phòng ngừa, ngăn chặn, đấu tranh chống ma túy; cai nghiện ma túy; "
            "quản lý người sau cai nghiện; trách nhiệm của cá nhân, gia đình, cơ quan, tổ chức "
            "trong phòng, chống ma túy."
        ),
        "source": "Luật Phòng chống ma túy 2021",
        "type": "legal",
    },
    {
        "id": "drug_penalty_possession",
        "keywords": ["tàng trữ", "hình phạt", "phạt tù", "possession", "penalty", "xử phạt", "tội"],
        "text": (
            "Theo BLHS 2015 sửa đổi 2017, Điều 249 — Tội tàng trữ trái phép chất ma túy: "
            "Phạt tù từ 1-5 năm (dưới 1g heroin/cocaine hoặc dưới 5g các chất khác); "
            "5-10 năm (1-100g heroin hoặc 5-30g methamphetamine); "
            "10-15 năm (100-300g heroin); 15-20 năm hoặc tù chung thân (trên 300g heroin)."
        ),
        "source": "BLHS 2015, Điều 249",
        "type": "legal",
    },
    {
        "id": "drug_trafficking",
        "keywords": ["buôn bán", "mua bán", "trafficking", "vận chuyển", "tội mua bán"],
        "text": (
            "BLHS 2015, Điều 251 — Tội mua bán trái phép chất ma túy: "
            "Phạt tù 2-7 năm (trường hợp thông thường); 7-15 năm (có tổ chức, lợi dụng chức vụ); "
            "15-20 năm hoặc tù chung thân (trên 100g heroin); Tử hình (trên 300g heroin "
            "hoặc 1kg các chất khác, hoặc tái phạm nguy hiểm)."
        ),
        "source": "BLHS 2015, Điều 251",
        "type": "legal",
    },
    {
        "id": "drug_rehabilitation",
        "keywords": ["cai nghiện", "rehabilitation", "bắt buộc", "tự nguyện", "cơ sở", "điều trị"],
        "text": (
            "Luật Phòng chống ma túy 2021, Chương VI — Cai nghiện ma túy: "
            "Người nghiện ma túy có thể cai nghiện tự nguyện (tại nhà, cộng đồng, cơ sở cai nghiện) "
            "hoặc cai nghiện bắt buộc (do Tòa án quyết định, thời hạn 12-24 tháng). "
            "Gia đình có thể đăng ký cai nghiện tự nguyện cho thành viên nghiện ma túy."
        ),
        "source": "Luật Phòng chống ma túy 2021, Chương VI",
        "type": "legal",
    },
    {
        "id": "drug_precursors",
        "keywords": ["tiền chất", "precursor", "hóa chất", "sản xuất", "manufacturing"],
        "text": (
            "Luật Phòng chống ma túy 2021, Điều 24-28 — Kiểm soát tiền chất: "
            "Tiền chất ma túy là hóa chất không thể thiếu trong quá trình sản xuất, điều chế ma túy. "
            "Việc sản xuất, kinh doanh tiền chất phải được cấp phép. "
            "Vi phạm quy định về tiền chất bị xử phạt hành chính hoặc hình sự tùy mức độ."
        ),
        "source": "Luật Phòng chống ma túy 2021, Điều 24-28",
        "type": "legal",
    },
    {
        "id": "celebrity_drug_arrests",
        "keywords": ["nghệ sĩ", "ca sĩ", "diễn viên", "bắt", "arrest", "celebrity", "nổi tiếng"],
        "text": (
            "Một số vụ nghệ sĩ Việt bị bắt vì ma túy đáng chú ý: "
            "Nam ca sĩ T.L. bị bắt năm 2021 với tang vật methamphetamine; "
            "Diễn viên C.T.L. bị xử phạt hành chính năm 2022; "
            "Nhiều nghệ sĩ bị tạm đình chỉ hoạt động nghệ thuật theo Nghị quyết của Bộ VHTT&DL. "
            "Theo quy định, nghệ sĩ liên quan ma túy có thể bị cấm sóng, xóa hợp đồng quảng cáo."
        ),
        "source": "Tin tức tổng hợp 2021-2024",
        "type": "news",
    },
    {
        "id": "drug_prevention_community",
        "keywords": ["phòng ngừa", "cộng đồng", "prevention", "giáo dục", "tuyên truyền", "trường học"],
        "text": (
            "Luật Phòng chống ma túy 2021, Chương II — Phòng ngừa ma túy: "
            "Nhà trường có trách nhiệm tuyên truyền, giáo dục về tác hại ma túy trong học sinh. "
            "Ủy ban nhân dân các cấp tổ chức phong trào toàn dân phòng, chống ma túy. "
            "Người phát hiện hành vi vi phạm có quyền tố cáo với cơ quan có thẩm quyền."
        ),
        "source": "Luật Phòng chống ma túy 2021, Chương II",
        "type": "legal",
    },
]


# ============================================================
# STATE DEFINITION
# ============================================================

def _list_append(a: list, b: list) -> list:
    """Reducer: append lists together."""
    return a + b


class RAGState(TypedDict):
    question: str
    query_type: str                              # "legal" | "news" | "general"
    keywords: list[str]
    retrieved_chunks: list[dict]
    formatted_context: str
    final_answer: str
    sources: list[str]
    worker_log: Annotated[list[str], _list_append]


# ============================================================
# SUPERVISOR — Entry point, sets routing strategy
# ============================================================

async def supervisor_entry(state: RAGState) -> dict:
    """
    Supervisor: Nhận câu hỏi, khởi tạo state cho workers.
    Trong phiên bản nâng cao, supervisor có thể quyết định
    bỏ qua một số workers hoặc chạy song song.
    """
    print("\n[SUPERVISOR] Nhận câu hỏi và khởi tạo routing...")
    print(f"  Question: {state['question'][:80]}{'...' if len(state['question']) > 80 else ''}")
    return {
        "worker_log": [f"[Supervisor] Nhận câu hỏi: '{state['question'][:60]}...'"],
    }


# ============================================================
# WORKER 1 — QueryAnalyzer
# ============================================================

async def worker_query_analyzer(state: RAGState) -> dict:
    """
    Worker 1: Phân tích câu hỏi.
    - Xác định loại câu hỏi (legal/news/general)
    - Trích xuất keywords quan trọng để tối ưu retrieval
    - Cải tiến so với Day 08: Day 08 dùng raw query trực tiếp, không optimize
    """
    print("\n[WORKER 1 — QueryAnalyzer] Đang phân tích câu hỏi...")
    llm = get_llm()

    messages = [
        SystemMessage(content=(
            "Bạn là query analyzer cho hệ thống RAG pháp luật Việt Nam về ma túy. "
            "Phân tích câu hỏi và trả lời CHÍNH XÁC theo format sau (không thêm gì khác):\n"
            "TYPE: [legal|news|general]\n"
            "KEYWORDS: [từ khóa 1, từ khóa 2, từ khóa 3, ...]\n\n"
            "TYPE rules:\n"
            "- legal: câu hỏi về luật, điều khoản, hình phạt, quy định\n"
            "- news: câu hỏi về sự kiện, nghệ sĩ, tin tức cụ thể\n"
            "- general: câu hỏi tổng quan, giải thích, giáo dục\n\n"
            "KEYWORDS: trích 3-6 từ khóa quan trọng nhất từ câu hỏi (tiếng Việt và/hoặc Anh)"
        )),
        HumanMessage(content=state["question"]),
    ]

    response = await llm.ainvoke(messages)
    raw = response.content.strip()

    query_type = "general"
    keywords = []

    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("TYPE:"):
            t = line.replace("TYPE:", "").strip().lower()
            if t in ("legal", "news", "general"):
                query_type = t
        elif line.startswith("KEYWORDS:"):
            kw_str = line.replace("KEYWORDS:", "").strip()
            kw_str = kw_str.strip("[]")
            keywords = [k.strip() for k in kw_str.split(",") if k.strip()]

    print(f"  Type: {query_type}")
    print(f"  Keywords: {keywords}")

    return {
        "query_type": query_type,
        "keywords": keywords,
        "worker_log": [f"[Worker 1] Type={query_type}, Keywords={keywords}"],
    }


# ============================================================
# WORKER 2 — RAGRetriever
# ============================================================

def _score_chunk(chunk: dict, question: str, keywords: list[str]) -> float:
    """Tính relevance score cho một chunk."""
    text_lower = chunk["text"].lower()
    question_lower = question.lower()

    keyword_score = sum(1.0 for kw in chunk["keywords"] if kw in question_lower)
    kw_in_text = sum(0.5 for kw in keywords if kw.lower() in text_lower)
    source_bonus = 0.2 if chunk["type"] == "legal" else 0.0

    return keyword_score + kw_in_text + source_bonus


def _reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Reorder chunks để tránh 'lost in the middle' effect.
    Giống hàm reorder_for_llm() trong Day 08/task10_generation.py.
    Best chunks ở đầu và cuối, worst ở giữa.
    """
    if len(chunks) <= 2:
        return chunks
    front = chunks[0::2]
    back = chunks[1::2][::-1]
    return front + back


async def worker_rag_retriever(state: RAGState) -> dict:
    """
    Worker 2: Retrieval + Reranking.
    - Keyword matching với DRUG_LAW_KNOWLEDGE
    - Score và rank theo relevance
    - Reorder để tránh lost-in-the-middle (từ Day 08)
    - Cải tiến: Worker độc lập → dễ swap sang vectorstore thực (ChromaDB, etc.)
    """
    print("\n[WORKER 2 — RAGRetriever] Đang truy xuất documents...")

    question = state["question"]
    keywords = state.get("keywords", [])
    query_type = state.get("query_type", "general")

    scored = []
    for chunk in DRUG_LAW_KNOWLEDGE:
        if query_type == "news" and chunk["type"] == "legal":
            penalty = 0.3
        elif query_type == "legal" and chunk["type"] == "news":
            penalty = 0.3
        else:
            penalty = 0.0

        score = _score_chunk(chunk, question, keywords) - penalty
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [chunk for _, chunk in scored[:5]]

    if not top_chunks:
        top_chunks = DRUG_LAW_KNOWLEDGE[:2]

    reordered = _reorder_for_llm(top_chunks)

    context_parts = []
    sources = []
    for i, chunk in enumerate(reordered, 1):
        context_parts.append(
            f"[Document {i} | Source: {chunk['source']} | Type: {chunk['type']}]\n"
            f"{chunk['text']}"
        )
        sources.append(f"{chunk['source']} (type: {chunk['type']})")

    formatted_context = "\n\n---\n\n".join(context_parts)

    print(f"  Retrieved {len(reordered)} chunks (type filter: {query_type})")
    for i, (score, chunk) in enumerate(scored[:5], 1):
        print(f"  {i}. [{score:.2f}] {chunk['source'][:50]}")

    return {
        "retrieved_chunks": reordered,
        "formatted_context": formatted_context,
        "sources": sources,
        "worker_log": [f"[Worker 2] Retrieved {len(reordered)} chunks, top source: {sources[0] if sources else 'none'}"],
    }


# ============================================================
# WORKER 3 — AnswerGenerator
# ============================================================

async def worker_answer_generator(state: RAGState) -> dict:
    """
    Worker 3: Sinh câu trả lời có citation.
    - Giống generate_with_citation() trong Day 08/task10_generation.py
    - Cải tiến: Worker độc lập → dễ swap model (Groq → OpenRouter → local)
    - Cải tiến: Nhận formatted_context từ Worker 2 thay vì tự format
    """
    print("\n[WORKER 3 — AnswerGenerator] Đang sinh câu trả lời...")

    llm = get_llm()

    system_prompt = (
        "Trả lời câu hỏi bằng tiếng Việt một cách toàn diện.\n"
        "Với mỗi thông tin, PHẢI chèn citation ngay sau, ví dụ: [Luật Phòng chống ma túy 2021, Điều 3] "
        "hoặc [BLHS 2015, Điều 249].\n\n"
        "Nếu thông tin không có trong context, trả lời: "
        "'Tôi không thể xác minh thông tin này từ nguồn hiện có.'\n\n"
        "Quy tắc:\n"
        "- Chỉ dùng thông tin từ context được cung cấp\n"
        "- Mọi thông tin cụ thể PHẢI có citation\n"
        "- Nếu context không đủ, nói rõ ràng\n"
        "- Cấu trúc câu trả lời rõ ràng, có đoạn văn"
    )

    user_message = (
        f"Context:\n{state['formatted_context']}\n\n"
        f"---\n\n"
        f"Câu hỏi: {state['question']}"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    response = await llm.ainvoke(messages)
    answer = response.content

    print(f"  Answer generated ({len(answer)} chars)")

    return {
        "final_answer": answer,
        "worker_log": [f"[Worker 3] Answer generated ({len(answer)} chars)"],
    }


# ============================================================
# GRAPH CONSTRUCTION — Supervisor-Workers
# ============================================================

def build_supervisor_workers_graph():
    """
    Build LangGraph StateGraph với Supervisor-Workers pattern.

    Topology:
        supervisor_entry
              ↓
        worker_query_analyzer   [Worker 1]
              ↓
        worker_rag_retriever    [Worker 2]
              ↓
        worker_answer_generator [Worker 3]
              ↓
             END
    """
    graph = StateGraph(RAGState)

    graph.add_node("supervisor", supervisor_entry)
    graph.add_node("query_analyzer", worker_query_analyzer)
    graph.add_node("rag_retriever", worker_rag_retriever)
    graph.add_node("answer_generator", worker_answer_generator)

    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "query_analyzer")
    graph.add_edge("query_analyzer", "rag_retriever")
    graph.add_edge("rag_retriever", "answer_generator")
    graph.add_edge("answer_generator", END)

    return graph.compile()


# ============================================================
# MAIN
# ============================================================

TEST_QUESTIONS = [
    "Hình phạt cho tội tàng trữ trái phép chất ma túy theo pháp luật Việt Nam là gì?",
    "Nghệ sĩ nào đã bị bắt vì liên quan đến ma túy ở Việt Nam?",
    "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma túy 2021 như thế nào?",
]


async def run_question(graph, question: str):
    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)

    initial_state: RAGState = {
        "question": question,
        "query_type": "",
        "keywords": [],
        "retrieved_chunks": [],
        "formatted_context": "",
        "final_answer": "",
        "sources": [],
        "worker_log": [],
    }

    result = await graph.ainvoke(initial_state)

    print("\n" + "-" * 70)
    print("FINAL ANSWER:")
    print("-" * 70)
    print(result["final_answer"])

    print("\n[Sources used]")
    for i, src in enumerate(result["sources"], 1):
        print(f"  {i}. {src}")

    print("\n[Worker execution log]")
    for log_entry in result["worker_log"]:
        print(f"  {log_entry}")


async def main():
    print("=" * 70)
    print("LAB ASSIGNMENT — Day 09: Supervisor-Workers RAG Agent")
    print("Cải tiến từ Day 08 RAG Pipeline với LangGraph")
    print("=" * 70)
    print()
    print("[Architecture]")
    print("  Supervisor → Worker1(QueryAnalyzer) → Worker2(RAGRetriever) → Worker3(AnswerGenerator)")
    print()
    print("[Graph Visualization]")

    graph = build_supervisor_workers_graph()

    graph_png_path = os.path.join(os.path.dirname(__file__), "graph.png")
    try:
        png_data = graph.get_graph().draw_mermaid_png()
        with open(graph_png_path, "wb") as f:
            f.write(png_data)
        print(f"  Graph saved → {graph_png_path}")
    except Exception as exc:
        print(f"  PNG export unavailable ({exc})")
        print("  Mermaid source:")
        print(graph.get_graph().draw_mermaid())

    for question in TEST_QUESTIONS:
        await run_question(graph, question)

    print("\n" + "=" * 70)
    print("HOÀN THÀNH — So sánh với Day 08:")
    print("=" * 70)
    print("  Day 08 (linear):     retrieve() → reorder() → format() → generate()")
    print("  Day 09 (supervisor): Supervisor → W1(analyze) → W2(retrieve) → W3(generate)")
    print()
    print("  Cải tiến chính:")
    print("  1. Mỗi bước là Worker độc lập — dễ test, swap, monitor riêng")
    print("  2. Worker 1 optimize query trước → retrieval chính xác hơn")
    print("  3. State rõ ràng (TypedDict) → dễ debug từng bước")
    print("  4. Dễ thêm Worker mới (QualityChecker, Translator, etc.)")
    print("  5. Supervisor có thể route khác nhau tùy loại câu hỏi")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
