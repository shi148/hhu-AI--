#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
河海水利与 AI 赋能 · 可选后端（零第三方依赖）

用法：
    python server.py            # http://127.0.0.1:8000
    python server.py 8765       # 指定端口

静态页面本身可离线打开；启动本服务后额外启用：
    POST /api/ask    知识库检索问答（可选接大模型）
    POST /api/vote   匿名投票
    POST /api/note   匿名留言
    POST /api/layer  模块三分层互动记录
    GET  /api/stats  聚合统计

可选大模型（OpenAI 兼容接口，Key 只留在服务端）：
    set LLM_BASE_URL=https://api.openai.com/v1
    set LLM_API_KEY=sk-...
    set LLM_MODEL=gpt-4o-mini
未配置时自动使用本地检索，不影响演示。
"""

import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict, deque
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "app.db"
KB_PATH = ROOT / "data" / "kb.json"

ALLOWED_SUFFIX = {".html", ".css", ".js", ".json", ".svg", ".png", ".jpg", ".jpeg", ".webp", ".ico", ".txt"}
MAX_BODY = 8 * 1024
RATE_LIMIT = 30          # 每个 IP 每分钟最多提问次数
RATE_WINDOW = 60
_hits = defaultdict(deque)

VOTE_CHOICES = ("ai", "joint", "human")
VOTE_LABELS = {"ai": "全部交给 AI", "joint": "人机共决", "human": "人说了算"}


# ---------------- database ----------------
def db():
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS votes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                choice TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS qa_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                entry_id TEXT,
                mode TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                layer TEXT NOT NULL,
                action TEXT,
                value TEXT,
                created_at TEXT NOT NULL
            );
            """
        )


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ---------------- knowledge base ----------------
def load_kb():
    with open(KB_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = []
    for i, e in enumerate(raw):
        out.append({
            "id": str(i + 1),
            "title": e.get("h") or e.get("title", ""),
            "keywords": e.get("k") or e.get("keywords", []),
            "text": e.get("t") or e.get("text", ""),
            "source": e.get("s") or e.get("source", ""),
        })
    return out


KB = load_kb()


def normalize(text):
    return re.sub(r"[^\u4e00-\u9fa5a-z0-9]", "", text.lower())


def bigrams(text):
    s = normalize(text)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def score(query, entry):
    q = normalize(query)
    if not q:
        return 0
    s = 0.0
    hits = 0
    for kw in entry["keywords"]:
        k = normalize(kw)
        if not k:
            continue
        if k in q:
            s += 8 + len(k) * 1.7
            hits += 1
        elif len(q) >= 2 and q in k:
            s += 5
            hits += 1
    title = normalize(entry["title"])
    text = normalize(entry["text"])
    for g in bigrams(query):
        if g in title:
            s += 3.2
        elif g in text:
            s += 0.8
    if hits >= 2:
        s += 5
    if title and title in q:
        s += 12
    return s


def retrieve(query, limit=3):
    ranked = sorted(KB, key=lambda e: score(query, e), reverse=True)
    top = [e for e in ranked[:limit] if score(query, e) >= 14]
    return top


SUGGESTIONS = [
    "水利工程到底有多强？",
    "大禹学院怎么招生？",
    "大一能转专业吗？",
    "录取规则是什么？",
    "港航专业学什么？",
]


# ---------------- optional llm ----------------
def ask_llm(question, entries):
    base = os.environ.get("LLM_BASE_URL", "").rstrip("/")
    key = os.environ.get("LLM_API_KEY", "")
    model = os.environ.get("LLM_MODEL", "")
    if not (base and key and model):
        return None

    context = "\n\n".join(f"【{e['title']}｜{e['source']}】{e['text']}" for e in entries)
    payload = {
        "model": model,
        "temperature": 0.2,
        "max_tokens": 320,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是河海大学学科与招生问答的讲解员，面向想了解河海的高中生和家长。"
                    "只依据给定资料回答，资料没有的内容就说「这个知识库还没有收录，"
                    "建议查河海大学官网或拨打 400-025-1915」。不要编造数字、政策或史实。"
                    "用简体中文，先给结论再补一句依据，控制在 100 字以内，语气克制。"
                ),
            },
            {"role": "user", "content": f"资料：\n{context}\n\n问题：{question}"},
        ],
    }
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"].strip()
    except Exception as exc:  # 网络或接口失败时静默回退到本地检索
        print("[llm] fallback:", exc)
        return None


# ---------------- handlers ----------------
class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    # --- helpers ---
    def send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            return None
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            return None

    def rate_limited(self):
        ip = self.client_address[0]
        bucket = _hits[ip]
        t = time.time()
        while bucket and t - bucket[0] > RATE_WINDOW:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT:
            return True
        bucket.append(t)
        return False

    # --- GET ---
    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/api/health":
            self.send_json({"ok": True, "kb": len(KB), "llm": bool(os.environ.get("LLM_API_KEY"))})
            return
        if path == "/api/stats":
            self.stats()
            return
        if path.startswith("/api/"):
            self.send_json({"error": "method_not_allowed"}, 405)
            return

        if path == "/":
            self.path = "/index.html"
            path = "/index.html"

        suffix = Path(path).suffix.lower()
        if suffix not in ALLOWED_SUFFIX or path.endswith("/"):
            self.send_json({"error": "not_found"}, 404)
            return

        super().do_GET()

    def stats(self):
        with db() as conn:
            rows = conn.execute("SELECT choice, COUNT(*) AS n FROM votes GROUP BY choice").fetchall()
            notes = conn.execute("SELECT COUNT(*) AS n FROM notes").fetchone()["n"]
            asks = conn.execute("SELECT COUNT(*) AS n FROM qa_log").fetchone()["n"]
            layer_rows = conn.execute("SELECT layer, COUNT(*) AS n FROM events GROUP BY layer").fetchall()
            recent = conn.execute("SELECT content FROM notes ORDER BY id DESC LIMIT 8").fetchall()
        counts = {c: 0 for c in VOTE_CHOICES}
        for r in rows:
            counts[r["choice"]] = r["n"]
        layers = {"l1": 0, "l2": 0, "l3": 0, "l4": 0}
        for r in layer_rows:
            layers[r["layer"]] = r["n"]
        self.send_json({
            "votes": counts,
            "votes_total": sum(counts.values()),
            "notes": notes,
            "recent_notes": [r["content"] for r in recent],
            "questions": asks,
            "layers": layers,
            "layers_total": sum(layers.values()),
        })

    # --- POST ---
    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/ask":
            self.ask()
        elif path == "/api/vote":
            self.vote()
        elif path == "/api/note":
            self.note()
        elif path == "/api/layer":
            self.layer()
        else:
            self.send_json({"error": "not_found"}, 404)

    def layer(self):
        data = self.read_json() or {}
        name = str(data.get("layer", ""))
        if name not in ("l1", "l2", "l3", "l4"):
            self.send_json({"error": "invalid_layer"}, 400)
            return
        action = str(data.get("action", ""))[:20]
        value = str(data.get("value", ""))[:40]
        with db() as conn:
            conn.execute(
                "INSERT INTO events (layer, action, value, created_at) VALUES (?, ?, ?, ?)",
                (name, action, value, now()),
            )
        self.send_json({"ok": True})

    def ask(self):
        if self.rate_limited():
            self.send_json({"error": "rate_limited", "message": "问得有点快，缓一秒再试。"}, 429)
            return

        data = self.read_json() or {}
        question = str(data.get("q", "")).strip()[:200]
        if len(question) < 2:
            self.send_json({"error": "empty_question"}, 400)
            return

        entries = retrieve(question)
        if not entries:
            answer = (
                "这个问题本地知识库没有覆盖。可以试试问：水利工程 A+、大禹学院、"
                "转专业、录取规则、学费，或者「港航专业学什么」。"
            )
            entry_id, title, source, mode = None, "学科与招生", "", "local"
        else:
            top = entries[0]
            entry_id, title, source = top["id"], top["title"], top["source"]
            answer, mode = top["text"], "local"
            llm_answer = ask_llm(question, entries[:3])
            if llm_answer:
                answer, mode = llm_answer, "llm"

        with db() as conn:
            conn.execute(
                "INSERT INTO qa_log (question, entry_id, mode, created_at) VALUES (?, ?, ?, ?)",
                (question, entry_id, mode, now()),
            )

        self.send_json({
            "answer": answer,
            "title": title,
            "source": source,
            "mode": mode,
            "related": [{"id": e["id"], "title": e["title"], "source": e["source"]} for e in entries[1:]],
            "suggestions": SUGGESTIONS,
        })

    def vote(self):
        data = self.read_json() or {}
        choice = str(data.get("choice", ""))
        if choice not in VOTE_CHOICES:
            self.send_json({"error": "invalid_choice"}, 400)
            return
        with db() as conn:
            conn.execute("INSERT INTO votes (choice, created_at) VALUES (?, ?)", (choice, now()))
            rows = conn.execute("SELECT choice, COUNT(*) AS n FROM votes GROUP BY choice").fetchall()
        counts = {c: 0 for c in VOTE_CHOICES}
        for r in rows:
            counts[r["choice"]] = r["n"]
        self.send_json({"ok": True, "votes": counts, "votes_total": sum(counts.values())})

    def note(self):
        data = self.read_json() or {}
        content = re.sub(r"<[^>]*>", "", str(data.get("content", ""))).strip()
        if not (2 <= len(content) <= 80):
            self.send_json({"error": "invalid_length", "message": "留言长度 2—80 字。"}, 400)
            return
        with db() as conn:
            conn.execute("INSERT INTO notes (content, created_at) VALUES (?, ?)", (content, now()))
        self.send_json({"ok": True})

    def log_message(self, fmt, *args):
        if "/api/" in (self.path or ""):
            print("[api] %s %s" % (self.command, self.path))


def main():
    init_db()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print("河海水利与 AI 赋能 · 后端已启动")
    print("  页面  http://127.0.0.1:%d/" % port)
    print("  知识块 %d 条 · 大模型 %s" % (len(KB), "已配置" if os.environ.get("LLM_API_KEY") else "未配置（本地检索）"))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
