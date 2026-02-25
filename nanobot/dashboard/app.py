"""FastAPI dashboard application for nanobot memory system audit."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Nanobot Memory Dashboard", version="1.0.0")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/dashboard/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# Config / workspace helpers
# ---------------------------------------------------------------------------

_config_cache: dict[str, Any] | None = None
_config_mtime: float = 0


def _load_config() -> dict[str, Any]:
    global _config_cache, _config_mtime
    config_path = Path.home() / ".nanobot" / "config.json"
    if not config_path.exists():
        raise RuntimeError(f"Config not found at {config_path}")
    mtime = config_path.stat().st_mtime
    if _config_cache is None or mtime != _config_mtime:
        _config_cache = json.loads(config_path.read_text())
        _config_mtime = mtime
    return _config_cache


def _workspace() -> Path:
    cfg = _load_config()
    ws = cfg.get("agents", {}).get("defaults", {}).get("workspace", "~/.nanobot/workspace")
    return Path(ws).expanduser()


def _agent_dir(name: str) -> Path:
    ws = _workspace()
    d = ws / "agents" / name
    if not d.exists():
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")
    return d


def _agent_names() -> list[str]:
    cfg = _load_config()
    profiles = cfg.get("agents", {}).get("profiles", {})
    return list(profiles.keys())


def _read_text(p: Path) -> str | None:
    if p.exists() and p.is_file():
        return p.read_text(errors="replace")
    return None


def _count_lines(p: Path) -> int:
    if not p.exists():
        return 0
    with open(p) as f:
        return sum(1 for _ in f)


def _safe_filename(key: str) -> str:
    return key.replace(":", "_").replace("/", "_")


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _parse_session(path: Path) -> tuple[dict, list[dict]]:
    metadata: dict = {}
    messages: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("_type") == "metadata":
                metadata = data
            else:
                messages.append(data)
    return metadata, messages


def _session_files(agent_dir: Path) -> list[Path]:
    sd = agent_dir / "sessions"
    if not sd.exists():
        return []
    return sorted(sd.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)


# ---------------------------------------------------------------------------
# LanceDB helpers (lazy init)
# ---------------------------------------------------------------------------

_lance_db = None
_lance_store = None


async def _get_lance_db():
    global _lance_db
    if _lance_db is not None:
        return _lance_db
    try:
        import lancedb
        cfg = _load_config()
        db_path = cfg.get("memoryGraph", {}).get("dbPath", "")
        if not db_path:
            return None
        db_path = os.path.expanduser(db_path)
        _lance_db = await lancedb.connect_async(db_path)
        return _lance_db
    except Exception:
        return None


async def _get_store():
    global _lance_store
    if _lance_store is not None:
        return _lance_store
    try:
        from agent_memory import LiteLLMEmbedding, MemoryGraphStore

        cfg = _load_config()
        mg = cfg.get("memoryGraph", {})
        db_path = os.path.expanduser(mg.get("dbPath", ""))
        if not db_path:
            return None
        emb_cfg = mg.get("embedding", {})
        api_key = emb_cfg.get("apiKey", "")
        model = emb_cfg.get("model", "openai/text-embedding-3-small")
        dims = emb_cfg.get("dimensions", 512)
        embedding = LiteLLMEmbedding(model=model, dimensions=dims, api_key=api_key)
        store = MemoryGraphStore(db_path=db_path, embedding=embedding)
        await store.initialize()
        _lance_store = store
        return store
    except Exception:
        return None


async def _lance_query(table_name: str, where: str | None = None,
                       limit: int = 50, offset: int = 0,
                       columns: list[str] | None = None) -> list[dict]:
    db = await _get_lance_db()
    if db is None:
        return []
    try:
        table = await db.open_table(table_name)
        q = table.query()
        if where:
            q = q.where(where)
        if columns:
            q = q.select(columns)
        q = q.offset(offset).limit(limit)
        rows = await q.to_list()
        # Convert pyarrow/lancedb rows to plain dicts
        result = []
        for r in rows:
            d = dict(r) if isinstance(r, dict) else {k: r[k] for k in r}
            # Strip vector field for JSON serialization
            d.pop("vector", None)
            # Convert numpy/arrow types
            for k, v in list(d.items()):
                if hasattr(v, "item"):
                    d[k] = v.item()
                elif hasattr(v, "as_py"):
                    d[k] = v.as_py()
            result.append(d)
        return result
    except Exception as e:
        return []


# ---------------------------------------------------------------------------
# Dashboard HTML
# ---------------------------------------------------------------------------

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(500, "Dashboard UI not found")
    return HTMLResponse(index.read_text())


# ---------------------------------------------------------------------------
# Agent endpoints
# ---------------------------------------------------------------------------

@app.get("/api/agents")
async def list_agents():
    cfg = _load_config()
    profiles = cfg.get("agents", {}).get("profiles", {})
    ws = _workspace()
    agents = []
    for name, profile in profiles.items():
        adir = ws / "agents" / name
        # Session stats
        sessions = _session_files(adir)
        total_msgs = 0
        for sp in sessions:
            try:
                _, msgs = _parse_session(sp)
                total_msgs += len(msgs)
            except Exception:
                pass

        # Memory stats
        mem_file = adir / "memory" / "MEMORY.md"
        mem_text = _read_text(mem_file) or ""
        mem_chars = len(mem_text)

        # History
        hist_dir = adir / "memory" / "history"
        hist_count = len(list(hist_dir.glob("*.md"))) if hist_dir.exists() else 0

        agents.append({
            "name": name,
            "model": profile.get("model"),
            "systemIdentity": (profile.get("systemIdentity") or "")[:80],
            "sessionCount": len(sessions),
            "messageCount": total_msgs,
            "memoryChars": mem_chars,
            "memoryTokens": _estimate_tokens(mem_text),
            "historyFileCount": hist_count,
            "discordChannels": profile.get("discordChannels", []),
        })
    return {"agents": agents}


@app.get("/api/agents/{name}/memory")
async def agent_memory(name: str):
    adir = _agent_dir(name)
    mem_file = adir / "memory" / "MEMORY.md"
    content = _read_text(mem_file) or ""
    chars = len(content)
    tokens = _estimate_tokens(content)
    budget_char_limit = 16000
    budget_token_limit = 4000
    return {
        "content": content,
        "chars": chars,
        "tokens": tokens,
        "budgetCharPercent": round(chars / budget_char_limit * 100, 1),
        "budgetTokenPercent": round(tokens / budget_token_limit * 100, 1),
    }


@app.get("/api/agents/{name}/history")
async def agent_history(name: str):
    adir = _agent_dir(name)
    hist_dir = adir / "memory" / "history"
    if not hist_dir.exists():
        return {"files": []}
    files = []
    for f in sorted(hist_dir.glob("*.md"), reverse=True):
        files.append({
            "date": f.stem,
            "size": f.stat().st_size,
            "modified": datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).isoformat(),
        })
    return {"files": files}


@app.get("/api/agents/{name}/history/{date}")
async def agent_history_date(name: str, date: str):
    adir = _agent_dir(name)
    f = adir / "memory" / "history" / f"{date}.md"
    if not f.exists():
        raise HTTPException(404, f"History file for {date} not found")
    return {"date": date, "content": f.read_text(errors="replace")}


@app.get("/api/agents/{name}/identity")
async def agent_identity(name: str):
    adir = _agent_dir(name)
    ws = _workspace()
    return {
        "identity": _read_text(adir / "IDENTITY.md"),
        "soul": _read_text(ws / "SOUL.md") or _read_text(adir / "SOUL.md"),
        "user": _read_text(ws / "USER.md") or _read_text(adir / "USER.md"),
    }


@app.get("/api/agents/{name}/sessions")
async def agent_sessions(name: str):
    adir = _agent_dir(name)
    sessions = []
    for sp in _session_files(adir):
        try:
            meta, msgs = _parse_session(sp)
            sessions.append({
                "key": meta.get("key", sp.stem),
                "file": sp.name,
                "messageCount": len(msgs),
                "lastConsolidated": meta.get("last_consolidated", 0),
                "createdAt": meta.get("created_at"),
                "updatedAt": meta.get("updated_at"),
                "size": sp.stat().st_size,
            })
        except Exception:
            sessions.append({
                "key": sp.stem,
                "file": sp.name,
                "messageCount": 0,
                "error": "Failed to parse",
            })
    return {"sessions": sessions}


def _resolve_session_path(name: str, key: str) -> Path:
    """Resolve a session key to its JSONL file path, or raise 404."""
    adir = _agent_dir(name)
    sd = adir / "sessions"
    candidates = [
        sd / f"{key}.jsonl",
        sd / f"{_safe_filename(key)}.jsonl",
        sd / key,
    ]
    for c in candidates:
        if c.exists():
            return c
    # Try matching by key in metadata
    for sp in _session_files(adir):
        meta, _ = _parse_session(sp)
        if meta.get("key") == key:
            return sp
    raise HTTPException(404, f"Session '{key}' not found")


@app.get("/api/agents/{name}/sessions/{key:path}")
async def agent_session_detail(name: str, key: str):
    # Intercept context log requests (key ends with /context)
    if key.endswith("/context"):
        actual_key = key[:-len("/context")]
        return await agent_session_context(name, actual_key)

    path = _resolve_session_path(name, key)
    meta, msgs = _parse_session(path)
    return {
        "metadata": meta,
        "messages": msgs,
        "messageCount": len(msgs),
    }


async def agent_session_context(name: str, key: str):
    """Return the per-turn context log for a session."""
    path = _resolve_session_path(name, key)
    context_path = path.with_suffix(".context.jsonl")
    if not context_path.exists():
        return {"entries": [], "count": 0}

    entries: list[dict] = []
    with open(context_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    entries.sort(key=lambda e: e.get("turn_index", 0))
    return {"entries": entries, "count": len(entries)}


# ---------------------------------------------------------------------------
# Graph endpoints
# ---------------------------------------------------------------------------

@app.get("/api/graph/memories")
async def graph_memories(
    peer_key: str | None = Query(None),
    type: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    sort: str = Query("created_at_ms"),
    forgotten: int = Query(0),
):
    where_parts = [f"forgotten = {forgotten}"]
    if peer_key:
        where_parts.append(f"peer_key = '{peer_key}'")
    if type:
        where_parts.append(f"memory_type = '{type}'")
    where = " AND ".join(where_parts)

    cols = [
        "id", "content", "memory_type", "importance", "source",
        "source_session", "peer_key", "entities", "created_at_ms",
        "updated_at_ms", "access_count", "forgotten", "context_tag",
    ]
    rows = await _lance_query("memories", where=where, limit=limit, offset=offset, columns=cols)
    return {"memories": rows, "limit": limit, "offset": offset}


@app.get("/api/graph/memories/{memory_id}")
async def graph_memory_detail(memory_id: str):
    db = await _get_lance_db()
    if db is None:
        raise HTTPException(503, "Graph database not available")
    try:
        table = await db.open_table("memories")
        rows = await table.query().where(f"id = '{memory_id}'").limit(1).to_list()
        if not rows:
            raise HTTPException(404, "Memory not found")
        d = dict(rows[0]) if isinstance(rows[0], dict) else {k: rows[0][k] for k in rows[0]}
        d.pop("vector", None)
        for k, v in list(d.items()):
            if hasattr(v, "item"):
                d[k] = v.item()
            elif hasattr(v, "as_py"):
                d[k] = v.as_py()
        return {"memory": d}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


class SearchRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    peer_key: str | None = None
    max_results: int = 10


@app.post("/api/graph/search")
async def graph_search(req: SearchRequest):
    store = await _get_store()
    if store is None:
        raise HTTPException(503, "Memory store not available (agent-memory not installed or not configured)")

    try:
        if req.mode == "vector":
            results = await store.vector_search(
                query=req.query, max_results=req.max_results,
                peer_key=req.peer_key,
            )
        elif req.mode == "keyword":
            results = await store.keyword_search(
                query=req.query, max_results=req.max_results,
                peer_key=req.peer_key,
            )
        else:
            results = await store.hybrid_search(
                query=req.query, max_results=req.max_results,
                peer_key=req.peer_key,
            )
        # Clean results for JSON
        clean = []
        for r in results:
            d = dict(r)
            d.pop("vector", None)
            for k, v in list(d.items()):
                if hasattr(v, "item"):
                    d[k] = v.item()
                elif hasattr(v, "as_py"):
                    d[k] = v.as_py()
            clean.append(d)
        return {"results": clean, "mode": req.mode, "query": req.query}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/graph/neighbors/{memory_id}")
async def graph_neighbors(memory_id: str, depth: int = Query(1, ge=1, le=3)):
    db = await _get_lance_db()
    if db is None:
        raise HTTPException(503, "Graph database not available")

    try:
        assoc_table = await db.open_table("associations")
        mem_table = await db.open_table("memories")

        visited_ids: set[str] = {memory_id}
        edges: list[dict] = []
        frontier = [memory_id]

        for _ in range(depth):
            if not frontier:
                break
            next_frontier = []
            for mid in frontier:
                # Get edges where this memory is source or target
                for direction, field in [("outgoing", "source_id"), ("incoming", "target_id")]:
                    rows = await assoc_table.query().where(f"{field} = '{mid}'").limit(100).to_list()
                    for r in rows:
                        d = dict(r) if isinstance(r, dict) else {k: r[k] for k in r}
                        for k, v in list(d.items()):
                            if hasattr(v, "item"):
                                d[k] = v.item()
                            elif hasattr(v, "as_py"):
                                d[k] = v.as_py()
                        edges.append(d)
                        other = d.get("target_id" if field == "source_id" else "source_id")
                        if other and other not in visited_ids:
                            visited_ids.add(other)
                            next_frontier.append(other)
            frontier = next_frontier

        # Fetch memory content for all visited nodes
        nodes = []
        for nid in visited_ids:
            rows = await mem_table.query().where(f"id = '{nid}'").select(
                ["id", "content", "memory_type", "importance", "peer_key", "entities"]
            ).limit(1).to_list()
            if rows:
                d = dict(rows[0]) if isinstance(rows[0], dict) else {k: rows[0][k] for k in rows[0]}
                d.pop("vector", None)
                for k, v in list(d.items()):
                    if hasattr(v, "item"):
                        d[k] = v.item()
                    elif hasattr(v, "as_py"):
                        d[k] = v.as_py()
                nodes.append(d)

        return {"nodes": nodes, "edges": edges, "rootId": memory_id}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/graph/stats")
async def graph_stats():
    db = await _get_lance_db()
    if db is None:
        return {"available": False}

    try:
        mem_table = await db.open_table("memories")
        assoc_table = await db.open_table("associations")

        # Get all memories for stats computation
        all_mems = await mem_table.query().select(
            ["id", "memory_type", "importance", "peer_key", "forgotten", "created_at_ms", "access_count"]
        ).limit(10000).to_list()

        total = len(all_mems)
        forgotten_count = 0
        type_dist: dict[str, int] = {}
        peer_dist: dict[str, int] = {}
        importance_sum = 0.0
        creation_dates: dict[str, int] = {}

        for r in all_mems:
            d = dict(r) if isinstance(r, dict) else {k: r[k] for k in r}
            for k, v in list(d.items()):
                if hasattr(v, "item"):
                    d[k] = v.item()
                elif hasattr(v, "as_py"):
                    d[k] = v.as_py()

            if d.get("forgotten", 0):
                forgotten_count += 1

            mt = d.get("memory_type", "unknown")
            type_dist[mt] = type_dist.get(mt, 0) + 1

            pk = d.get("peer_key") or "none"
            peer_dist[pk] = peer_dist.get(pk, 0) + 1

            imp = d.get("importance", 0)
            if isinstance(imp, (int, float)):
                importance_sum += imp

            ts = d.get("created_at_ms", 0)
            if ts:
                date_str = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                creation_dates[date_str] = creation_dates.get(date_str, 0) + 1

        # Edge count
        all_edges = await assoc_table.query().select(["id"]).limit(10000).to_list()
        edge_count = len(all_edges)

        return {
            "available": True,
            "totalMemories": total,
            "totalEdges": edge_count,
            "forgottenCount": forgotten_count,
            "avgImportance": round(importance_sum / max(total, 1), 3),
            "typeDistribution": dict(sorted(type_dist.items(), key=lambda x: -x[1])),
            "peerKeyDistribution": dict(sorted(peer_dist.items(), key=lambda x: -x[1])),
            "creationTimeline": dict(sorted(creation_dates.items())),
        }
    except Exception as e:
        return {"available": False, "error": str(e)}
