from __future__ import annotations
import os
os.environ.setdefault("GOOGLE_API_KEY", "dummy")

import asyncio
import json
import os
import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, List, Iterator, Tuple

import pandas as pd
import streamlit as st
from fpdf import FPDF

# -----------------------------
# Import your compiled LangGraph app
# -----------------------------
from bwa_backend import app


# -----------------------------
# Helpers
# -----------------------------
def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))
        if images_dir.exists() and images_dir.is_dir():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p))
    return buf.getvalue()


def images_zip(images_dir: Path) -> Optional[bytes]:
    if not images_dir.exists() or not images_dir.is_dir():
        return None
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in images_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p))
    return buf.getvalue()


def generate_pdf(md_text: str, blog_title: str) -> bytes:
    """Convert markdown text to a clean PDF using fpdf2."""
    pdf = FPDF()
    pdf.set_margins(left=20, top=20, right=20)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    def safe_text(t: str) -> str:
        """Strip non-latin1 chars fpdf2/Helvetica can't encode, truncate huge tokens."""
        t = t.encode("latin-1", errors="ignore").decode("latin-1")
        # Break any unbreakable token longer than 60 chars (URLs, code hashes)
        words = t.split(" ")
        out = []
        for w in words:
            if len(w) > 60:
                # insert soft breaks every 60 chars
                out.append(" ".join(w[i:i+60] for i in range(0, len(w), 60)))
            else:
                out.append(w)
        return " ".join(out)

    # Strip markdown image syntax and inline links
    clean = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md_text)
    clean = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", clean)
    # Strip bold/italic markers
    clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", clean)
    clean = re.sub(r"\*([^*]+)\*", r"\1", clean)
    # Strip blockquote markers
    clean = re.sub(r"^>\s*", "", clean, flags=re.MULTILINE)

    in_code_block = False
    for line in clean.splitlines():
        stripped = line.strip()

        # Toggle code block
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            try:
                pdf.set_font("Courier", "", 9)
                pdf.multi_cell(0, 5, safe_text(stripped) or " ")
            except Exception:
                pass
            continue

        if not stripped:
            pdf.ln(3)
        elif stripped.startswith("# "):
            pdf.set_font("Helvetica", "B", 18)
            pdf.multi_cell(0, 10, safe_text(stripped[2:]))
            pdf.ln(2)
        elif stripped.startswith("## "):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 14)
            pdf.multi_cell(0, 8, safe_text(stripped[3:]))
            pdf.ln(1)
        elif stripped.startswith("### "):
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(0, 7, safe_text(stripped[4:]))
        elif stripped.startswith(("- ", "* ", "+ ")):
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 6, "  * " + safe_text(stripped[2:]))
        elif re.match(r"^\d+\.\s", stripped):
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 6, safe_text(stripped))
        else:
            pdf.set_font("Helvetica", "", 10)
            try:
                pdf.multi_cell(0, 6, safe_text(stripped))
            except Exception:
                pass  # skip lines that still fail after sanitization

    return bytes(pdf.output())



# -----------------------------
# try_stream: async graph via thread bridge
# - uses stream_mode="updates" so each individual node (including each worker) emits separately
# - yields ("update", {node_name: node_output}) per step, then ("final", last_full_state)
# -----------------------------
def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    import queue as _queue
    import threading

    result_q = _queue.Queue()

    def thread_target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _collect():
            try:
                last_values = None
                # stream both updates (per-node) and values (full state) simultaneously
                async for step in graph_app.astream(inputs, stream_mode=["updates", "values"]):
                    kind_inner, data = step
                    if kind_inner == "updates":
                        result_q.put(("update", data))
                    elif kind_inner == "values":
                        last_values = data
                result_q.put(("final", last_values))
            except Exception as e:
                print(f"[try_stream] astream failed: {e}")
                try:
                    out = await graph_app.ainvoke(inputs)
                    result_q.put(("final", out))
                except Exception as e2:
                    result_q.put(("error", e2))
            finally:
                result_q.put(("done", None))

        loop.run_until_complete(_collect())
        loop.close()

    t = threading.Thread(target=thread_target, daemon=True)
    t.start()

    while True:
        kind, payload = result_q.get()
        if kind == "done":
            break
        elif kind == "error":
            raise payload
        else:
            yield (kind, payload)


# -----------------------------
# Node inference from state diff (needed for "values" stream mode)
# In "values" mode each payload is full state, not {"node_name": {...}}
# so we infer the node by which keys changed vs previous state.
# -----------------------------
def _infer_node(changed_keys: List[str]) -> Optional[str]:
    if "mode" in changed_keys or "needs_research" in changed_keys:
        return "router"
    if "evidence" in changed_keys:
        return "research"
    if "plan" in changed_keys:
        return "orchestrator"
    if "sections" in changed_keys:
        return "worker"
    if "merged_md" in changed_keys and "rewrite_count" not in changed_keys:
        return "merge_content"
    if "critique" in changed_keys:
        return "critic"
    if "rewrite_count" in changed_keys:
        return "rewrite"
    if "image_specs" in changed_keys:
        return "decide_images"
    if "final" in changed_keys:
        return "generate_and_place_images"
    return None


def extract_latest_state(current_state: Dict[str, Any], step_payload: Any) -> Dict[str, Any]:
    if isinstance(step_payload, dict):
        # "values" mode: payload IS the full state — just update directly
        current_state.update(step_payload)
    return current_state


# -----------------------------
# Markdown renderer that supports local images
# -----------------------------
_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_LINE_RE = re.compile(r"^\*(?P<cap>.+)\*$")


def _resolve_image_path(src: str) -> Path:
    src = src.strip().lstrip("./")
    return Path(src).resolve()


def render_markdown_with_local_images(md: str):
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        return

    parts: List[Tuple[str, str]] = []
    last = 0
    for m in matches:
        before = md[last : m.start()]
        if before:
            parts.append(("md", before))
        alt = (m.group("alt") or "").strip()
        src = (m.group("src") or "").strip()
        parts.append(("img", f"{alt}|||{src}"))
        last = m.end()

    tail = md[last:]
    if tail:
        parts.append(("md", tail))

    i = 0
    while i < len(parts):
        kind, payload = parts[i]

        if kind == "md":
            st.markdown(payload, unsafe_allow_html=False)
            i += 1
            continue

        alt, src = payload.split("|||", 1)

        caption = None
        if i + 1 < len(parts) and parts[i + 1][0] == "md":
            nxt = parts[i + 1][1].lstrip()
            if nxt.strip():
                first_line = nxt.splitlines()[0].strip()
                mcap = _CAPTION_LINE_RE.match(first_line)
                if mcap:
                    caption = mcap.group("cap").strip()
                    rest = "\n".join(nxt.splitlines()[1:])
                    parts[i + 1] = ("md", rest)

        if src.startswith("http://") or src.startswith("https://"):
            st.image(src, caption=caption or (alt or None), use_container_width=True)
        else:
            img_path = _resolve_image_path(src)
            if img_path.exists():
                st.image(str(img_path), caption=caption or (alt or None), use_container_width=True)
            else:
                st.warning(f"Image not found: `{src}` (looked for `{img_path}`)")

        i += 1


# -----------------------------
# Past blogs helpers
# -----------------------------
def list_past_blogs() -> List[Path]:
    cwd = Path(".")
    files = [p for p in cwd.glob("*.md") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def read_md_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def extract_title_from_md(md: str, fallback: str) -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            t = line[2:].strip()
            return t or fallback
    return fallback


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="LangGraph Blog Writer", layout="wide")

st.title("Blog Writing Agent")

with st.sidebar:
    st.header("Generate New Blog")
    topic = st.text_area("Topic", height=120)
    as_of = st.date_input("As-of date", value=date.today())
    run_btn = st.button("🚀 Generate Blog", type="primary")

    st.divider()
    st.subheader("Past blogs")

    past_files = list_past_blogs()
    if not past_files:
        st.caption("No saved blogs found (*.md in current folder).")
        selected_md_file = None
    else:
        options: List[str] = []
        file_by_label: Dict[str, Path] = {}
        for p in past_files[:50]:
            try:
                md_text = read_md_file(p)
                title = extract_title_from_md(md_text, p.stem)
            except Exception:
                title = p.stem
            label = f"{title}  ·  {p.name}"
            options.append(label)
            file_by_label[label] = p

        selected_label = st.radio(
            "Select a blog to load",
            options=options,
            index=0,
            label_visibility="collapsed",
        )
        selected_md_file = file_by_label.get(selected_label)

        if st.button("📂 Load selected blog"):
            if selected_md_file:
                md_text = read_md_file(selected_md_file)
                st.session_state["last_out"] = {
                    "plan": None,
                    "evidence": [],
                    "image_specs": [],
                    "final": md_text,
                }
                st.session_state["topic_prefill"] = extract_title_from_md(md_text, selected_md_file.stem)

if "topic_prefill" in st.session_state and isinstance(st.session_state["topic_prefill"], str):
    pass

if "last_out" not in st.session_state:
    st.session_state["last_out"] = None

tab_plan, tab_evidence, tab_preview, tab_images, tab_logs = st.tabs(
    ["🧩 Plan", "🔎 Evidence", "📝 Markdown Preview", "🖼️ Images", "🧾 Logs"]
)

logs: List[str] = []


def log(msg: str):
    logs.append(msg)


if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    inputs: Dict[str, Any] = {
        "topic": topic.strip(),
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": as_of.isoformat(),
        "recency_days": 7,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
        "critique": None,
        "rewrite_count": 0,
    }

    status = st.status("Running graph…", expanded=True)
    progress_area = st.empty()
    sections_area = st.empty()

    current_state: Dict[str, Any] = {}
    last_node = None
    sections_done = 0
    total_tasks = 0

    for kind, payload in try_stream(app, inputs):
        if kind == "update":
            # payload is {node_name: node_output_dict}
            for node_name, node_output in payload.items():
                if not isinstance(node_output, dict):
                    continue

                if node_name != last_node:
                    status.write(f"➡️ Node: `{node_name}`")
                    last_node = node_name

                # Merge node output into current_state
                current_state.update(node_output)

                # Track plan total tasks once orchestrator fires
                if node_name == "orchestrator" and "plan" in node_output:
                    plan_obj = node_output["plan"]
                    if isinstance(plan_obj, dict):
                        total_tasks = len(plan_obj.get("tasks", []))
                    elif hasattr(plan_obj, "tasks"):
                        total_tasks = len(plan_obj.tasks)

                # worker node may appear as "worker" or "reducer:worker" (subgraph prefix)
                is_worker = node_name == "worker" or node_name.endswith(":worker")
                if is_worker and "sections" in node_output:
                    new_sections = node_output["sections"]
                    sections_done += len(new_sections) if isinstance(new_sections, list) else 1
                    if total_tasks > 0:
                        sections_area.progress(
                            min(sections_done / total_tasks, 1.0),
                            text=f"✍️ Sections written: {sections_done}/{total_tasks}"
                        )

                summary = {
                    "mode": current_state.get("mode"),
                    "needs_research": current_state.get("needs_research"),
                    "evidence_count": len(current_state.get("evidence", []) or []),
                    "tasks": total_tasks or None,
                    "sections_done": sections_done,
                    "images": len(current_state.get("image_specs", []) or []),
                    "rewrite_count": current_state.get("rewrite_count", 0),
                }
                progress_area.json(summary)
                log(f"[update:{node_name}] {json.dumps(node_output, default=str)[:800]}")
                # DEBUG: log all node names seen (visible in Logs tab)
                if node_name not in (last_node or ""):
                    log(f"[node_seen] {node_name}")

        elif kind == "final":
            out = payload
            st.session_state["last_out"] = out
            sections_area.empty()
            status.update(label="✅ Done", state="complete", expanded=False)

            # Show critic score card immediately after generation
            critique = out.get("critique")
            if critique:
                crit = critique if isinstance(critique, dict) else (critique.model_dump() if hasattr(critique, "model_dump") else None)
                if crit:
                    passed = crit.get("passed", False)
                    score = crit.get("score", "?")
                    color = "🟢" if passed else "🔴"
                    st.info(
                        f"{color} **Quality Gate** — Score: **{score}/10** | "
                        f"Coherence: {crit.get('coherence')}/10 | "
                        f"Coverage: {crit.get('coverage')}/10 | "
                        f"Rewrites: {out.get('rewrite_count', 0)}\n\n"
                        f"**Critic feedback:** {crit.get('feedback', 'N/A')}"
                    )

            log("[final] received final state")

# -----------------------------
# Render last result (if any)
# -----------------------------
out = st.session_state.get("last_out")
if out:
    # --- Plan tab ---
    with tab_plan:
        st.subheader("Plan")

        # Critic score card (persistent)
        critique = out.get("critique")
        if critique:
            crit = critique if isinstance(critique, dict) else (critique.model_dump() if hasattr(critique, "model_dump") else None)
            if crit:
                passed = crit.get("passed", False)
                score = crit.get("score", "?")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Overall Score", f"{score}/10", delta="Pass ✅" if passed else "Rewritten 🔁")
                c2.metric("Coherence", f"{crit.get('coherence', '?')}/10")
                c3.metric("Coverage", f"{crit.get('coverage', '?')}/10")
                c4.metric("Rewrites", out.get("rewrite_count", 0))
                if not passed:
                    st.caption(f"Critic feedback: {crit.get('feedback', '')}")
                st.divider()
        plan_obj = out.get("plan")
        if not plan_obj:
            st.info("No plan found in output.")
        else:
            if hasattr(plan_obj, "model_dump"):
                plan_dict = plan_obj.model_dump()
            elif isinstance(plan_obj, dict):
                plan_dict = plan_obj
            else:
                plan_dict = json.loads(json.dumps(plan_obj, default=str))

            st.write("**Title:**", plan_dict.get("blog_title"))
            cols = st.columns(3)
            cols[0].write("**Audience:** " + str(plan_dict.get("audience")))
            cols[1].write("**Tone:** " + str(plan_dict.get("tone")))
            cols[2].write("**Blog kind:** " + str(plan_dict.get("blog_kind", "")))

            tasks = plan_dict.get("tasks", [])
            if tasks:
                df = pd.DataFrame(
                    [
                        {
                            "id": t.get("id"),
                            "title": t.get("title"),
                            "target_words": t.get("target_words"),
                            "requires_research": t.get("requires_research"),
                            "requires_citations": t.get("requires_citations"),
                            "requires_code": t.get("requires_code"),
                            "tags": ", ".join(t.get("tags") or []),
                        }
                        for t in tasks
                    ]
                ).sort_values("id")
                st.dataframe(df, use_container_width=True, hide_index=True)

                with st.expander("Task details"):
                    st.json(tasks)

    # --- Evidence tab ---
    with tab_evidence:
        st.subheader("Evidence")
        evidence = out.get("evidence") or []
        if not evidence:
            st.info("No evidence returned (maybe closed_book mode or no Tavily key/results).")
        else:
            rows = []
            for e in evidence:
                if hasattr(e, "model_dump"):
                    e = e.model_dump()
                rows.append(
                    {
                        "title": e.get("title"),
                        "published_at": e.get("published_at"),
                        "source": e.get("source"),
                        "url": e.get("url"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- Preview tab ---
    with tab_preview:
        st.subheader("Markdown Preview")
        final_md = out.get("final") or ""
        if not final_md:
            st.warning("No final markdown found.")
        else:
            render_markdown_with_local_images(final_md)

            plan_obj = out.get("plan")
            if hasattr(plan_obj, "blog_title"):
                blog_title = plan_obj.blog_title
            elif isinstance(plan_obj, dict):
                blog_title = plan_obj.get("blog_title", "blog")
            else:
                blog_title = extract_title_from_md(final_md, "blog")

            md_filename = f"{safe_slug(blog_title)}.md"
            st.download_button(
                "⬇️ Download Markdown",
                data=final_md.encode("utf-8"),
                file_name=md_filename,
                mime="text/markdown",
            )

            bundle = bundle_zip(final_md, md_filename, Path("images"))
            st.download_button(
                "📦 Download Bundle (MD + images)",
                data=bundle,
                file_name=f"{safe_slug(blog_title)}_bundle.zip",
                mime="application/zip",
            )

            try:
                pdf_bytes = generate_pdf(final_md, blog_title)
                st.download_button(
                    "📄 Download PDF",
                    data=pdf_bytes,
                    file_name=f"{safe_slug(blog_title)}.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.warning(f"PDF generation failed: {e}")

    # --- Images tab ---
    with tab_images:
        st.subheader("Images")
        specs = out.get("image_specs") or []
        images_dir = Path("images")

        if not specs and not images_dir.exists():
            st.info("No images generated for this blog.")
        else:
            if specs:
                st.write("**Image plan:**")
                st.json(specs)

            if images_dir.exists():
                files = [p for p in images_dir.iterdir() if p.is_file()]
                if not files:
                    st.warning("images/ exists but is empty.")
                else:
                    for p in sorted(files):
                        st.image(str(p), caption=p.name, use_container_width=True)

                z = images_zip(images_dir)
                if z:
                    st.download_button(
                        "⬇️ Download Images (zip)",
                        data=z,
                        file_name="images.zip",
                        mime="application/zip",
                    )

    # --- Logs tab ---
    with tab_logs:
        st.subheader("Logs")
        if "logs" not in st.session_state:
            st.session_state["logs"] = []
        if logs:
            st.session_state["logs"].extend(logs)

        st.text_area("Event log", value="\n\n".join(st.session_state["logs"][-80:]), height=520)
else:
    st.info("Enter a topic and click **Generate Blog**.")