"""
app.py — Shop Go
────────────────────────────────────────────────────────────────────────────
Entry point and Gradio UI for the Shop Go price comparison agent.

Run:
    pip install gradio anthropic
    export ANTHROPIC_API_KEY=sk-...
    python app.py
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time

import gradio as gr

from orchestrator_agent import ShopGoOrchestrator, ShopGoResult

# ─────────────────────────────────────────────────────────────────────────────
# Shared state
# ─────────────────────────────────────────────────────────────────────────────

orchestrator = ShopGoOrchestrator()

_step_queue: queue.Queue[dict] = queue.Queue()
orchestrator.on_step_update(lambda s: _step_queue.put(s))

# ─────────────────────────────────────────────────────────────────────────────
# HTML renderers
# ─────────────────────────────────────────────────────────────────────────────

STORE_COLORS = {
    "Amazon":     ("#FF9900", "#131921"),
    "eBay":       ("#E53238", "#F5F5F5"),
    "AliExpress": ("#FF4747", "#fff"),
    "Walmart":    ("#0071CE", "#FFC220"),
    "Jumia":      ("#F68B1E", "#fff"),
    "Kilimall":   ("#e91e8c", "#fff"),
    "Masoko":     ("#009247", "#fff"),
    "Jiji":       ("#388E3C", "#fff"),
}


def _store_pill(store: str, logo: str) -> str:
    accent, _ = STORE_COLORS.get(store, ("#6366f1", "#fff"))
    return (
        f'<span style="background:{accent}22;color:{accent};border:1px solid {accent}44;'
        f'padding:2px 10px;border-radius:20px;font-size:11px;font-weight:700;'
        f'letter-spacing:.4px">{logo} {store}</span>'
    )


def _stars(rating: float) -> str:
    full  = int(rating)
    half  = 1 if rating - full >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + "⯨" * half + "☆" * empty


def render_product_card(item: dict, rank: int) -> str:
    badge    = item.get("badge", "")
    is_best  = rank == 0
    accent   = "#f59e0b" if is_best else "#6366f1"
    border   = f"2px solid {accent}" if is_best else "1px solid #1e293b"
    shadow   = f"0 4px 32px {accent}33" if is_best else "0 2px 12px #0005"

    currency_sym = {"USD": "$", "KES": "KSh", "NGN": "₦", "EGP": "£", "GBP": "£", "EUR": "€"}.get(
        item.get("currency", "USD"), "$"
    )
    score_bar = int(item.get("deal_score", 50))

    return f"""
<div style="
    background:#0f172a;border:{border};border-radius:16px;
    padding:18px;margin-bottom:14px;position:relative;
    box-shadow:{shadow};transition:transform .2s;font-family:'DM Sans',sans-serif;
" onmouseover="this.style.transform='translateY(-2px)'" onmouseout="this.style.transform='none'">

  {'<div style="position:absolute;top:-10px;left:20px;background:'+accent+';color:#0f172a;font-size:11px;font-weight:800;padding:3px 14px;border-radius:20px;letter-spacing:.5px">'+badge+'</div>' if badge else ''}

  <div style="display:flex;gap:16px;align-items:flex-start">
    <div style="flex-shrink:0">
      <img src="{item['image_url']}" alt="{item['title']}"
           style="width:88px;height:88px;object-fit:cover;border-radius:10px;border:1px solid #1e293b"/>
    </div>

    <div style="flex:1;min-width:0">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;flex-wrap:wrap">
        {_store_pill(item['store'], item.get('store_logo','🛒'))}
        {'<span style="background:#fef3c7;color:#92400e;padding:2px 8px;border-radius:12px;font-size:10px;font-weight:700">IN STOCK</span>' if item.get('in_stock') else '<span style="background:#fee2e2;color:#991b1b;padding:2px 8px;border-radius:12px;font-size:10px;font-weight:700">OUT OF STOCK</span>'}
      </div>

      <div style="font-size:14px;font-weight:600;color:#e2e8f0;margin-bottom:4px;
                  white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
        {item['title']}
      </div>

      <div style="font-size:12px;color:#64748b;margin-bottom:8px;
                  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden">
        {item['description']}
      </div>

      <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">
        <div>
          <span style="font-size:22px;font-weight:800;color:{accent}">${item['price']:.2f}</span>
          <span style="font-size:11px;color:#475569;margin-left:4px">
            {currency_sym}{item['original_price']:.0f} {item['currency']}
          </span>
        </div>
        <div style="font-size:12px;color:#fbbf24">{_stars(item['rating'])}
          <span style="color:#64748b;margin-left:4px">{item['rating']} ({item['review_count']:,})</span>
        </div>
        <div style="font-size:12px;color:#34d399">⚡ {item['delivery_days']}d delivery</div>
      </div>
    </div>

    <div style="text-align:center;flex-shrink:0">
      <div style="font-size:10px;color:#475569;margin-bottom:4px;letter-spacing:.5px">DEAL SCORE</div>
      <div style="font-size:26px;font-weight:900;color:{accent}">{score_bar}</div>
      <div style="width:48px;height:4px;background:#1e293b;border-radius:4px;margin:4px auto">
        <div style="width:{score_bar}%;height:100%;background:{accent};border-radius:4px"></div>
      </div>
      <a href="{item['url']}" target="_blank"
         style="display:inline-block;margin-top:10px;background:{accent};color:#0f172a;
                padding:6px 14px;border-radius:8px;font-size:11px;font-weight:800;
                text-decoration:none;letter-spacing:.3px">
        View Deal →
      </a>
    </div>
  </div>
</div>
"""


def render_results(result: ShopGoResult) -> str:
    if not result.ranked:
        return '<div style="color:#64748b;padding:40px;text-align:center">No results found. Try a different search.</div>'

    cards = "".join(render_product_card(item, i) for i, item in enumerate(result.ranked[:12]))

    # Store breakdown bar
    stores_seen = {}
    for r in result.ranked:
        stores_seen[r["store"]] = stores_seen.get(r["store"], 0) + 1

    store_pills = " ".join(
        f'<span style="background:#1e293b;color:#94a3b8;padding:3px 12px;border-radius:20px;font-size:11px">'
        f'{r["store_logo"]} {r["store"]} <b style="color:#e2e8f0">{c}</b></span>'
        for r, c in [
            (next(x for x in result.ranked if x["store"] == s), n)
            for s, n in stores_seen.items()
        ]
    )

    return f"""
<div style="font-family:'DM Sans',sans-serif">
  <div style="background:linear-gradient(135deg,#1e293b,#0f172a);border:1px solid #1e293b;
              border-radius:16px;padding:20px;margin-bottom:20px">
    <div style="font-size:12px;color:#64748b;letter-spacing:.8px;margin-bottom:6px">AI RECOMMENDATION</div>
    <div style="font-size:14px;color:#cbd5e1;line-height:1.7">{result.summary}</div>
  </div>

  <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:20px">{store_pills}</div>

  <div style="font-size:11px;color:#475569;margin-bottom:12px;letter-spacing:.6px">
    {len(result.ranked)} RESULTS — SORTED BY DEAL SCORE
  </div>

  {cards}
</div>
"""


def render_log(steps: list[dict]) -> str:
    if not steps:
        return "_Agents standing by…_"

    icons = {"running": "🔄", "done": "✅", "error": "❌", "pending": "⬜", "awaiting_human": "🟡"}
    lines = []
    for s in steps:
        icon = icons.get(s["status"], "•")
        line = f"{icon} **[{s['agent']}]** {s['task']}"
        if s["status"] in ("done",) and s["result"]:
            line += f"\n   ↳ _{s['result'][:100]}_"
        lines.append(line)
    return "\n\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Search logic (runs in background thread)
# ─────────────────────────────────────────────────────────────────────────────

def _do_search(query: str, holder: list) -> None:
    try:
        result = orchestrator.orchestrate(query, max_per_store=3)
        holder.append(("ok", result))
    except Exception as exc:
        holder.append(("error", str(exc)))


def search(query: str):
    """Generator — yields (results_html, log_md) updates while agents run."""
    if not query.strip():
        yield '<div style="color:#64748b;padding:40px;text-align:center">Enter a product to search</div>', ""
        return

    yield (
        '<div style="color:#94a3b8;padding:60px;text-align:center;font-size:14px">'
        '🔍 Searching 8 stores…</div>',
        "_Starting agents…_",
    )

    holder: list = []
    thread = threading.Thread(target=_do_search, args=(query, holder), daemon=True)
    thread.start()

    steps: list[dict] = []
    while thread.is_alive() or not _step_queue.empty():
        while not _step_queue.empty():
            steps.append(_step_queue.get_nowait())
        yield (
            '<div style="color:#94a3b8;padding:60px;text-align:center;font-size:14px">'
            f'🔍 Searching… ({len(steps)} steps completed)</div>',
            render_log(steps),
        )
        time.sleep(0.3)

    # drain
    while not _step_queue.empty():
        steps.append(_step_queue.get_nowait())

    if holder:
        kind, payload = holder[0]
        if kind == "ok":
            yield render_results(payload), render_log(steps)
        else:
            yield f'<div style="color:#f87171;padding:20px">Error: {payload}</div>', render_log(steps)
    else:
        yield '<div style="color:#f87171;padding:20px">No response received.</div>', render_log(steps)


# ─────────────────────────────────────────────────────────────────────────────
# CSS + UI
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700;900&family=Syne:wght@700;800&display=swap');

body, .gradio-container {
    background: #020817 !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #e2e8f0 !important;
}

/* hero header */
#sg-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    position: relative;
}
#sg-header::before {
    content: '';
    position: absolute;
    top: 0; left: 50%; transform: translateX(-50%);
    width: 600px; height: 260px;
    background: radial-gradient(ellipse at center, #f59e0b18 0%, transparent 70%);
    pointer-events: none;
}
#sg-header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -1.5px;
    background: linear-gradient(135deg, #f59e0b 0%, #fb923c 50%, #f43f5e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.3rem;
    line-height: 1;
}
#sg-header p {
    color: #475569;
    font-size: 1rem;
    margin: 0 0 1.2rem;
}

/* store badges */
.store-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin: 3px;
}

/* search box */
#search-row {
    max-width: 700px;
    margin: 0 auto 2rem;
    display: flex;
    gap: 10px;
}
#search-input textarea {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 14px !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 14px 18px !important;
}
#search-input textarea:focus {
    border-color: #f59e0b !important;
    outline: none !important;
}
#search-btn {
    background: linear-gradient(135deg, #f59e0b, #fb923c) !important;
    border: none !important;
    border-radius: 14px !important;
    color: #0f172a !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    min-width: 110px !important;
    height: 52px !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0.3px !important;
}
#search-btn:hover { opacity: 0.88 !important; }

/* results panel */
#results-panel {
    background: transparent !important;
    border: none !important;
    min-height: 300px;
}

/* log panel */
#log-panel {
    background: #0a0f1e !important;
    border: 1px solid #1e293b !important;
    border-radius: 14px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    padding: 14px !important;
    color: #475569 !important;
    min-height: 200px !important;
}

/* example chips */
.example-chip {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 20px;
    padding: 6px 16px;
    font-size: 12px;
    color: #94a3b8;
    cursor: pointer;
    transition: all .2s;
    display: inline-block;
    margin: 4px;
}
.example-chip:hover {
    border-color: #f59e0b;
    color: #f59e0b;
}

/* tabs */
.tab-nav button {
    background: transparent !important;
    color: #64748b !important;
    border-bottom: 2px solid transparent !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
}
.tab-nav button.selected {
    color: #f59e0b !important;
    border-bottom-color: #f59e0b !important;
}
"""

EXAMPLES = [
    "iPhone 15 Pro",
    "Sony WH-1000XM5 headphones",
    "Samsung 55 inch 4K TV",
    "Nike Air Max 270",
    "MacBook Air M2",
    "Canon EOS R50 camera",
]

with gr.Blocks(css=CSS, title="Shop Go — AI Price Comparison") as demo:

    # ── Hero header ──────────────────────────────────────────────────────────
    gr.HTML("""
    <div id="sg-header">
      <h1>Shop Go</h1>
      <p>AI-powered price comparison across 8 global &amp; local stores</p>
      <div>
        <span class="store-badge" style="background:#FF990022;color:#FF9900;border:1px solid #FF990044">🛒 Amazon</span>
        <span class="store-badge" style="background:#E5323822;color:#E53238;border:1px solid #E5323844">🏷️ eBay</span>
        <span class="store-badge" style="background:#FF474722;color:#FF4747;border:1px solid #FF474744">📦 AliExpress</span>
        <span class="store-badge" style="background:#0071CE22;color:#0071CE;border:1px solid #0071CE44">🏪 Walmart</span>
        <span class="store-badge" style="background:#F68B1E22;color:#F68B1E;border:1px solid #F68B1E44">🌍 Jumia</span>
        <span class="store-badge" style="background:#e91e8c22;color:#e91e8c;border:1px solid #e91e8c44">🦁 Kilimall</span>
        <span class="store-badge" style="background:#00924722;color:#009247;border:1px solid #00924744">📱 Masoko</span>
        <span class="store-badge" style="background:#388E3C22;color:#388E3C;border:1px solid #388E3C44">🤝 Jiji</span>
      </div>
    </div>
    """)

    # ── Search bar ───────────────────────────────────────────────────────────
    with gr.Row(elem_id="search-row"):
        search_input = gr.Textbox(
            placeholder="Search for any product… e.g. iPhone 15, Sony headphones, Nike shoes",
            show_label=False,
            lines=1,
            elem_id="search-input",
            scale=5,
        )
        search_btn = gr.Button("🔍 Search", elem_id="search-btn", scale=1)

    # ── Example chips ────────────────────────────────────────────────────────
    gr.HTML(
        '<div style="text-align:center;margin-bottom:28px">'
        + "".join(
            f'<span class="example-chip" onclick="document.querySelector(\'#search-input textarea\').value=\'{e}\';'
            f'document.querySelector(\'#search-input textarea\').dispatchEvent(new Event(\'input\'))">{e}</span>'
            for e in EXAMPLES
        )
        + "</div>"
    )

    # ── Main content ─────────────────────────────────────────────────────────
    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            results_html = gr.HTML(
                value='<div style="color:#334155;text-align:center;padding:80px;font-size:14px">'
                      '✦ Search for a product to see deals from 8 stores</div>',
                elem_id="results-panel",
            )

        with gr.Column(scale=1):
            gr.Markdown("### 🤖 Agent Activity")
            log_md = gr.Markdown(value="_Agents standing by…_", elem_id="log-panel")

    # ── Event wiring ─────────────────────────────────────────────────────────
    search_btn.click(
        fn=search,
        inputs=[search_input],
        outputs=[results_html, log_md],
        show_progress=False,
    )
    search_input.submit(
        fn=search,
        inputs=[search_input],
        outputs=[results_html, log_md],
        show_progress=False,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Launch
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠️  Set ANTHROPIC_API_KEY before running.")
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
    )
