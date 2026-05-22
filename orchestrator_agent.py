"""
orchestrator_agent.py — Shop Go
────────────────────────────────────────────────────────────────────────────
Multi-agent orchestrator for the Shop Go shopping assistant.

Agent roster
  ┌──────────────────────────────────────────────────────────┐
  │                   ShopGoOrchestrator                     │
  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
  │   │  SearchAgent  │  │ CompareAgent │  │  DealAgent   │  │
  │   │ (find items) │  │ (rank/score) │  │ (best picks) │  │
  │   └──────────────┘  └──────────────┘  └──────────────┘  │
  │              MCP layer  ·  Memory  ·  HITL               │
  └──────────────────────────────────────────────────────────┘

Flow for a user query:
  1. SearchAgent  → queries all stores in parallel (threads), collects listings
  2. CompareAgent → scores each listing (price, rating, delivery, trust)
  3. DealAgent    → selects top deals, assigns badges, writes summary text
  4. Orchestrator → merges + returns structured JSON to the UI
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from base_agent import BaseAgent, LongTermMemory, ShortTermMemory, ToolRegistry
from stores import ALL_STORES, ProductListing, get_stores

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Execution log
# ─────────────────────────────────────────────────────────────────────────────

class StepStatus(str, Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    AWAITING = "awaiting_human"
    DONE     = "done"
    ERROR    = "error"


@dataclass
class ExecutionStep:
    agent_name: str
    task:       str
    status:     StepStatus = StepStatus.PENDING
    result:     str        = ""
    timestamp:  float      = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "agent":     self.agent_name,
            "task":      self.task,
            "status":    self.status.value,
            "result":    self.result[:200] + ("…" if len(self.result) > 200 else ""),
            "timestamp": self.timestamp,
        }


class ExecutionLog:
    def __init__(self) -> None:
        self._steps: list[ExecutionStep] = []

    def add(self, step: ExecutionStep) -> ExecutionStep:
        self._steps.append(step)
        return step

    def as_dicts(self) -> list[dict]:
        return [s.to_dict() for s in self._steps]

    def clear(self) -> None:
        self._steps.clear()


# ─────────────────────────────────────────────────────────────────────────────
# HITL gate
# ─────────────────────────────────────────────────────────────────────────────

class HITLGate:
    def __init__(self) -> None:
        self._fn: Callable[[str], tuple[bool, str]] | None = None

    def register(self, fn: Callable) -> None:
        self._fn = fn

    def checkpoint(self, prompt: str) -> tuple[bool, str]:
        if self._fn is None:
            logger.warning("HITL: no handler → auto-approving")
            return True, ""
        return self._fn(prompt)


# ─────────────────────────────────────────────────────────────────────────────
# MCP client
# ─────────────────────────────────────────────────────────────────────────────

class MCPClient:
    """Thin MCP abstraction. Wire to real MCP servers in production."""

    def __init__(self) -> None:
        self._servers: dict[str, dict] = {}

    def connect_server(self, name: str, description: str, tools: dict) -> None:
        self._servers[name] = {"description": description, "tools": tools}
        logger.info("MCP server connected: %s", name)

    def call_tool(self, server: str, tool: str, **kwargs) -> Any:
        return self._servers[server]["tools"][tool](**kwargs)

    def list_servers(self) -> list[str]:
        return list(self._servers.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Scoring helpers
# ─────────────────────────────────────────────────────────────────────────────

# Trust multiplier per store (0-1). Real system would use dynamic reviews.
STORE_TRUST = {
    "Amazon":     0.97,
    "eBay":       0.88,
    "AliExpress": 0.72,
    "Walmart":    0.94,
    "Jumia":      0.90,
    "Kilimall":   0.82,
    "Masoko":     0.86,
    "Jiji":       0.78,
}

def score_listing(listing: ProductListing, all_prices: list[float]) -> float:
    """
    Composite deal score 0-100.
      40% price rank  (cheaper = better)
      25% rating
      20% trust
      15% delivery speed
    """
    if not all_prices:
        return 50.0

    min_p, max_p = min(all_prices), max(all_prices)
    price_score = (
        100 * (max_p - listing.price) / (max_p - min_p)
        if max_p > min_p else 50.0
    )
    rating_score   = listing.rating / 5.0 * 100
    trust_score    = STORE_TRUST.get(listing.store, 0.80) * 100
    delivery_score = max(0, 100 - listing.delivery_days * 4)

    return round(
        0.40 * price_score
        + 0.25 * rating_score
        + 0.20 * trust_score
        + 0.15 * delivery_score,
        2,
    )


def assign_badges(ranked: list[tuple[ProductListing, float]]) -> list[tuple[ProductListing, float, str]]:
    """Assign human-readable deal badges to top listings."""
    result = []
    prices        = [l.price for l, _ in ranked]
    delivery_days = [l.delivery_days for l, _ in ranked]
    ratings       = [l.rating for l, _ in ranked]

    best_price_idx    = prices.index(min(prices))
    fastest_idx       = delivery_days.index(min(delivery_days))
    top_rated_idx     = ratings.index(max(ratings))

    for i, (listing, score) in enumerate(ranked):
        badge = ""
        if i == 0:
            badge = "🏆 Best Deal"
        elif i == best_price_idx:
            badge = "💰 Lowest Price"
        elif i == fastest_idx:
            badge = "⚡ Fastest Delivery"
        elif i == top_rated_idx:
            badge = "⭐ Top Rated"
        elif listing.store_region == "africa":
            badge = "🌍 Local Pick"
        result.append((listing, score, badge))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Specialised agents
# ─────────────────────────────────────────────────────────────────────────────

class SearchAgent(BaseAgent):
    """
    Queries all registered stores in parallel and returns raw listings.
    Uses threads — one per store — for fast concurrent fetching.
    """

    name   = "SearchAgent"
    system = textwrap.dedent("""\
        You are a product search specialist for Shop Go.
        Your job is to interpret user shopping queries and extract:
          - product_name: the core product the user wants
          - filters: any constraints (brand, budget, colour, size, etc.)
        Respond ONLY with a JSON object: {"product_name": "...", "filters": {...}}
    """)

    def __init__(self, model: str | None = None) -> None:
        super().__init__(model=model)
        self._stores = ALL_STORES

    def search_all_stores(
        self,
        query: str,
        max_per_store: int = 3,
        store_names: list[str] | None = None,
    ) -> list[ProductListing]:
        stores = get_stores(store_names) if store_names else self._stores
        all_results: list[ProductListing] = []

        with ThreadPoolExecutor(max_workers=len(stores)) as executor:
            futures = {
                executor.submit(store.search, query, max_per_store): store.store_name
                for store in stores
            }
            for future in as_completed(futures):
                store_name = futures[future]
                try:
                    results = future.result(timeout=10)
                    all_results.extend(results)
                    logger.info("SearchAgent: %s → %d results", store_name, len(results))
                except Exception as exc:
                    logger.warning("SearchAgent: %s failed — %s", store_name, exc)

        return all_results

    def parse_query(self, raw_query: str) -> dict:
        """Use the LLM to extract structured intent from a free-text query."""
        try:
            raw = self.run(raw_query, inject_history=False)
            clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(clean)
        except Exception:
            return {"product_name": raw_query, "filters": {}}


class CompareAgent(BaseAgent):
    """
    Scores and ranks product listings.
    Uses tool calling to calculate composite deal scores.
    """

    name   = "CompareAgent"
    system = textwrap.dedent("""\
        You are a price comparison expert for Shop Go.
        Given a list of product listings you analyse value, reliability, and speed.
        Use the score_listings tool to produce a ranked comparison.
        Return ONLY the JSON result of the tool call.
    """)

    def _register_tools(self) -> None:
        self._tool_registry.register(
            {
                "name": "score_listings",
                "description": "Score and rank a list of product listings by deal quality.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "listings_json": {
                            "type": "string",
                            "description": "JSON array of ProductListing dicts",
                        }
                    },
                    "required": ["listings_json"],
                },
            },
            self._score_listings_tool,
        )

    @staticmethod
    def _score_listings_tool(listings_json: str) -> dict:
        listings = [ProductListing(**d) for d in json.loads(listings_json)]
        all_prices = [l.price for l in listings]
        scored = [(l, score_listing(l, all_prices)) for l in listings]
        scored.sort(key=lambda x: x[1], reverse=True)
        with_badges = assign_badges(scored)
        return {
            "ranked": [
                {**l.to_dict(), "deal_score": s, "badge": b}
                for l, s, b in with_badges
            ]
        }

    def compare(self, listings: list[ProductListing]) -> list[dict]:
        """Score and rank listings, returning enriched dicts."""
        all_prices  = [l.price for l in listings]
        scored      = [(l, score_listing(l, all_prices)) for l in listings]
        scored.sort(key=lambda x: x[1], reverse=True)
        with_badges = assign_badges(scored)
        return [
            {**l.to_dict(), "deal_score": s, "badge": b}
            for l, s, b in with_badges
        ]


class DealAgent(BaseAgent):
    """
    Selects the single best deal per category and writes a
    human-readable summary / recommendation.
    """

    name   = "DealAgent"
    system = textwrap.dedent("""\
        You are Shop Go's deal curator. Given a ranked list of products,
        write a short, punchy recommendation (3-4 sentences) explaining:
          • Why the top pick is the best deal overall
          • When you'd choose the cheapest alternative instead
          • When you'd pay more for the fastest / best-rated option
        Be direct, helpful, and honest. No fluff.
    """)

    def summarise(self, query: str, ranked: list[dict]) -> str:
        """Generate a natural language deal summary for the top results."""
        top5 = ranked[:5]
        context = json.dumps([
            {
                "store":    r["store"],
                "title":    r["title"],
                "price":    f"${r['price']:.2f}",
                "rating":   r["rating"],
                "delivery": f"{r['delivery_days']}d",
                "badge":    r["badge"],
                "score":    r["deal_score"],
            }
            for r in top5
        ], indent=2)

        prompt = (
            f"User searched for: \"{query}\"\n\n"
            f"Top results:\n{context}\n\n"
            "Write the deal summary now."
        )
        return self.run(prompt, inject_history=False)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class ShopGoOrchestrator:
    """
    Top-level coordinator for Shop Go.

    orchestrate(query) → ShopGoResult

    Pipeline:
      1. SearchAgent.parse_query()        → structured intent
      2. SearchAgent.search_all_stores()  → raw listings (parallel)
      3. CompareAgent.compare()           → scored + ranked listings
      4. DealAgent.summarise()            → natural language recommendation
      5. Memory                           → store search in history
    """

    def __init__(self, model: str | None = None) -> None:
        self._model    = model
        self.search    = SearchAgent(model=model)
        self.compare   = CompareAgent(model=model)
        self.deal      = DealAgent(model=model)
        self.log       = ExecutionLog()
        self.hitl      = HITLGate()
        self.mcp       = MCPClient()
        self.memory    = LongTermMemory()
        self._on_step: Callable[[dict], None] | None = None

        self._wire_mcp()

    # ── public configuration ─────────────────────────────────────────────────

    def on_step_update(self, fn: Callable[[dict], None]) -> None:
        self._on_step = fn

    # ── main pipeline ────────────────────────────────────────────────────────

    def orchestrate(self, query: str, max_per_store: int = 3) -> "ShopGoResult":
        self.log.clear()
        logger.info("ShopGo ← %s", query)

        # ── 1. Parse intent ──────────────────────────────────────────────────
        s1 = self._step("SearchAgent", f"Parse query: {query}")
        parsed = self.search.parse_query(query)
        product_name = parsed.get("product_name", query)
        filters      = parsed.get("filters", {})
        self._done(s1, f"product={product_name}, filters={filters}")

        # ── 2. Search all stores ─────────────────────────────────────────────
        s2 = self._step("SearchAgent", f"Search all stores for '{product_name}'")
        listings = self.search.search_all_stores(product_name, max_per_store=max_per_store)
        self._done(s2, f"{len(listings)} listings collected")

        if not listings:
            return ShopGoResult(query=query, ranked=[], summary="No results found.", log=self.log.as_dicts())

        # ── 3. Score & rank ──────────────────────────────────────────────────
        s3 = self._step("CompareAgent", f"Score & rank {len(listings)} listings")
        ranked = self.compare.compare(listings)
        self._done(s3, f"Top: {ranked[0]['store']} @ ${ranked[0]['price']:.2f}")

        # ── 4. Deal summary ──────────────────────────────────────────────────
        s4 = self._step("DealAgent", "Write deal recommendation")
        summary = self.deal.summarise(query, ranked)
        self._done(s4, summary[:120])

        # ── 5. Persist to memory ─────────────────────────────────────────────
        ts = int(time.time())
        self.memory.remember(f"search:{ts}", {
            "query":   query,
            "top":     ranked[0]["store"],
            "price":   ranked[0]["price"],
            "results": len(ranked),
        })

        return ShopGoResult(
            query      = query,
            product    = product_name,
            filters    = filters,
            ranked     = ranked,
            summary    = summary,
            log        = self.log.as_dicts(),
            search_history = list(self.memory.search("search:").values()),
        )

    # ── internal helpers ─────────────────────────────────────────────────────

    def _step(self, agent: str, task: str) -> ExecutionStep:
        step = self.log.add(ExecutionStep(agent_name=agent, task=task, status=StepStatus.RUNNING))
        self._notify(step)
        return step

    def _done(self, step: ExecutionStep, result: str) -> None:
        step.status = StepStatus.DONE
        step.result = result
        self._notify(step)

    def _notify(self, step: ExecutionStep) -> None:
        if self._on_step:
            try:
                self._on_step(step.to_dict())
            except Exception:
                pass

    def _wire_mcp(self) -> None:
        """Built-in MCP servers for Shop Go."""

        def price_alert_set(product: str, target_price: float) -> str:
            """Set a price alert for a product."""
            self.memory.remember(f"alert:{product}", {"target": target_price, "active": True})
            return f"Alert set: notify when '{product}' drops below ${target_price:.2f}"

        def price_alert_check(product: str) -> dict:
            """Check if any price alerts have triggered."""
            return self.memory.recall(f"alert:{product}", default={"active": False})

        def wishlist_add(product: str, store: str, price: float) -> str:
            """Add a product to the wishlist."""
            self.memory.remember(f"wish:{product}", {"store": store, "price": price})
            return f"Added '{product}' from {store} @ ${price:.2f} to wishlist"

        def wishlist_get() -> list:
            """Get all wishlist items."""
            return list(self.memory.search("wish:").values())

        self.mcp.connect_server("price-alerts", "Price alert management", {
            "set_alert":   price_alert_set,
            "check_alert": price_alert_check,
        })
        self.mcp.connect_server("wishlist", "User wishlist", {
            "add":  wishlist_add,
            "get":  wishlist_get,
        })


# ─────────────────────────────────────────────────────────────────────────────
# Result model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ShopGoResult:
    query:          str
    ranked:         list[dict]
    summary:        str
    log:            list[dict]
    product:        str              = ""
    filters:        dict             = field(default_factory=dict)
    search_history: list[dict]       = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query":          self.query,
            "product":        self.product,
            "filters":        self.filters,
            "ranked":         self.ranked,
            "summary":        self.summary,
            "log":            self.log,
            "search_history": self.search_history,
        }
