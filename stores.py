"""
stores.py — Shop Go
────────────────────────────────────────────────────────────────────────────
Store registry and per-store adapter layer.

Each adapter implements:
    search(query, max_results) → list[ProductListing]

In production swap the simulated data with real HTTP requests / scrapers /
official partner APIs. The adapter interface stays the same — only the
internals change.

Registered stores (global + East-Africa local mix):
  Global  : Amazon, eBay, AliExpress, Walmart
  Regional: Jumia (KE/NG/EG), Kilimall, Masoko (Safaricom), Jiji
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field
from typing import Protocol


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProductListing:
    store:          str           # e.g. "Amazon"
    store_region:   str           # "global" | "africa"
    title:          str
    price:          float         # normalised to USD
    currency:       str           # original currency code
    original_price: float         # in original currency
    url:            str
    image_url:      str
    description:    str
    rating:         float         # 0-5
    review_count:   int
    in_stock:       bool
    delivery_days:  int           # estimated delivery days
    store_logo:     str           # emoji or short label
    badge:          str = ""      # "Best Deal", "Fastest", "Top Rated", …

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ─────────────────────────────────────────────────────────────────────────────
# Store adapter protocol
# ─────────────────────────────────────────────────────────────────────────────

class StoreAdapter(Protocol):
    store_name:   str
    store_region: str
    store_logo:   str

    def search(self, query: str, max_results: int = 5) -> list[ProductListing]:
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

# Unsplash-based deterministic product images by keyword
_IMAGE_SEEDS = {
    "phone":      "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=400",
    "laptop":     "https://images.unsplash.com/photo-1517336714731-489689fd1ca8?w=400",
    "headphone":  "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=400",
    "shoe":       "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400",
    "watch":      "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=400",
    "bag":        "https://images.unsplash.com/photo-1548036328-c9fa89d128fa?w=400",
    "tv":         "https://images.unsplash.com/photo-1593359677879-a4bb92f4834c?w=400",
    "camera":     "https://images.unsplash.com/photo-1502920917128-1aa500764cbd?w=400",
    "shirt":      "https://images.unsplash.com/photo-1529374255404-311a2a4f1fd9?w=400",
    "book":       "https://images.unsplash.com/photo-1512820790803-83ca734da794?w=400",
    "default":    "https://images.unsplash.com/photo-1583847268964-b28dc8f51f92?w=400",
}

def _img(query: str) -> str:
    q = query.lower()
    for kw, url in _IMAGE_SEEDS.items():
        if kw in q:
            return url
    return _IMAGE_SEEDS["default"]

def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

def _vary(base: float, pct: float = 0.25) -> float:
    """Return base ± pct% variation."""
    delta = base * pct
    return round(random.uniform(base - delta, base + delta), 2)

def _usd_to(amount_usd: float, currency: str) -> float:
    rates = {
        "USD": 1.0,  "GBP": 0.79, "EUR": 0.92, "KES": 130.0,
        "NGN": 1550.0, "EGP": 48.0, "UGX": 3700.0, "CNY": 7.25,
    }
    return round(amount_usd * rates.get(currency, 1.0), 2)


# ─────────────────────────────────────────────────────────────────────────────
# Simulated store adapters
# In production, replace `_simulate_results` body with real API / HTTP calls.
# ─────────────────────────────────────────────────────────────────────────────

class _BaseSimulatedAdapter:
    store_name:   str
    store_region: str
    store_logo:   str
    _base_url:    str
    _currency:    str
    _price_mult:  float = 1.0   # relative price level vs USD reference
    _delivery:    tuple = (3, 10)

    def search(self, query: str, max_results: int = 5) -> list[ProductListing]:
        results = []
        ref_price = self._reference_price(query)
        for i in range(max_results):
            price_usd = _vary(ref_price * self._price_mult)
            orig      = _usd_to(price_usd, self._currency)
            results.append(ProductListing(
                store          = self.store_name,
                store_region   = self.store_region,
                store_logo     = self.store_logo,
                title          = self._title(query, i),
                price          = price_usd,
                currency       = self._currency,
                original_price = orig,
                url            = f"{self._base_url}/s/{_slug(query)}-{i+1}",
                image_url      = _img(query),
                description    = self._desc(query, i),
                rating         = round(random.uniform(3.5, 5.0), 1),
                review_count   = random.randint(12, 8000),
                in_stock       = random.random() > 0.1,
                delivery_days  = random.randint(*self._delivery),
            ))
        return results

    # ── subclass hooks ────────────────────────────────────────────────────────

    def _reference_price(self, query: str) -> float:
        """Return a realistic USD base price for the query."""
        q = query.lower()
        if any(w in q for w in ["iphone","samsung","pixel","smartphone","phone"]):
            return 699.0
        if any(w in q for w in ["macbook","laptop","notebook","chromebook"]):
            return 1099.0
        if any(w in q for w in ["airpod","headphone","earbud","buds"]):
            return 149.0
        if any(w in q for w in ["watch","smartwatch"]):
            return 249.0
        if any(w in q for w in ["tv","television","oled","qled"]):
            return 599.0
        if any(w in q for w in ["camera","dslr","mirrorless"]):
            return 799.0
        if any(w in q for w in ["shoe","sneaker","boot","trainer"]):
            return 89.0
        if any(w in q for w in ["bag","backpack","handbag"]):
            return 59.0
        if any(w in q for w in ["shirt","dress","jacket","trouser","jeans"]):
            return 39.0
        return 49.0

    def _title(self, query: str, idx: int) -> str:
        variants = [
            f"{query.title()} — {self.store_name} Pick #{idx+1}",
            f"Best {query.title()} {2024+idx//3}",
            f"{query.title()} Pro Edition",
            f"Premium {query.title()} Bundle",
            f"{query.title()} (Top Seller)",
        ]
        return variants[idx % len(variants)]

    def _desc(self, query: str, idx: int) -> str:
        descs = [
            f"High-quality {query} with free returns. Includes 1-year warranty.",
            f"Top-rated {query} on {self.store_name}. Fast dispatch, secure checkout.",
            f"Genuine {query} — verified seller. Over 1,000 happy buyers.",
            f"Exclusive {query} deal. Limited stock remaining. Ships from local warehouse.",
            f"Award-winning {query}. Rated best value by independent reviewers.",
        ]
        return descs[idx % len(descs)]


# ── Global stores ─────────────────────────────────────────────────────────────

class AmazonAdapter(_BaseSimulatedAdapter):
    store_name   = "Amazon"
    store_region = "global"
    store_logo   = "🛒"
    _base_url    = "https://www.amazon.com"
    _currency    = "USD"
    _price_mult  = 1.0
    _delivery    = (1, 5)


class eBayAdapter(_BaseSimulatedAdapter):
    store_name   = "eBay"
    store_region = "global"
    store_logo   = "🏷️"
    _base_url    = "https://www.ebay.com"
    _currency    = "USD"
    _price_mult  = 0.88    # eBay typically cheaper
    _delivery    = (3, 10)


class AliExpressAdapter(_BaseSimulatedAdapter):
    store_name   = "AliExpress"
    store_region = "global"
    store_logo   = "📦"
    _base_url    = "https://www.aliexpress.com"
    _currency    = "USD"
    _price_mult  = 0.62    # lowest prices, longer shipping
    _delivery    = (10, 30)


class WalmartAdapter(_BaseSimulatedAdapter):
    store_name   = "Walmart"
    store_region = "global"
    store_logo   = "🏪"
    _base_url    = "https://www.walmart.com"
    _currency    = "USD"
    _price_mult  = 0.94
    _delivery    = (2, 6)


# ── East Africa / regional stores ─────────────────────────────────────────────

class JumiaAdapter(_BaseSimulatedAdapter):
    store_name   = "Jumia"
    store_region = "africa"
    store_logo   = "🌍"
    _base_url    = "https://www.jumia.co.ke"
    _currency    = "KES"
    _price_mult  = 1.08    # slight import premium
    _delivery    = (2, 7)


class KilimallAdapter(_BaseSimulatedAdapter):
    store_name   = "Kilimall"
    store_region = "africa"
    store_logo   = "🦁"
    _base_url    = "https://www.kilimall.co.ke"
    _currency    = "KES"
    _price_mult  = 0.97
    _delivery    = (3, 8)


class MasokoAdapter(_BaseSimulatedAdapter):
    store_name   = "Masoko"
    store_region = "africa"
    store_logo   = "📱"
    _base_url    = "https://www.masoko.com"
    _currency    = "KES"
    _price_mult  = 1.05
    _delivery    = (1, 4)    # Safaricom logistics — fast local


class JijiAdapter(_BaseSimulatedAdapter):
    store_name   = "Jiji"
    store_region = "africa"
    store_logo   = "🤝"
    _base_url    = "https://jiji.co.ke"
    _currency    = "KES"
    _price_mult  = 0.75    # peer-to-peer, lowest local prices
    _delivery    = (1, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Store registry
# ─────────────────────────────────────────────────────────────────────────────

ALL_STORES: list[_BaseSimulatedAdapter] = [
    AmazonAdapter(),
    eBayAdapter(),
    AliExpressAdapter(),
    WalmartAdapter(),
    JumiaAdapter(),
    KilimallAdapter(),
    MasokoAdapter(),
    JijiAdapter(),
]

STORE_MAP: dict[str, _BaseSimulatedAdapter] = {s.store_name: s for s in ALL_STORES}


def get_stores(names: list[str] | None = None) -> list[_BaseSimulatedAdapter]:
    if names is None:
        return ALL_STORES
    return [STORE_MAP[n] for n in names if n in STORE_MAP]
