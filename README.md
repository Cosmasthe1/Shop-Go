# Shop Go

Shop Go is a prototype shopping price-comparison assistant built in Python.
It uses a lightweight multi-agent architecture to simulate product search across multiple stores, score listings, and present ranked deals through a Gradio UI.

## Key Features

- Simulated multi-store search across global and regional marketplaces
- Composite deal scoring using price, rating, trust, and delivery speed
- Agent orchestration with a search, compare, and deal-ranking workflow
- Live Gradio interface with rich HTML result cards and execution logs
- Pluggable store adapter layer for an easy transition to real APIs

## Files

- `app.py` — Main application entry point and Gradio UI renderer.
- `orchestrator_agent.py` — Multi-agent orchestrator (`ShopGoOrchestrator`) with search, compare, and deal agents.
- `base_agent.py` — Shared agent framework, memory helpers, and tool registry.
- `stores.py` — Simulated store adapters and product listing generation.
- `requirements.txt` — Python dependency list.
- `shopgo_ui_preview.html` — UI preview / design reference.

## Requirements

- Python 3.11+ recommended
- `anthropic` Python SDK
- `gradio`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set your Anthropic API key:
   ```bash
   export ANTHROPIC_API_KEY="sk-..."
   ```

3. Run the app:
   ```bash
   python app.py
   ```

4. Open the local Gradio URL shown in the terminal.

## How It Works

1. `app.py` starts the Gradio interface and initializes `ShopGoOrchestrator`.
2. A user query is sent to `ShopGoOrchestrator.orchestrate()`.
3. `SearchAgent` queries every store adapter in parallel to collect listings.
4. `CompareAgent` scores and ranks the collected listings.
5. `DealAgent` selects top deals, assigns badges, and generates a summary.
6. The UI renders cards, store breakdowns, and a live execution log.

## Notes

- Store results are simulated in `stores.py`; replace these adapters with real APIs or scrapers for production.
- The agent framework in `base_agent.py` is designed around Anthropic tool-enabled message loops.
- There is built-in support for human-in-the-loop (HITL) checkpointing and execution logging.

## Extending Shop Go

- Add new store adapters in `stores.py` by implementing `search(query, max_results)`.
- Improve scoring or badge logic in `orchestrator_agent.py`.
- Replace the simulated data layer with real HTTP/API calls.

## License

This project is a demo prototype and does not include a license file.
# Shop-Go
