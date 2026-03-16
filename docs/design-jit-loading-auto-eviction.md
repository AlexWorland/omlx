# Design: JIT Model Loading & Auto Model Eviction

**Date:** 2026-03-15
**Status:** Approved
**Branch:** feature/memory-pressure-management

## Overview

Add just-in-time model loading and automatic time-based model eviction to omlx. Models are discovered at startup and via periodic directory rescans, but weights are only loaded when a request arrives. Idle models are automatically unloaded after a configurable TTL, freeing unified memory for other models.

## Goals

1. **JIT loading** — Models exist in a "discovered but not loaded" state until a request triggers weight loading
2. **Hot-add** — New models dropped into the models directory are detected via periodic rescan without server restart
3. **Auto eviction** — Idle models are automatically unloaded after a configurable TTL (default: 180 seconds)
4. **Configurable** — All settings exposed in admin dashboard; per-model TTL overrides global default

## Non-Goals

- Predictive loading/eviction based on usage patterns
- Formal state machine abstraction (implicit states are sufficient)
- Remote model registry or download-on-demand
- Changes to the memory pressure system (dual-signal zones remain as-is)

## Existing Infrastructure

omlx already has significant infrastructure that this design builds on:

| Component | Status | What exists |
|-----------|--------|-------------|
| `ModelSettings.ttl_seconds` | Exists, unused | Per-model TTL field in settings |
| `EngineEntry.last_access` | Active | Updated on every model access |
| `EngineEntry.is_pinned` | Active | Prevents LRU eviction |
| `check_ttl_expirations()` | Exists, not wired | Iterates models, checks idle time |
| `_find_lru_victim()` | Active | LRU victim selection for memory pressure |
| `_unload_engine()` | Active | Full unload sequence with GC |
| `discover_models()` | Active | Scans directories, pre-calculates sizes |
| `_ensure_memory_available()` | Active | Pre-load memory check with eviction |

The work is primarily wiring these together and adding the periodic loops.

## Model States (Implicit)

No new state machine. Model state is derived from existing data structures:

| State | Representation |
|-------|---------------|
| **Discovered** | In `_model_registry` but no `EngineEntry` in `_engines` |
| **Loading** | `EngineEntry` exists, `loading_event` not yet set |
| **Loaded** | `EngineEntry` exists, `loading_event` set, engine active |
| **Evicted** | `EngineEntry` removed from `_engines`, model still in `_model_registry` |

`_model_registry` is a new `dict[str, ModelInfo]` that tracks all known models regardless of load state. It replaces the current pattern where `discover_models()` eagerly populates `_engines`.

### ModelInfo

```python
@dataclass
class ModelInfo:
    model_id: str
    model_path: Path
    model_type: str           # "llm", "vlm", "embedding", "reranker"
    weight_size_bytes: int    # Pre-calculated from safetensors
    discovered_at: float      # time.time() when first seen
```

## TTL Eviction Loop

### Server Lifespan Integration

A new `asyncio.Task` started in `server.py` lifespan:

```python
async def _ttl_eviction_loop(pool: EnginePool, settings_manager: ModelSettingsManager, interval: int = 30):
    """Check for idle models every 30 seconds and unload expired ones."""
    while True:
        await asyncio.sleep(interval)
        pool.check_ttl_expirations(settings_manager)
```

### Eviction Logic

`check_ttl_expirations()` already exists. The enhanced logic:

```
For each loaded model (EngineEntry in _engines):
  1. Skip if is_pinned
  2. Skip if currently loading (loading_event not set)
  3. Skip if has active requests (scheduler.active_request_count > 0)
  4. Determine effective TTL:
     a. Per-model ttl_seconds if set and > 0
     b. Else global_ttl_seconds from settings
     c. If effective TTL is 0, skip (TTL disabled for this model)
  5. Calculate idle_time = now - last_access
  6. If idle_time > effective_ttl:
     a. Log: "Auto-evicting {model_id} after {idle_time}s idle (TTL: {effective_ttl}s)"
     b. Call _unload_engine(model_id)
     c. Model remains in _model_registry (can be JIT-loaded again)
```

### Per-Model TTL Override

- `ModelSettings.ttl_seconds > 0` — use this value
- `ModelSettings.ttl_seconds == 0` — never auto-evict (TTL disabled for this model)
- `ModelSettings.ttl_seconds == None/unset` — fall back to `global_ttl_seconds`

## Periodic Model Rescan

### Rescan Loop

```python
async def _model_rescan_loop(pool: EnginePool, interval: int = 60):
    """Rescan model directories for new or removed models."""
    while True:
        await asyncio.sleep(interval)
        pool.rescan_models()
```

### rescan_models()

New method on `EnginePool`:

```
rescan_models():
  1. Run existing discover_models() logic to get current_models set
  2. Compare with _model_registry:
     - New models (in current_models but not registry):
       a. Add to _model_registry
       b. Log: "Hot-added model: {model_id} ({size_bytes} bytes)"
     - Removed models (in registry but not current_models):
       a. If currently loaded: log warning, skip removal
       b. If not loaded: remove from _model_registry
       c. Log: "Model removed: {model_id} (files no longer present)"
     - Existing models: no action
```

### Scan Interval Setting

`model_scan_interval_seconds` in `MemorySettings`:
- Default: `60` seconds
- Set to `0` to disable periodic rescanning
- Minimum: `10` seconds (to prevent excessive I/O)

## JIT Loading in get_engine()

### Modified Flow

```
get_engine(model_id) -> Engine:
  1. If model_id in _engines and loaded:
     → Update last_access, return engine

  2. If model_id in _engines and loading:
     → Wait for loading_event, return engine

  3. If model_id in _model_registry but not _engines:
     → JIT load path (see below)

  4. If model_id not in _model_registry:
     → Raise ModelNotFoundError (404)
```

### JIT Load Path

```
JIT load (model_id):
  If jit_loading_behavior == "block":
    1. _ensure_memory_available(model_info.weight_size_bytes)
       - May evict LRU models to make room (existing logic)
    2. _load_engine(model_id, model_info.model_path)
       - Creates EngineEntry, starts weight loading
    3. Wait for loading_event (blocks the request)
    4. Return engine

  If jit_loading_behavior == "reject":
    1. Start background load task (same as above, but async)
    2. Return HTTP 503 with:
       - Retry-After: estimated_load_seconds
       - Body: {"error": {"message": "Model loading", "type": "model_loading",
                "model": model_id, "estimated_seconds": N}}
    3. Subsequent requests for same model hit step 2 (wait for loading_event)
```

### Load Time Estimation

For the 503 response, estimate load time from model size:

```python
def _estimate_load_seconds(weight_size_bytes: int) -> int:
    """Estimate model load time based on weight size.

    Assumes ~2 GB/s read speed for Apple Silicon SSD.
    Adds 5s overhead for initialization.
    """
    return int(weight_size_bytes / (2 * 1024**3)) + 5
```

## Settings Changes

### New Fields in MemorySettings

| Setting | Type | Default | Validation | Description |
|---------|------|---------|------------|-------------|
| `global_ttl_seconds` | `int` | `180` | `>= 0` | Default TTL for models without per-model TTL. 0 = disabled. |
| `model_scan_interval_seconds` | `int` | `60` | `0 or >= 10` | How often to rescan model directories. 0 = disabled. |
| `jit_loading_behavior` | `str` | `"block"` | `"block"` or `"reject"` | How to handle requests for unloaded models. |

### Admin Dashboard UI

New "Model Lifecycle" section on the settings page:

```
Model Lifecycle
├── Global TTL (seconds): [180]
│   "Models auto-unload after this many seconds of inactivity. 0 to disable."
├── Model Scan Interval (seconds): [60]
│   "How often to check for new models in the models directory. 0 to disable."
└── Loading Behavior: [Block and Load ▼]
    "How to handle requests for unloaded models."
    Options: "Block and Load" | "Reject (503)"
```

### Per-Model TTL in Model Settings

The existing `ModelSettings.ttl_seconds` field is already available. The admin model settings UI should expose it:

```
Model: llama-3-8b
├── TTL (seconds): [    ] (blank = use global default, 0 = never evict)
├── Pinned: [ ] (pinned models are exempt from TTL eviction)
└── ...existing settings...
```

## API Changes

### GET /v1/models (Extended Response)

Add lifecycle fields to each model entry:

```json
{
  "data": [
    {
      "id": "llama-3-8b",
      "object": "model",
      "owned_by": "omlx",
      "status": "loaded",
      "idle_seconds": 45,
      "ttl_seconds": 180,
      "evicts_in_seconds": 135,
      "weight_size_bytes": 14000000000,
      "last_access": 1678886400
    },
    {
      "id": "phi-3-mini",
      "object": "model",
      "owned_by": "omlx",
      "status": "discovered",
      "weight_size_bytes": 7600000000
    }
  ]
}
```

Status values: `"discovered"`, `"loading"`, `"loaded"`

### POST /v1/models/{model_id}/load (New)

Pre-load a model without sending an inference request:

```
POST /v1/models/llama-3-8b/load

Response 200:
{"status": "loaded", "load_time_seconds": 12.4}

Response 202:
{"status": "loading", "estimated_seconds": 45}

Response 404:
{"error": {"message": "Model not found", "type": "model_not_found"}}

Response 507:
{"error": {"message": "Insufficient memory", "type": "insufficient_memory"}}
```

### POST /v1/models/{model_id}/unload (Existing)

Already exists. No changes needed.

## Interaction with Memory Pressure System

### TTL Eviction vs. Pressure Eviction

These are independent systems that complement each other:

- **TTL eviction**: Time-based, predictable, runs every 30s. Frees memory proactively.
- **Pressure eviction**: Reactive, triggered by memory zones. Frees memory urgently.

TTL eviction reduces the frequency of pressure eviction by keeping memory cleaner. The memory pressure system remains unchanged — if a model is loaded and memory hits CRITICAL, the existing `_evict_all_non_pinned()` still fires regardless of TTL.

### Pinned Models

Pinned models are exempt from both TTL and LRU eviction:
- `is_pinned = True` → never auto-evicted by TTL
- `is_pinned = True` → never selected as LRU victim
- `is_pinned = True` → only evicted by CRITICAL zone (existing behavior) or manual unload

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Model files disappear while loaded | Log warning, keep loaded until unloaded normally |
| Model files disappear while not loaded | Remove from registry on next rescan |
| Load fails (corrupt weights) | Remove EngineEntry, model stays in registry, log error |
| Memory insufficient for JIT load | Evict LRU models first; if still insufficient, return 507 |
| Multiple concurrent requests for unloaded model | First request triggers load, subsequent requests wait on same `loading_event` |

## Files Changed

| File | Changes |
|------|---------|
| `omlx/engine_pool.py` | Add `_model_registry`, `ModelInfo`, `rescan_models()`, modify `get_engine()` for JIT, modify `discover_models()` to populate registry |
| `omlx/server.py` | Start TTL loop + rescan loop in lifespan, wire settings |
| `omlx/settings.py` | Add `global_ttl_seconds`, `model_scan_interval_seconds`, `jit_loading_behavior` to `MemorySettings` |
| `omlx/admin/routes.py` | Expose new settings in API, extend model status response |
| `omlx/admin/templates/dashboard/_status.html` | Add model lifecycle info to status card |
| `omlx/admin/templates/settings.html` | New "Model Lifecycle" settings section |
| `omlx/admin/static/js/dashboard.js` | TTL countdown display, model status indicators |
| `omlx/admin/i18n/en.json` | Translation keys for new UI elements |
| `tests/test_jit_loading.py` | New test file covering JIT load, TTL eviction, rescan |

## Testing Strategy

### Unit Tests

```
test_ttl_evicts_idle_model           — Model unloaded after idle > TTL
test_ttl_respects_pinned             — Pinned models never TTL-evicted
test_ttl_skips_active_requests       — Models with in-flight requests kept
test_ttl_per_model_override          — Per-model TTL overrides global
test_ttl_zero_disables               — TTL of 0 means never auto-evict
test_ttl_global_default_fallback     — Uses global when per-model unset
test_rescan_finds_new_model          — New model directory detected
test_rescan_removes_missing_model    — Deleted model removed from registry
test_rescan_keeps_loaded_model       — Loaded model not removed even if files gone
test_jit_loads_on_request            — Discovered model loads on first request
test_jit_concurrent_requests         — Multiple requests share single load
test_jit_reject_mode                 — 503 returned in reject mode
test_jit_evict_and_reload            — Model evicted by TTL, reloaded on next request
test_model_registry_populated        — discover_models fills registry
test_settings_validation             — Invalid settings rejected
```

### Integration Tests

```
test_full_lifecycle                  — discover → load → serve → idle → evict → reload
test_hot_add_and_serve               — Drop model files, wait for rescan, send request
test_memory_pressure_during_jit      — JIT load triggers LRU eviction when memory tight
```

## Rollout

This feature is non-breaking. All changes are additive:

1. Models that are currently loaded at startup continue to work
2. The default `global_ttl_seconds: 180` will start auto-evicting idle models — users who want the old behavior can set it to `0`
3. Hot-add is passive (periodic rescan) — no filesystem watcher to install
4. JIT loading defaults to block-and-load — existing API clients work without changes
