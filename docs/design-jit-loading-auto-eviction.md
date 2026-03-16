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
| `ModelSettings.ttl_seconds` | Exists, per-model only | Per-model TTL field in settings; only evicts models with explicit TTL set |
| `EngineEntry.last_access` | Active | Updated on every model access |
| `EngineEntry.is_pinned` | Active | Prevents LRU eviction |
| `EngineEntry.is_loading` | Active | Boolean flag; currently raises `ModelLoadingError` for concurrent requests |
| `check_ttl_expirations()` | Active, limited | Wired into server lifespan TTL loop (and enforcer polling loop); skips models without explicit per-model TTL |
| `_find_lru_victim()` | Active | LRU victim selection for memory pressure |
| `_unload_engine()` | Active | Full unload sequence with GC |
| `discover_models()` | Active | Scans directories (requires `model_dirs`, `pinned_models` args), pre-calculates sizes |
| `_ensure_memory_available()` | Active | Pre-load memory check with eviction |

The main work is: (a) adding `global_ttl_seconds` fallback to `check_ttl_expirations()`, (b) adding the rescan loop, (c) implementing JIT load with concurrent request coordination, and (d) fixing lock contention during loads.

## Model States (Implicit)

No new state machine. Model state is derived from the existing `_entries` dictionary (which already supports unloaded entries where `engine=None`):

| State | Representation |
|-------|---------------|
| **Discovered** | `EngineEntry` in `_entries` with `engine=None` and `is_loading=False` |
| **Loading** | `EngineEntry` in `_entries` with `is_loading=True` and `loading_event` not yet set |
| **Loaded** | `EngineEntry` in `_entries` with `engine` set and `loading_event` signaled |
| **Evicted** | `EngineEntry` removed from `_entries`; re-discoverable via rescan |

No separate `_model_registry` — the existing `_entries` dict already tracks both discovered and loaded models. The `discover_models()` method populates `_entries` with unloaded entries (`engine=None`). After TTL eviction, a model is removed from `_entries` and rediscovered on the next rescan cycle.

### New Field: `loading_event`

`EngineEntry` currently has `is_loading: bool` and `abort_loading: bool` but no coordination mechanism for concurrent requests. When `is_loading=True`, the current code raises `ModelLoadingError`, rejecting concurrent requests outright.

**Add `loading_event: asyncio.Event`** to `EngineEntry`:

```python
@dataclass
class EngineEntry:
    # ... existing fields ...
    loading_event: asyncio.Event = field(default_factory=asyncio.Event)
```

- Created unset when `_load_engine()` starts
- Set when loading completes (success or failure)
- Concurrent requests `await loading_event.wait()` instead of raising `ModelLoadingError`
- On load failure, the event is still set but the `EngineEntry` is removed — waiters check for removal and raise appropriately

## TTL Eviction Loop

### Server Lifespan Integration

The TTL check loop already exists in the server lifespan — it runs as part of the enforcer's polling loop (when the enforcer is active) or as a standalone loop (when it's not). **No new loop is needed.** The change is internal to `check_ttl_expirations()`: adding the `global_ttl_seconds` fallback.

### Eviction Logic

`check_ttl_expirations()` already exists and is already wired. The enhanced logic adds global TTL fallback:

```
For each loaded model (EngineEntry in _entries where engine is not None):
  1. Skip if is_pinned
  2. Skip if currently loading (is_loading=True)
  3. Skip if has active requests (check engine._engine.engine._output_collectors)
  4. Determine effective TTL:
     a. Per-model ttl_seconds if set and > 0
     b. Else global_ttl_seconds from settings
     c. If effective TTL is 0, skip (TTL disabled for this model)
  5. Calculate idle_time = now - last_access
  6. If idle_time > effective_ttl:
     a. Log: "Auto-evicting {model_id} after {idle_time}s idle (TTL: {effective_ttl}s)"
     b. Call _unload_engine(model_id)
     c. Model removed from _entries (rediscovered on next rescan, or on explicit /load call)
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

New method on `EnginePool`. Uses `self._model_dirs` (stored at init from the `model_dirs` argument passed to `discover_models()`) and `settings_manager.get_pinned_model_ids()` for pinned status:

```
rescan_models(settings_manager):
  1. Run existing discover_models(self._model_dirs, pinned_ids) logic to get current_models set
  2. Compare with _entries:
     - New models (in current_models but not _entries):
       a. Add to _entries as unloaded EngineEntry (engine=None)
       b. Log: "Hot-added model: {model_id} ({size_bytes} bytes)"
     - Removed models (in _entries but not current_models):
       a. If currently loaded (engine is not None): log warning, skip removal
       b. If not loaded: remove from _entries
       c. Log: "Model removed: {model_id} (files no longer present)"
     - Existing models: no action
```

### Scan Interval Setting

`model_scan_interval_seconds` in `MemorySettings`:
- Default: `60` seconds
- Set to `0` to disable periodic rescanning
- Minimum: `10` seconds (to prevent excessive I/O)

## JIT Loading in get_engine()

### Lock Contention Fix

**Problem:** The current `get_engine()` holds `self._lock` (an `asyncio.Lock`) for the entire duration of `_load_engine()`, which can take 10-60+ seconds. This blocks ALL `get_engine()` calls, including requests for already-loaded models. With JIT loading, "model not loaded" becomes the normal case, making this unacceptable.

**Solution: Two-phase locking with `loading_event` coordination.**

```
Phase 1 (under lock, fast):
  - Check state, create EngineEntry placeholder with loading_event unset
  - Release lock

Phase 2 (outside lock, slow):
  - Load weights
  - Set loading_event when done

Concurrent requests:
  - For loaded models: acquire lock, find entry, return immediately
  - For loading models: acquire lock, find entry with is_loading=True, release lock,
    await loading_event.wait(), return
```

### Modified Flow

```
get_engine(model_id) -> Engine:
  Acquire lock:
    1. If model_id in _entries and loaded (engine is not None, not loading):
       → Update last_access, release lock, return engine

    2. If model_id in _entries and loading (is_loading=True):
       → Grab reference to loading_event, release lock
       → await loading_event.wait()
       → If entry was removed (load failed): raise ModelLoadError
       → Return engine

    3. If model_id in _entries and not loaded (engine=None, discovered state):
       → Create placeholder EngineEntry with is_loading=True, loading_event unset
       → Release lock
       → JIT load path (see below)

    4. If model_id not in _entries:
       → Release lock, raise ModelNotFoundError (404)
```

### JIT Load Path

```
JIT load (model_id):  [lock NOT held]
  1. _ensure_memory_available(weight_size_bytes)
     - May evict LRU models to make room (acquires lock internally)

  If jit_loading_behavior == "block":
    2. _load_engine(model_id, model_path)  [loads weights, may take 10-60s]
    3. Set loading_event (unblocks concurrent waiters)
    4. Return engine
    (On failure: set loading_event, remove entry from _entries, raise error)

  If jit_loading_behavior == "reject":
    2. Start background load task (same as above, but in asyncio.create_task)
    3. Return HTTP 503 with:
       - Retry-After: estimated_load_seconds
       - Body: {"error": {"message": "Model loading", "type": "model_loading",
                "model": model_id, "estimated_seconds": N}}
    4. Subsequent requests for same model hit step 2 of Modified Flow (wait on event)
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

Pre-load a model without sending an inference request. This endpoint always blocks until the model is loaded, regardless of the `jit_loading_behavior` setting (since it's an explicit user action, not an inference request).

```
POST /v1/models/llama-3-8b/load

Behavior matrix:
  - Already loaded          → 200 {"status": "loaded", "load_time_seconds": 0}
  - Not loaded              → blocks, loads, returns 200 {"status": "loaded", "load_time_seconds": 12.4}
  - Already loading (by JIT)→ waits for existing load, returns 200 {"status": "loaded", "load_time_seconds": 8.1}
  - Model not found         → 404 {"error": {"message": "Model not found", "type": "model_not_found"}}
  - Insufficient memory     → 507 {"error": {"message": "Insufficient memory", "type": "insufficient_memory"}}
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
| `omlx/engine_pool.py` | Add `loading_event` to `EngineEntry`, add `rescan_models()`, store `_model_dirs`, modify `get_engine()` for JIT with two-phase locking, add global TTL fallback to `check_ttl_expirations()` |
| `omlx/server.py` | Add rescan loop to lifespan, wire `global_ttl_seconds` into existing TTL check |
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

This feature introduces a **behavior change**: idle models will now auto-evict after 180 seconds by default. This should be called out in release notes.

1. Models that are currently loaded at startup continue to work
2. **Behavior change:** The default `global_ttl_seconds: 180` will start auto-evicting idle models. Users who want the previous "keep loaded forever" behavior should set `global_ttl_seconds: 0` in settings. This is intentional — the 3-minute default keeps unified memory available for active workloads.
3. Hot-add is passive (periodic rescan) — no filesystem watcher to install
4. JIT loading defaults to block-and-load — existing API clients work without changes (they may see higher latency on first request to an evicted model)
