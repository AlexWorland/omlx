# oMLX Scheduler & Memory Checking System - Comprehensive Exploration

**Last Updated:** 2026-03-15
**Focus:** Scheduler memory limits, batch generator memory checking, paged cache management, and memory estimation

---

## Executive Summary

The oMLX scheduler implements a **soft/hard memory limit architecture** for gating prefill request processing. Memory limits are checked at multiple stages:

1. **Pre-allocation phase** (SchedulerMemoryBudget): Estimates if new prefill chunk will fit within budget
2. **Prefill processing phase** (_BoundarySnapshotBatchGenerator): Checks active memory against hard limit after processing prefill boundary
3. **Block eviction phase** (PagedCacheManager): Evicts LRU blocks when memory pressure exceeds thresholds

The system operates in **paged SSD-only mode** where all KV cache data is stored on paged SSD, not GPU memory. GPU memory holds only model weights and block metadata.

---

## 1. SchedulerMemoryBudget (Complete File)

**File:** `/Users/alexworland/omlx/omlx/cache/scheduler_memory_budget.py`
**Purpose:** Pre-computation memory check before allocating next prefill chunk

### Class: SchedulerMemoryBudget

#### Attributes
- `hard_limit_bytes: int` — Maximum allowed memory (hard limit)
- `soft_limit_bytes: int` — Soft limit (warning threshold before hard limit)
- `active_memory: int` — Currently active memory usage
- `byte_count: int` — Accumulated byte count
- `max_tokens: int` — Maximum tokens tracked

#### Methods

**`__init__(self, hard_limit_bytes: int, soft_limit_bytes: int) -> None`**
- Initializes memory budget with hard and soft limits
- Sets `active_memory = 0`, `byte_count = 0`, `max_tokens = 0`

**`estimate_tokens_budget(self) -> Tuple[int, int, int]`**
- **Returns:** `(tokens_to_soft, tokens_to_hard, max_tokens)`
- Estimates maximum tokens remaining before hitting soft/hard limits
- ⚠️ **Bug identified:** Line 32 sets `num_bytes` but never assigns it; hard-coded to `self.soft_limit_bytes`
- Returns remaining tokens as raw byte differences (logic appears incomplete)

**`estimate_memory_from_tokens(self, tokens: int, bytes_per_token: int) -> None`**
- Adds estimated memory cost from tokens
- Updates `byte_count += tokens * bytes_per_token`
- Updates `max_tokens += int(tokens)` if `bytes_per_token > 0`

**`budget_check(self) -> Tuple[int, int, bool, str]`**
- **Returns:** `(remaining_soft, remaining_hard, fits, message)`
- Calculates remaining soft/hard limits: `soft = soft_limit - byte_count`, `hard = hard_limit - byte_count`
- Decision logic:
  - If `byte_count > hard_limit`: Returns `(remaining_hard, remaining_soft, False, "too tight")`
  - Else if `byte_count <= soft_limit`: Returns `(remaining_soft, remaining_hard, True, "fits")`
  - Else: Returns `(remaining_soft, remaining_hard, False, "exceeds soft")`
- ⚠️ **Bug identified:** Line 62-64 has unreachable loop with undefined variables `i` and `bytes_per_token`

**`refine_limit(self, limit_bytes: int, limit_type: str) -> int`**
- **Returns:** Refined limit value
- Parameters: `limit_type` in `{'soft', 'hard', 'budget'}`
- Returns `limit_bytes + self.soft_limit_bytes` (soft), `limit_bytes + self.hard_limit_bytes` (hard), or `self.max_tokens` (budget)

---

## 2. Scheduler Memory Limits

**File:** `/Users/alexworland/omlx/omlx/scheduler.py` (41,654 tokens total)

### Class: Scheduler

#### Memory Limit Attributes (line 1031-1032)
- `_memory_limit_bytes: int = 0` — Soft memory limit (warnings only)
- `_memory_hard_limit_bytes: int = 0` — Hard memory limit (forces prefill stop)
- **Calculation:** `hard_limit = system_ram - 4GB` (conservative margin)

#### Memory Limit Propagation
```python
# Line 1547: Scheduler → BatchGenerator
bg._memory_limit_bytes = self._memory_limit_bytes
bg._memory_hard_limit_bytes = self._memory_hard_limit_bytes
```

### Class: _BoundarySnapshotBatchGenerator(BatchGenerator)

#### Memory Limit Attributes (lines 115-116)
- `_memory_limit_bytes: int = 0` — Soft limit (warning threshold)
- `_memory_hard_limit_bytes: int = 0` — Hard limit (enforcement threshold)

#### Memory Checking Locations

**Primary Prefill Path (lines 461-485)**
```
After processing prefill boundary:
  if self._memory_limit_bytes > 0:
    active = mx.get_active_memory()
    if active > self._memory_hard_limit_bytes:
      ✗ RuntimeError: Hard limit exceeded, stop prefill
    elif active > self._memory_limit_bytes:
      ⚠ Warning: Soft limit exceeded, monitor
```

**Secondary Prefill Path (lines 576-599)**
- Identical memory checking pattern as primary path
- Both paths check after boundary processing completes

#### Decision Points
1. **Soft limit exceeded:** Log warning, continue prefill (advisory only)
2. **Hard limit exceeded:** Raise RuntimeError, force stop prefill (enforcement)

### Method: Scheduler.step() → SchedulerOutput
- **Line 3263:** Main scheduling step entry point
- Flow: `_schedule_waiting()` → batch generation → memory checking

### Method: Scheduler._schedule_waiting() → List[Request]
- **Line 2609:** Moves requests from waiting to running queue
- Checks request homogeneity before scheduling

### Method: Scheduler._check_memory_pressure() → None
- **Line 3611:** Currently a no-op in paged SSD-only mode
- KV cache data on SSD, so GPU memory pressure is not applicable

---

## 3. PagedCacheManager (Paged KV Cache)

**File:** `/Users/alexworland/omlx/omlx/cache/paged_cache.py`

### Class: CacheBlock (line 93)

#### Attributes
- `block_id: int` — Unique block identifier
- `ref_count: int` — Reference count for shared blocks
- `block_hash: Optional[bytes]` — SHA256 hash for prefix caching
- `prev_free_block: Optional[CacheBlock]` — Doubly linked list (LRU)
- `next_free_block: Optional[CacheBlock]` — Doubly linked list (LRU)
- `is_null: bool` — Sentinel flag for null block
- `token_count: int` — Number of tokens in block
- `last_access: float` — Timestamp of last access for LRU ordering

#### Methods
- `is_full() -> bool` — Checks if block at capacity
- `is_shared() -> bool` — Returns `ref_count > 1`
- `reset_hash() -> None` — Clears block hash
- `touch() -> None` — Updates last access time for LRU tracking

### Class: FreeKVCacheBlockQueue (line 160)

**Purpose:** Doubly linked list for O(1) LRU operations

#### Methods
- `popleft() -> Optional[CacheBlock]` — Remove least-recently-used block from front
- `popleft_n(n: int) -> List[CacheBlock]` — Remove N LRU blocks atomically
- `remove(block: CacheBlock) -> None` — Remove specific block from middle (O(1))
- `append(block: CacheBlock) -> None` — Add block to end (most recently used)

### Class: PagedCacheManager (line 450)

#### Initialization Parameters
- `block_size: int` — Tokens per block
- `max_blocks: int` — Maximum block pool size
- `enable_caching: bool` — Prefix caching enable flag
- `model_name: str` — Model identifier for isolation
- `initial_blocks: int` — Initial block pool size (grows dynamically)

#### Attributes
- `blocks: Dict[int, CacheBlock]` — Block ID → CacheBlock mapping
- `free_block_queue: FreeKVCacheBlockQueue` — LRU-ordered free blocks
- `cached_block_hash_to_block: BlockHashToBlockMap` — Hash → Block (prefix cache)
- `request_tables: Dict[int, RequestTable]` — Per-request block tables
- `allocated_blocks: Set[CacheBlock]` — Currently allocated blocks
- `null_block: CacheBlock` — Sentinel for empty/null blocks

#### Block Allocation Methods

**`allocate_block(self) -> Optional[CacheBlock]`** (line 597)
- Allocates one block from free queue
- **Returns:** CacheBlock or None if allocation fails
- **Flow:**
  1. Pop block from free queue head
  2. If insufficient free blocks, trigger `_grow_blocks()`
  3. Increment ref_count, remove from free queue
  4. Update allocated_blocks set
- **Dynamic Growth:** If free queue below threshold, grows pool toward max_blocks

**`get_new_blocks(self, num_blocks: int) -> List[CacheBlock]`** (line 627)
- Allocates multiple blocks atomically
- **Returns:** List of CacheBlock objects
- **Raises:** ValueError if insufficient blocks available after growth attempt
- **Used for:** Batch allocation of prefill blocks

#### Block Freeing Methods

**`free_block(self, block_id: int) -> bool`** (line 695)
- Frees a single block
- **Flow:**
  1. Decrement ref_count
  2. If ref_count reaches 0, add block to free queue
  3. Remove from allocated_blocks set
- **Returns:** True if block fully freed, False if still referenced

**`free_blocks(self, blocks: Iterable[CacheBlock]) -> None`** (line 732)
- Batch free operation for multiple blocks
- Calls `free_block()` for each block

#### Reference Counting Methods

**`touch(blocks: List[CacheBlock]) -> None`** (line 765)
- Increments ref_count for shared block access
- Removes block from free queue if needed (moves to allocated)
- Used for: Copy-on-write, block reuse, prefix caching

#### Eviction Methods

**`evict_lru_blocks(self, num_blocks: int) -> int`** (line 1214)
- Evicts num_blocks from LRU (front of free queue)
- **Returns:** Number of blocks successfully evicted
- **Flow:**
  1. Pop N blocks from free queue (LRU-ordered)
  2. Clear block metadata
  3. Return to free queue for reallocation
- **Time Complexity:** O(N) where N = num_blocks

**`get_evictable_blocks(self, count: int) -> List[CacheBlock]`** (line 1370)
- Returns up to count LRU-ordered evictable blocks
- **Returns:** List of CacheBlock objects sorted by LRU order
- **Flow:**
  1. Iterate free queue in order (LRU-ordered)
  2. Return first count blocks
  3. If insufficient, iterate allocated_blocks sorted by last_access
- **Time Complexity:** O(count + allocated) worst case

**`evict_block_permanently(self, block_id: int) -> bool`** (line 1453)
- Removes block from hash cache and resets metadata
- **Flow:**
  1. Remove from cached_block_hash_to_block map
  2. Clear block_hash
  3. Return block to free queue
- **Used for:** Cleanup of expired/stale cached blocks

#### Statistics Methods
- `get_stats() -> Dict` — Returns allocation/eviction statistics
- `get_memory_usage() -> int` — Calculates total memory from blocks
- `reset_stats() -> None` — Clears performance counters
- `clear() -> None` — Resets entire cache state

### Class: BlockHashToBlockMap (line 376+)

**Purpose:** Hash-to-block mapping for prefix caching

#### Features
- Content-addressable storage via SHA256 hashes
- Enables prefix caching: reuse blocks with identical token sequences
- Thread-safe access for concurrent requests

---

## 4. PagedSSDCacheManager (SSD Block Storage)

**File:** `/Users/alexworland/omlx/omlx/cache/paged_ssd_cache.py`

### Async I/O Infrastructure

**`_compute_max_pending_writes() -> int`** (line 53)
- Computes write queue depth based on system memory
- **Scaling:** 512GB = 256, 32GB = 32, minimum 32
- Formula: `max(32, min(256, int(total_gb / 2)))`
- **Purpose:** Absorb burst saves from large requests (e.g., 64 blocks per 4096-token request)

**`_MAX_PENDING_WRITES`** (line 70)
- Global constant for write queue capacity
- Prevents queue saturation during prefill bursts

### Block Saving Methods

**`save_block(self, block_hash: bytes, cache_data: List[Any], token_count: int, model_name: str = "", layer_cache_types: Optional[List[str]] = None, layer_meta_states: Optional[List[Tuple]] = None) -> bool`** (line 905)

- **Purpose:** Save KV cache block to SSD storage (non-blocking)
- **Parameters:**
  - `block_hash: bytes` — Content hash for the block
  - `cache_data: List[Any]` — Per-layer data (keys, values) or CacheList sub-tensors
  - `token_count: int` — Number of tokens in block
  - `model_name: str` — Model name for cache isolation
  - `layer_cache_types: Optional[List[str]]` — Cache type names per layer (e.g., "KVCache", "CacheList")
  - `layer_meta_states: Optional[List[Tuple]]` — Reconstruction metadata per layer

- **Returns:** True if enqueued successfully, False on queue saturation/error

- **Key Flow:**
  1. Check if block already exists in index (hit)
  2. Check hot cache / pending writes buffer
  3. Enforce size limit if SSD path (not hot cache)
  4. Prepare safetensors arrays for background writing
  5. Enqueue write task to background worker thread
  6. Block immediately available for reads via pending-writes buffer (write-back cache)

- **Non-blocking:** Data enqueued immediately; background thread performs actual I/O
- **Queue Saturation:** Returns False if write queue is full (`_write_queue.full()`)
- **Size Management:** `_enforce_size_limit_for_new_block()` ensures LRU eviction before saving new blocks

### Background Writer Thread

**`_writer_loop(self) -> None`** (line 830)
- Background thread that processes enqueued write tasks
- **Flow:**
  1. Dequeue write task from queue
  2. Serialize to safetensors format (GPU/Metal-free, numpy-based)
  3. Write to disk atomically
  4. Update index on success
  5. Update stats (writes, failures)

**`_enqueue_ssd_write(self, block_hash: bytes, entry: Dict) -> bool`** (line 639)
- Enqueues write task to background worker
- **Parameters:** `block_hash`, entry dict with cache data and metadata
- **Returns:** False if queue is full
- **Time Complexity:** O(1) enqueue operation

### Safetensors Serialization

**`_write_safetensors_no_mx(filename: str, arrays: Dict[str, np.ndarray], metadata: Dict[str, str]) -> bool`** (line 179)

- **Purpose:** Write safetensors file without MLX/Metal API (background thread safe)
- **Parameters:**
  - `filename: str` — Output file path
  - `arrays: Dict[str, np.ndarray]` — Numpy arrays to serialize
  - `metadata: Dict[str, str]` — JSON metadata (layer info, shapes, etc.)
- **Returns:** True on success, False on write error

- **Key Features:**
  - GPU/Metal-free (numpy-only): Allows background thread execution without Metal API deadlock
  - Handles bfloat16 dtype mapping (safetensors "BF16" format)
  - Atomic file writes (temp file → rename pattern)
  - Thread-safe serialization

### Eviction for Size Management

**`evict_until_size(self, target_size: int) -> List[PagedSSDBlockMetadata]`** (line 443)
- Evicts blocks until cache size <= target_size
- **Returns:** List of evicted block metadata
- **Algorithm:** LRU-based eviction using index

**`evict(self, key: Any) -> bool`** (line 1787)
- Evicts specific block by content hash
- **Returns:** True if evicted, False if not found

---

## 5. MemoryMonitor (Memory Estimation)

**File:** `/Users/alexworland/omlx/omlx/memory_monitor.py`

### Class: MemoryInfo (line 39)

**Dataclass** representing current GPU memory state

#### Attributes
- `total_bytes: int` — Total available GPU memory
- `used_bytes: int` — Currently used memory (estimated)
- `available_bytes: int` — Available memory
- `utilization: float` — Memory utilization ratio (0.0 to 1.0)

### Class: MemoryMonitor (line 56)

**Purpose:** Memory monitor for paged SSD-based KV cache with memory estimation utilities

#### Initialization Parameters
- `max_kv_cache_memory: int` — Maximum memory for KV cache (required, must be > 0)
- `check_interval: float = 1.0` — Minimum seconds between memory checks (throttling)

#### Attributes
- `_max_kv_cache_memory: int` — Maximum KV cache memory limit
- `_check_interval: float` — Throttle interval for memory checks
- `_max_memory: int` — System memory limit from `get_max_working_set_bytes()`
- `_baseline_memory: int = 0` — Memory after model load (model weights)
- `_num_layers: Optional[int]` — Transformer layer count
- `_num_kv_heads: Optional[int]` — KV attention head count
- `_head_dim: Optional[int]` — Dimension per attention head
- `_dtype_size: int = 2` — Bytes per element (2 for float16, 4 for float32)
- `_paged_cache_manager: Optional[PagedCacheManager]` — Connected cache manager
- `_block_size: int = 256` — Default tokens per block
- `_running_requests: int = 0` — Currently running requests
- `_waiting_requests: int = 0` — Waiting request count

#### Memory Estimation Methods

**`set_baseline_memory(self) -> None`** (line 144)
- Captures baseline memory after model load using `mx.get_active_memory()`
- **Purpose:** KV cache memory = active_memory - baseline_memory
- Allows accurate detection of memory pressure from KV cache growth alone

**`set_model_info(self, num_layers: int, num_kv_heads: int, head_dim: int, dtype_size: int = 2) -> None`** (line 264)
- Sets model parameters for memory estimation
- Logs estimated memory per 64-token block after setting

**`estimate_block_memory(self, block_size: int, num_layers: Optional[int] = None, num_kv_heads: Optional[int] = None, head_dim: Optional[int] = None, dtype_size: Optional[int] = None) -> int`** (line 294)

- **Purpose:** Estimate memory usage for a KV cache block
- **Parameters:**
  - `block_size: int` — Number of tokens in block
  - `num_layers: Optional[int]` — Override stored value (default: 32 for ~7B model)
  - `num_kv_heads: Optional[int]` — Override stored value (default: 8)
  - `head_dim: Optional[int]` — Override stored value (default: 128)
  - `dtype_size: Optional[int]` — Override stored value (default: 2 for float16)

- **Returns:** Estimated memory in bytes for one block

- **Formula:**
  ```
  memory_per_layer = block_size × kv_heads × head_dim × dtype_size × 2  # ×2 for keys+values
  total_memory = memory_per_layer × num_layers
  ```

- **Example:** 64-token block, 7B model (32 layers, 8 KV heads, 128 head_dim, float16)
  ```
  per_layer = 64 × 8 × 128 × 2 × 2 = 262,144 bytes
  total = 262,144 × 32 = 8,388,608 bytes (8 MB per block)
  ```

**`estimate_blocks_to_free(self, bytes_to_free: int, block_size: int) -> int`** (line 327)

- **Purpose:** Estimate number of blocks to evict to free given bytes
- **Parameters:**
  - `bytes_to_free: int` — Target bytes to free
  - `block_size: int` — Tokens per block
- **Returns:** Number of blocks to evict (rounds up, minimum 1)
- **Formula:** `num_blocks = ceil(bytes_to_free / estimate_block_memory(block_size))`

#### Memory Status Methods

**`get_memory_info(self) -> MemoryInfo`** (line 208)
- Returns current memory state
- **Throttling:** Returns cached result if called within `check_interval`
- **Flow:**
  1. Check throttle timer
  2. Get current memory usage (0 in SSD-only mode)
  3. Calculate utilization ratio
  4. Cache result with timestamp

**`is_under_pressure(self) -> bool`** (line 239)
- Check if memory pressure exists
- **Returns:** False in paged SSD-only mode (KV cache on SSD, not GPU memory)

**`bytes_to_free(self) -> int`** (line 251)
- Calculate bytes needed to free
- **Returns:** 0 in paged SSD-only mode

#### Statistics and Configuration

**`set_paged_cache_manager(self, manager: PagedCacheManager, block_size: int = 64) -> None`** (line 127)
- Connects PagedCacheManager for memory monitoring

**`set_request_stats(self, running: int, waiting: int) -> None`** (line 168)
- Updates request stats for logging (used in diagnostics)

**`get_stats(self) -> dict`** (line 356)
- **Returns:** Dictionary with memory statistics:
  - `total_bytes`, `used_bytes`, `available_bytes`, `utilization`
  - `max_kv_cache_memory`
  - Formatted versions of above (e.g., "8.5GB")

**`__repr__(self) -> str`** (line 375)
- Returns string representation: `MemoryMonitor(max_kv_cache={formatted}, used={formatted})`

---

## 6. Memory Flow Summary: Request → Scheduler → Batch Generator → SSD

### Phase 1: Request Arrival
1. Request enters scheduler queue (waiting state)
2. Scheduler calls `_schedule_waiting()` to move homogeneous requests to running queue

### Phase 2: Batch Generation
1. Scheduler creates BatchGenerator instance
2. Propagates memory limits: `bg._memory_limit_bytes = self._memory_limit_bytes`
3. Calls `generate_one()` to process next batch

### Phase 3: Prefill Processing
1. BatchGenerator processes prefill tokens for all requests in batch
2. After boundary processing completes:
   - Gets active memory: `active = mx.get_active_memory()`
   - Compares to hard limit: if `active > _memory_hard_limit_bytes` → RuntimeError (stop prefill)
   - Compares to soft limit: if `active > _memory_limit_bytes` → Warning (log only)

### Phase 4: Block Allocation & SSD Write
1. PagedCacheManager allocates blocks for prefill KV cache
2. Blocks enqueued for SSD write via PagedSSDCacheManager
3. Background thread serializes blocks to safetensors format
4. Blocks written to SSD in hash-based directory structure
5. Index updated for later retrieval

### Phase 5: Memory Reclamation
1. If memory pressure detected, PagedCacheManager evicts LRU blocks
2. `evict_lru_blocks()` removes blocks from free queue
3. Blocks are written to SSD before free space reclaimed
4. Freed block slots reused for new requests

---

## 7. Key Design Patterns

### Soft/Hard Limit Strategy
- **Soft limit:** Advisory only (warning messages), prefill continues
- **Hard limit:** Enforcement (RuntimeError), prefill stops immediately
- **Rationale:** Gradual degradation vs. catastrophic failure; allows system to recover

### Non-blocking SSD Writes
- Save requests enqueued to background thread
- Block immediately available for reads via pending-writes buffer
- Prevents I/O latency from blocking request scheduling

### Reference Counting & Copy-on-Write
- Blocks have ref_count for sharing across requests
- `touch()` increments ref_count, moves block to allocated set
- `free_block()` decrements, returns to free queue when ref_count == 0
- Enables efficient prefix caching and block reuse

### O(1) LRU Operations
- FreeKVCacheBlockQueue uses doubly linked list
- Append (MRU): O(1)
- Pop from front (LRU): O(1)
- Remove from middle: O(1) with direct reference
- Enables efficient eviction even with millions of blocks

### Paged SSD-Only Mode
- KV cache stored entirely on paged SSD
- GPU memory holds only: model weights + block metadata
- `MemoryMonitor._get_current_memory_usage()` returns 0 for KV cache
- `is_under_pressure()` always returns False (no GPU memory pressure from cache)
- SSD I/O latency managed via lazy restore and background writes

---

## 8. Known Issues & Incomplete Features

### SchedulerMemoryBudget Bugs
1. **Line 32:** `num_bytes` assigned but never used in `estimate_tokens_budget()`
2. **Lines 62-64:** Unreachable loop with undefined variables `i` and `bytes_per_token`
3. **Overall:** Logic appears incomplete; unclear how tokens are converted to memory estimates

### Memory Pressure Detection
1. **Scheduler._check_memory_pressure()** (line 3611) is a no-op
2. No active memory pressure detection in paged SSD-only mode
3. Eviction is passive (only when new block allocation needed)

### Edge Cases
1. **Queue saturation:** SSD write queue can fill if prefill bursts exceed background thread throughput
2. **Missing dtype handling:** No verification that dtype_size matches actual model dtype
3. **Block allocation failures:** No retry logic if initial growth attempt insufficient

---

## 9. Testing Recommendations

1. **Memory limit enforcement:** Verify RuntimeError raised when hard limit exceeded
2. **LRU correctness:** Verify eviction order matches access order
3. **Prefix caching:** Verify hash collisions handled correctly
4. **Block sharing:** Verify ref_count incremented/decremented correctly
5. **SSD I/O:** Verify blocks survive background thread serialization/deserialization
6. **Paged SSD mode:** Verify no GPU memory used for KV cache storage

---

## 10. Code Structure Reference

| Component | File | Key Class | Purpose |
|-----------|------|-----------|---------|
| Memory Budget | `scheduler_memory_budget.py` | SchedulerMemoryBudget | Pre-allocation checking |
| Scheduler | `scheduler.py` | Scheduler, _BoundarySnapshotBatchGenerator | Request scheduling, memory gating |
| Paged Cache | `paged_cache.py` | PagedCacheManager, CacheBlock, FreeKVCacheBlockQueue | Block allocation/eviction |
| SSD Cache | `paged_ssd_cache.py` | PagedSSDCacheManager | Block persistence, safetensors I/O |
| Memory Monitor | `memory_monitor.py` | MemoryMonitor | Memory estimation, status reporting |

---

End of Exploration Document
