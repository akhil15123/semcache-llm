# Roadmap

Small, shippable steps. Each one should land with tests.

- [x] Core cache: cosine lookup, threshold, namespaces
- [x] LRU eviction and TTL expiry
- [x] Stats (hit rate, tokens saved, evictions)
- [x] `get_or_call` and `@cached` decorator
- [x] Offline benchmark with threshold sweep
- [x] Persistence: save/load cache to disk (`.npz` + JSON metadata)
- [x] Async support (`aget_or_call`, `@acached`) with in-flight request coalescing
- [ ] Labelled paraphrase eval set; report precision/recall per threshold
- [ ] Adaptive threshold: tune per namespace from feedback (wrong-hit reports)
- [ ] Approximate search backend (FAISS / hnswlib) for >100k entries
- [ ] Streaming responses: cache the assembled stream, replay as chunks
- [ ] CLI: `semcache stats`, `semcache warm --file prompts.jsonl`
- [ ] Redis backend for multi-process deployments
