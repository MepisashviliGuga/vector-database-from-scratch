# Source papers

Numbered in the order the project consumes them. Every algorithmic claim in this
repo should trace back to a section of one of these.

| # | File | Paper | Authors | Venue | Used by |
|---|------|-------|---------|-------|---------|
| 01 | `01-vertiorizon-lsm-growth.pdf` | How to Grow an LSM-tree? Towards Bridging the Gap Between Theory and Practice | Mo, Luo, Idreos | SIGMOD 2025 (PACMMOD 3(3), Art. 173) · [arXiv:2504.17178](https://arxiv.org/abs/2504.17178) | Phase 1 — **Vertiorizon** growth scheme (tree shape) |
| 02 | `02-ecotune-compaction-policy.pdf` | Rethinking The Compaction Policies in LSM-trees | Wang, Qiu, Yuan, Zhang | SIGMOD 2025 (PACMMOD 3(3), Art. 207) | Phase 2 — **EcoTune** DP compaction scheduler (timing/aggressiveness) |
| 03 | `03-rabitq-extended-quantization.pdf` | Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for ANN Search | Gao, Gou, Xu, Yang, Long, Wong | SIGMOD 2025 · [arXiv:2409.09913](https://arxiv.org/abs/2409.09913) | Phase 5 — **extended RaBitQ** quantizer |
| 04 | `04-symphonyqg-quantization-graph.pdf` | SymphonyQG: Towards Symphonious Integration of Quantization and Graph for ANN Search | Gou, Gao, Xu, Long | SIGMOD 2025 · [arXiv:2411.12229](https://arxiv.org/abs/2411.12229) | Phase 5 — **primary ANN index** |
| 05 | `05-deg-hybrid-vector-search.pdf` | DEG: Efficient Hybrid Vector Search Using the Dynamic Edge Navigation Graph | Yin, Gao, Balsebre, Cong, Long | SIGMOD 2025 · [arXiv:2502.07343](https://arxiv.org/abs/2502.07343) | Phase 7 stretch — hybrid/filtered search |
| 06 | `06-ruskey-rl-lsm.pdf` | Learning to Optimize LSM-trees: Towards A Reinforcement Learning based Key-Value Store for Dynamic Workloads | Mo, Chen, Luo, Shan | PACMMOD 1(4), 2023 (SIGMOD 2024) · [arXiv:2308.07013](https://arxiv.org/abs/2308.07013) | Phase 7 stretch — RL vs. EcoTune's DP |

Papers 01 and 06 share a lab (Mo, Luo — NTU Singapore); 03, 04 and 05 share
another (Gao, Gou, Long — NTU Singapore). Papers 03/04 in particular are designed
to compose, which is why SymphonyQG is the target rather than plain HNSW.

## Reference implementation

The authors of papers 03 and 04 publish a C++ library at
<https://github.com/VectorDB-NTU/RaBitQ-Library>.

**How this project uses it: as an oracle, never as a dependency.** Our RaBitQ and
SymphonyQG implementations are written from the papers. The reference library is
used only to check our output — comparing quantized codes, estimated distances,
and recall@k on identical inputs, so a recall gap can be attributed to a bug in
our code rather than to a misreading of the paper. Anything it is used for must
be labelled as a cross-check in the writeup, and no reported result may come from
running it in place of ours.
