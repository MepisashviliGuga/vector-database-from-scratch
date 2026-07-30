//! DEG: the Dynamic Edge Navigation Graph — paper 05, Phase 7.
//!
//! # The problem
//!
//! A Hybrid Vector Query (§3.1) gives every object two feature vectors — say an
//! image embedding `o.e` and a text embedding `o.s` — and lets each *query* pick
//! how to weight them:
//!
//! ```text
//! Dist(q,o) = α·δe(q,o) + (1−α)·δs(q,o),   α ∈ [0,1] chosen per query
//! ```
//!
//! The two vectors are not the hard part. The hard part is that α arrives *with
//! the query*, and every graph ANN index prunes its edges using distances assumed
//! fixed at build time. Fix α at build time and you get an index that is strong
//! near that α and degrades away from it — §3.3 measures exactly this for both
//! prior approaches, and neither is a poor implementation. They are structurally
//! committed to a weight they do not yet know.
//!
//! # What makes it tractable
//!
//! `Dist` is affine in α: for a fixed pair of objects it is the straight line
//! `δs + α(δe − δs)`. So every distance *comparison* is a comparison of two
//! lines, flipping at one crossing point, and the question "for which α would
//! the RNG rule prune this edge?" has a closed-form answer. [`pruning`] is that
//! solution; [`interval`] is the set arithmetic it produces.
//!
//! # Scope
//!
//! Two modalities only. This is the authors' own limit, not a simplification
//! here: §4.3 notes the strategy "does not apply to multi-vector queries with
//! more than two vectors, as the active range becomes a hyperplane", and leaves
//! that as future work.

pub mod gps;
pub mod graph;
pub mod hybrid;
pub mod interval;
pub mod pareto;
pub mod pruning;
pub mod select;

pub use gps::gps;
pub use graph::{CandidateSource, DegConfig, DegIndex, EntryPolicy, PruningPolicy};
pub use hybrid::HybridSet;
pub use interval::AlphaSet;
pub use pareto::{dominates, frontier_layers, ParetoPoint};
pub use pruning::{prunes_at, pruned_by, solve, HybridDistance};
pub use select::{select_neighbours, DegEdge};
