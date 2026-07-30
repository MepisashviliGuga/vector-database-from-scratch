//! The vector database: durable storage plus approximate search.
//!
//! ```text
//!   insert(id, vector, metadata)
//!        │
//!        ├──► LSM-tree ......... full-precision vector + metadata, durable
//!        └──► IVF index ........ quantized code, in memory, approximate
//!
//!   search(query, k)
//!        │
//!        ├──► IVF index ........ cheap pass over codes → candidate ids
//!        └──► LSM-tree ......... fetch full vectors, re-rank exactly → top k
//! ```
//!
//! # Why both layers earn their place
//!
//! The quantized index is fast and small but *lossy* — measured at 95.8%
//! recall@10 on SIFT1M by itself. The storage engine is exact but scanning it is
//! the brute-force cost the index exists to avoid.
//!
//! Composing them recovers most of the loss: the index proposes a candidate set
//! larger than `k`, and those candidates are re-ranked against **full-precision
//! vectors read back from storage**. Ranking errors inside the candidate set
//! disappear entirely; only candidates the index failed to *propose* are lost.
//! So the recall ceiling is set by the index's ability to surface the right
//! candidates, not by its ability to order them — a much easier job.
//!
//! This is also where the storage engine stops being a separate project. Exact
//! re-ranking needs a durable, low-latency point lookup for arbitrary ids, which
//! is precisely what the LSM-tree provides.
//!
//! # Consistency, stated honestly
//!
//! **Storage is the source of truth; the index is a hint.** That asymmetry has
//! consequences worth being explicit about:
//!
//! - **Updates.** Overwriting a vector rewrites the stored record immediately,
//!   but the index keeps the old code until a rebuild. Re-ranking reads the
//!   *new* vector, so a returned distance is never stale — but the candidate
//!   set was chosen using the old one, so an updated vector may be proposed when
//!   it no longer deserves to be, or missed when it now does.
//! - **Deletes.** A tombstone lands in storage at once and the id joins a
//!   deleted set that search filters. The code stays in the index until a
//!   rebuild, occupying a candidate slot.
//! - **Neither is corruption.** A search never returns a deleted id, and never
//!   returns a distance computed from stale data. What degrades is *recall*, and
//!   [`VectorStore::rebuild_index`] restores it.

use std::collections::HashSet;
use std::io;
use std::path::Path;

use crate::ann::{squared_l2, IvfConfig, IvfIndex, Neighbor};
use crate::storage::{LsmConfig, LsmTree};

/// How the database is configured.
#[derive(Debug, Clone)]
pub struct VectorStoreConfig {
    pub dimension: usize,
    pub storage: LsmConfig,
    pub index: IvfConfig,
    /// Vectors buffered before the index is first trained.
    ///
    /// k-means needs data to cluster, so the index cannot exist until some has
    /// arrived. Until then searches fall back to an exact scan, which is correct
    /// and slow rather than fast and wrong.
    pub training_threshold: usize,
}

impl VectorStoreConfig {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            storage: LsmConfig::default(),
            index: IvfConfig::default(),
            training_threshold: 1_000,
        }
    }
}

/// Search-time knobs.
#[derive(Debug, Clone, Copy)]
pub struct SearchParams {
    /// Inverted lists probed. Trades recall against work.
    pub nprobe: usize,
    /// Candidates the index proposes before exact re-ranking.
    ///
    /// Larger means more storage reads and better recall. `k` disables the
    /// benefit of re-ranking entirely, since there is nothing to re-order.
    pub rerank_candidates: usize,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            nprobe: 32,
            rerank_candidates: 100,
        }
    }
}

/// A stored record: the full-precision vector and its metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct Record {
    pub vector: Vec<f32>,
    pub metadata: Vec<u8>,
}

/// Durable vectors with approximate search over them.
#[derive(Debug)]
pub struct VectorStore {
    storage: LsmTree,
    index: Option<IvfIndex>,
    config: VectorStoreConfig,
    /// Ids present in the index but deleted since it was built.
    deleted: HashSet<u32>,
    /// Every live id, so an untrained store can scan and a rebuild can gather.
    live: Vec<u32>,
}

impl VectorStore {
    /// Open or create a store in `dir`.
    ///
    /// Rebuilds the index from storage if records are already there — the index
    /// is derived state and is not persisted, so a restart pays to reconstruct
    /// it. That is a deliberate simplification: persisting it would mean a
    /// second durability format to keep consistent with the first, and the
    /// rebuild is bounded by data already on disk.
    pub fn open(dir: impl AsRef<Path>, config: VectorStoreConfig) -> io::Result<Self> {
        assert!(config.dimension > 0, "vectors need at least one dimension");
        let storage = LsmTree::open(dir, config.storage.clone())?;

        let mut store = Self {
            storage,
            index: None,
            config,
            deleted: HashSet::new(),
            live: Vec::new(),
        };

        // Recover the id set from storage, which is the source of truth.
        let ids: Vec<u32> = store
            .storage
            .iter()
            .map(|entry| entry.map(|(key, _)| decode_id(&key)))
            .collect::<io::Result<Vec<_>>>()?;
        store.live = ids;

        if store.live.len() >= store.config.training_threshold {
            store.rebuild_index()?;
        }
        Ok(store)
    }

    pub fn dimension(&self) -> usize {
        self.config.dimension
    }

    /// Live records.
    pub fn len(&self) -> usize {
        self.live.len()
    }

    pub fn is_empty(&self) -> bool {
        self.live.is_empty()
    }

    /// Whether an index exists yet. Until it does, searches scan exactly.
    pub fn is_indexed(&self) -> bool {
        self.index.is_some()
    }

    /// Ids in the index that have since been deleted, and so waste a candidate
    /// slot until the next rebuild.
    pub fn stale_deletes(&self) -> usize {
        self.deleted.len()
    }

    /// Insert or overwrite a vector.
    ///
    /// # Panics
    ///
    /// If `vector` is the wrong length.
    pub fn insert(&mut self, id: u32, vector: &[f32], metadata: &[u8]) -> io::Result<()> {
        assert_eq!(
            vector.len(),
            self.config.dimension,
            "vector has the wrong length"
        );

        // Storage first: it is the source of truth, so a crash between the two
        // writes must leave the durable copy present and the index behind, never
        // the reverse.
        let existed = self.storage.get(&encode_id(id))?.is_some();
        self.storage
            .put(encode_id(id), encode_record(vector, metadata))?;

        if !existed {
            self.live.push(id);
        }
        // Re-inserting a deleted id makes it live again.
        self.deleted.remove(&id);

        if let Some(index) = &mut self.index {
            // An overwrite leaves the old code in place; re-ranking still reads
            // the new vector, so results stay correct. See the module docs.
            if !existed {
                index.insert(id, vector);
            }
        } else if self.live.len() >= self.config.training_threshold {
            self.rebuild_index()?;
        }
        Ok(())
    }

    /// Read a record back.
    pub fn get(&self, id: u32) -> io::Result<Option<Record>> {
        Ok(self
            .storage
            .get(&encode_id(id))?
            .and_then(|bytes| decode_record(&bytes, self.config.dimension)))
    }

    /// Delete a vector.
    ///
    /// The tombstone is durable at once; the index keeps the code until a
    /// rebuild, and search filters it out in the meantime.
    pub fn delete(&mut self, id: u32) -> io::Result<bool> {
        if self.storage.get(&encode_id(id))?.is_none() {
            return Ok(false);
        }
        self.storage.delete(encode_id(id))?;
        self.live.retain(|&live| live != id);
        if self.index.is_some() {
            self.deleted.insert(id);
        }
        Ok(true)
    }

    /// The `k` nearest stored vectors, by exact distance.
    ///
    /// Distances are **exact**: candidates come from the quantized index but
    /// every returned result was re-ranked against its full-precision vector
    /// read back from storage.
    ///
    /// # Panics
    ///
    /// If `query` is the wrong length.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        params: SearchParams,
    ) -> io::Result<Vec<Neighbor>> {
        assert_eq!(
            query.len(),
            self.config.dimension,
            "query has the wrong length"
        );
        if k == 0 || self.live.is_empty() {
            return Ok(Vec::new());
        }

        let candidates: Vec<u32> = match &self.index {
            Some(index) => index
                .search(query, params.rerank_candidates.max(k), params.nprobe)
                .into_iter()
                .map(|neighbour| neighbour.id as u32)
                // A deleted id is still in the index until a rebuild.
                .filter(|id| !self.deleted.contains(id))
                .collect(),
            // No index yet: an exact scan is correct, and correctness before
            // enough data has arrived is worth more than speed.
            None => self.live.clone(),
        };

        let mut scored = Vec::with_capacity(candidates.len());
        for id in candidates {
            // The re-ranking read. This is why the storage engine needs a fast
            // point lookup for arbitrary ids.
            let Some(record) = self.get(id)? else {
                // Deleted between the index pass and this read.
                continue;
            };
            scored.push(Neighbor {
                id: id as u64,
                distance: squared_l2(query, &record.vector),
            });
        }

        scored.sort_unstable();
        scored.truncate(k);
        Ok(scored)
    }

    /// Rebuild the index from what storage currently holds.
    ///
    /// Retrains centroids, so it also clears the drift that accumulates as
    /// inserts land against centroids fitted to older data, and reclaims the
    /// candidate slots deleted ids were occupying.
    pub fn rebuild_index(&mut self) -> io::Result<()> {
        let mut ids = Vec::with_capacity(self.live.len());
        let mut data = Vec::with_capacity(self.live.len() * self.config.dimension);

        for entry in self.storage.iter() {
            let (key, value) = entry?;
            let Some(record) = decode_record(&value, self.config.dimension) else {
                continue;
            };
            ids.push(decode_id(&key));
            data.extend_from_slice(&record.vector);
        }

        self.live = ids.clone();
        self.deleted.clear();

        if ids.is_empty() {
            self.index = None;
            return Ok(());
        }

        // Train centroids, then insert under the caller's real ids. `build`
        // would assign ids by position, which mislabels everything the moment
        // ids are sparse — as they are after any delete.
        let mut index = IvfIndex::train_only(&data, self.config.dimension, self.config.index);
        for (position, &id) in ids.iter().enumerate() {
            let start = position * self.config.dimension;
            index.insert(id, &data[start..start + self.config.dimension]);
        }

        self.index = Some(index);
        Ok(())
    }

    /// Flush buffered writes to disk.
    pub fn sync(&mut self) -> io::Result<()> {
        self.storage.flush()?;
        self.storage.sync()
    }

    /// Bytes the index holds in memory, against the durable bytes on disk.
    pub fn footprint(&self) -> (usize, u64) {
        let index_bytes = self.index.as_ref().map_or(0, IvfIndex::packed_bytes);
        (index_bytes, self.storage.stats().disk_bytes)
    }
}

/// Keys are big-endian so byte order matches numeric order, which keeps a range
/// scan over ids meaningful.
fn encode_id(id: u32) -> Vec<u8> {
    id.to_be_bytes().to_vec()
}

fn decode_id(key: &[u8]) -> u32 {
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(&key[..4.min(key.len())]);
    u32::from_be_bytes(bytes)
}

/// `[dimension u32][vector f32…][metadata…]`.
///
/// The dimension is stored rather than assumed so a record decodes without
/// reference to configuration, and a mismatch is detectable instead of being
/// read as garbage.
fn encode_record(vector: &[f32], metadata: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(4 + vector.len() * 4 + metadata.len());
    bytes.extend_from_slice(&(vector.len() as u32).to_le_bytes());
    for value in vector {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes.extend_from_slice(metadata);
    bytes
}

fn decode_record(bytes: &[u8], expected_dimension: usize) -> Option<Record> {
    if bytes.len() < 4 {
        return None;
    }
    let dimension = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;
    if dimension != expected_dimension || bytes.len() < 4 + dimension * 4 {
        return None;
    }

    let vector = bytes[4..4 + dimension * 4]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("4 bytes")))
        .collect();
    Some(Record {
        vector,
        metadata: bytes[4 + dimension * 4..].to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::BenchDir;
    use crate::storage::{GrowthKind, MergeKind, SyncPolicy};
    use crate::workload::Rng;

    fn config(dimension: usize) -> VectorStoreConfig {
        VectorStoreConfig {
            dimension,
            storage: LsmConfig {
                memtable_threshold_bytes: 64 * 1024,
                sync_policy: SyncPolicy::Manual,
                target_file_size_bytes: 256 * 1024,
                growth: GrowthKind::Vertical {
                    buffer_bytes: 64 * 1024,
                    size_ratio: 4,
                },
                merge: MergeKind::Leveling,
                ..Default::default()
            },
            index: IvfConfig {
                clusters: 16,
                bits: 5,
                kmeans_iterations: 15,
                training_sample: None,
                seed: 7,
            },
            training_threshold: 200,
        }
    }

    fn synthetic(count: usize, dimension: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = Rng::new(seed);
        (0..count)
            .map(|_| {
                (0..dimension)
                    .map(|_| rng.next_f64() as f32 * 2.0 - 1.0)
                    .collect()
            })
            .collect()
    }

    fn populate(store: &mut VectorStore, vectors: &[Vec<f32>]) -> io::Result<()> {
        for (id, vector) in vectors.iter().enumerate() {
            store.insert(id as u32, vector, format!("meta-{id}").as_bytes())?;
        }
        Ok(())
    }

    fn exact(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<u64> {
        let mut scored: Vec<(f32, u64)> = vectors
            .iter()
            .enumerate()
            .map(|(id, v)| (squared_l2(v, query), id as u64))
            .collect();
        scored.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        scored.into_iter().take(k).map(|(_, id)| id).collect()
    }

    // -----------------------------------------------------------------
    // The round trip
    // -----------------------------------------------------------------

    #[test]
    fn records_round_trip_through_storage() -> io::Result<()> {
        let dir = BenchDir::new("engine-roundtrip")?;
        let mut store = VectorStore::open(dir.path(), config(8))?;

        let vector = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        store.insert(42, &vector, b"hello")?;

        let record = store.get(42)?.expect("record");
        assert_eq!(record.vector, vector);
        assert_eq!(record.metadata, b"hello");
        assert_eq!(store.get(43)?, None);
        Ok(())
    }

    #[test]
    fn record_encoding_rejects_a_dimension_mismatch() {
        let encoded = encode_record(&[1.0, 2.0, 3.0], b"m");
        assert!(decode_record(&encoded, 3).is_some());
        assert_eq!(
            decode_record(&encoded, 4),
            None,
            "a record must not decode against the wrong dimension"
        );
        assert_eq!(decode_record(&[1, 2], 3), None, "truncated");
    }

    #[test]
    fn ids_encode_in_numeric_order() {
        for pair in [(0u32, 1u32), (255, 256), (65_535, 65_536), (1, u32::MAX)] {
            assert!(encode_id(pair.0) < encode_id(pair.1));
            assert_eq!(decode_id(&encode_id(pair.0)), pair.0);
        }
    }

    // -----------------------------------------------------------------
    // The integration claim
    // -----------------------------------------------------------------

    /// **The reason both layers exist.** Re-ranking against full-precision
    /// vectors read from storage must beat the quantized index alone, because
    /// ranking errors inside the candidate set disappear entirely.
    #[test]
    fn re_ranking_beats_the_index_alone() -> io::Result<()> {
        let dimension = 32;
        let dir = BenchDir::new("engine-rerank")?;
        let mut store = VectorStore::open(dir.path(), config(dimension))?;

        let vectors = synthetic(2000, dimension, 11);
        populate(&mut store, &vectors)?;
        assert!(store.is_indexed(), "the index should have trained");

        let queries = synthetic(30, dimension, 13);
        let mut with_rerank = 0.0;
        let mut index_only = 0.0;

        for query in &queries {
            let truth: std::collections::HashSet<u64> =
                exact(&vectors, query, 10).into_iter().collect();

            let reranked = store.search(
                query,
                10,
                SearchParams {
                    nprobe: 8,
                    rerank_candidates: 200,
                },
            )?;
            with_rerank += reranked.iter().filter(|n| truth.contains(&n.id)).count() as f64 / 10.0;

            // The same candidate pass, but taking the index's own ordering.
            let raw = store.index.as_ref().expect("index").search(query, 10, 8);
            index_only += raw.iter().filter(|n| truth.contains(&n.id)).count() as f64 / 10.0;
        }

        let with_rerank = with_rerank / queries.len() as f64;
        let index_only = index_only / queries.len() as f64;
        assert!(
            with_rerank > index_only,
            "re-ranking gave {with_rerank:.4}, no better than the index's own \
             {index_only:.4}"
        );
        Ok(())
    }

    /// Returned distances must be exact, computed from stored vectors.
    #[test]
    fn returned_distances_are_exact() -> io::Result<()> {
        let dimension = 16;
        let dir = BenchDir::new("engine-exact")?;
        let mut store = VectorStore::open(dir.path(), config(dimension))?;

        let vectors = synthetic(600, dimension, 17);
        populate(&mut store, &vectors)?;

        let query = &synthetic(1, dimension, 19)[0];
        for found in store.search(query, 10, SearchParams::default())? {
            let truth = squared_l2(&vectors[found.id as usize], query);
            assert!(
                (found.distance - truth).abs() < 1e-4,
                "id {} reported {} against a true {truth}",
                found.id,
                found.distance
            );
        }
        Ok(())
    }

    /// Before enough data exists to cluster, search must still be correct — an
    /// exact scan is the right fallback.
    #[test]
    fn an_untrained_store_searches_exactly() -> io::Result<()> {
        let dimension = 8;
        let dir = BenchDir::new("engine-untrained")?;
        let mut store = VectorStore::open(dir.path(), config(dimension))?;

        let vectors = synthetic(50, dimension, 23);
        populate(&mut store, &vectors)?;
        assert!(!store.is_indexed(), "50 vectors is below the threshold");

        let query = &synthetic(1, dimension, 29)[0];
        let found: Vec<u64> = store
            .search(query, 5, SearchParams::default())?
            .into_iter()
            .map(|n| n.id)
            .collect();
        assert_eq!(
            found,
            exact(&vectors, query, 5),
            "the fallback must be exact"
        );
        Ok(())
    }

    // -----------------------------------------------------------------
    // Consistency under mutation
    // -----------------------------------------------------------------

    /// A deleted vector must never come back, even though its code survives in
    /// the index until a rebuild.
    #[test]
    fn deleted_vectors_never_surface() -> io::Result<()> {
        let dimension = 16;
        let dir = BenchDir::new("engine-delete")?;
        let mut store = VectorStore::open(dir.path(), config(dimension))?;

        let vectors = synthetic(800, dimension, 31);
        populate(&mut store, &vectors)?;

        // Delete the true nearest neighbours of a query, then ask for them.
        let query = &synthetic(1, dimension, 37)[0];
        let doomed = exact(&vectors, query, 5);
        for &id in &doomed {
            assert!(store.delete(id as u32)?);
        }
        assert_eq!(store.stale_deletes(), 5, "codes linger until a rebuild");

        let found = store.search(
            query,
            10,
            SearchParams {
                nprobe: 16,
                rerank_candidates: 200,
            },
        )?;
        for neighbour in &found {
            assert!(
                !doomed.contains(&neighbour.id),
                "deleted id {} was returned",
                neighbour.id
            );
        }
        assert!(store.get(doomed[0] as u32)?.is_none());
        Ok(())
    }

    /// An overwritten vector must be scored on its **new** value. The index
    /// holds a stale code, but re-ranking reads storage.
    #[test]
    fn an_updated_vector_is_scored_on_its_new_value() -> io::Result<()> {
        let dimension = 16;
        let dir = BenchDir::new("engine-update")?;
        let mut store = VectorStore::open(dir.path(), config(dimension))?;

        let mut vectors = synthetic(600, dimension, 41);
        populate(&mut store, &vectors)?;

        // Move vector 0 exactly onto a query point.
        let query = &synthetic(1, dimension, 43)[0];
        store.insert(0, query, b"moved")?;
        vectors[0] = query.clone();

        assert_eq!(store.get(0)?.expect("record").vector, *query);

        let found = store.search(
            query,
            5,
            SearchParams {
                nprobe: 16,
                rerank_candidates: 600,
            },
        )?;
        assert_eq!(found[0].id, 0, "the moved vector should now be nearest");
        assert!(found[0].distance < 1e-6);
        Ok(())
    }

    #[test]
    fn re_inserting_a_deleted_id_revives_it() -> io::Result<()> {
        let dimension = 8;
        let dir = BenchDir::new("engine-revive")?;
        let mut store = VectorStore::open(dir.path(), config(dimension))?;

        let vectors = synthetic(300, dimension, 47);
        populate(&mut store, &vectors)?;

        store.delete(5)?;
        assert!(store.get(5)?.is_none());

        store.insert(5, &vectors[5], b"back")?;
        assert!(store.get(5)?.is_some());
        assert!(
            !store.deleted.contains(&5),
            "a revived id must leave the deleted set"
        );

        let found = store.search(&vectors[5], 1, SearchParams::default())?;
        assert_eq!(found[0].id, 5);
        Ok(())
    }

    /// A rebuild must reclaim the slots deleted ids occupied.
    #[test]
    fn rebuilding_clears_stale_deletes() -> io::Result<()> {
        let dimension = 16;
        let dir = BenchDir::new("engine-rebuild")?;
        let mut store = VectorStore::open(dir.path(), config(dimension))?;

        let vectors = synthetic(600, dimension, 53);
        populate(&mut store, &vectors)?;
        for id in 0..50u32 {
            store.delete(id)?;
        }
        assert_eq!(store.stale_deletes(), 50);

        store.rebuild_index()?;
        assert_eq!(store.stale_deletes(), 0);
        assert_eq!(store.len(), 550);

        // And the deleted ones are still gone.
        let query = &synthetic(1, dimension, 59)[0];
        for neighbour in store.search(query, 20, SearchParams::default())? {
            assert!(neighbour.id >= 50, "id {} was deleted", neighbour.id);
        }
        Ok(())
    }

    // -----------------------------------------------------------------
    // Durability
    // -----------------------------------------------------------------

    /// The whole point of putting an LSM-tree underneath: vectors survive a
    /// restart, and the index reconstructs from them.
    #[test]
    fn vectors_and_search_survive_a_reopen() -> io::Result<()> {
        let dimension = 16;
        let dir = BenchDir::new("engine-reopen")?;
        let vectors = synthetic(700, dimension, 61);
        let query = &synthetic(1, dimension, 67)[0];

        let before = {
            let mut store = VectorStore::open(dir.path(), config(dimension))?;
            populate(&mut store, &vectors)?;
            store.delete(3)?;
            store.sync()?;
            store.search(query, 10, SearchParams::default())?
        };

        let store = VectorStore::open(dir.path(), config(dimension))?;
        assert_eq!(store.len(), 699, "one was deleted");
        assert!(store.is_indexed(), "the index rebuilds on open");
        assert_eq!(store.get(3)?, None, "the delete survived");

        let record = store.get(10)?.expect("record");
        assert_eq!(record.vector, vectors[10]);
        assert_eq!(record.metadata, b"meta-10");

        let after = store.search(query, 10, SearchParams::default())?;
        assert_eq!(
            after.iter().map(|n| n.id).collect::<Vec<_>>(),
            before.iter().map(|n| n.id).collect::<Vec<_>>(),
            "results must not change across a restart"
        );
        Ok(())
    }

    #[test]
    fn sparse_ids_are_handled() -> io::Result<()> {
        let dimension = 8;
        let dir = BenchDir::new("engine-sparse")?;
        let mut store = VectorStore::open(dir.path(), config(dimension))?;

        // Ids far from a dense 0..n range, which the rebuild must not assume.
        let vectors = synthetic(300, dimension, 71);
        for (position, vector) in vectors.iter().enumerate() {
            store.insert(position as u32 * 1000 + 7, vector, b"")?;
        }
        store.rebuild_index()?;

        let found = store.search(&vectors[5], 1, SearchParams::default())?;
        assert_eq!(found[0].id, 5 * 1000 + 7, "ids must survive a rebuild");
        Ok(())
    }

    #[test]
    fn degenerate_searches_are_handled() -> io::Result<()> {
        let dimension = 8;
        let dir = BenchDir::new("engine-degenerate")?;
        let mut store = VectorStore::open(dir.path(), config(dimension))?;
        let query = vec![0.0f32; dimension];

        assert!(store.search(&query, 5, SearchParams::default())?.is_empty());
        store.insert(1, &query, b"")?;
        assert!(store.search(&query, 0, SearchParams::default())?.is_empty());
        assert_eq!(store.search(&query, 100, SearchParams::default())?.len(), 1);

        assert!(!store.delete(999)?, "deleting an absent id reports false");
        Ok(())
    }

    #[test]
    #[should_panic(expected = "vector has the wrong length")]
    fn a_wrong_sized_vector_panics() {
        let dir = BenchDir::new("engine-badsize").expect("dir");
        let mut store = VectorStore::open(dir.path(), config(8)).expect("open");
        store.insert(1, &[1.0, 2.0], b"").expect("insert");
    }
}
