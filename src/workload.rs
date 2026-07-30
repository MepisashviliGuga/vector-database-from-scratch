//! Deterministic workload generation for the storage benchmarks.
//!
//! # Why this exists rather than a crate
//!
//! Every number this project reports has to be reproducible from a seed, and
//! the generator is part of what is being reported — a benchmark that hides its
//! key distribution behind someone else's RNG cannot be checked. It is also
//! small: a seeded PRNG, three key distributions, and an operation mix.
//!
//! # Key distribution is a first-class axis
//!
//! Earlier in this project, leveling and tiering measured *identical* write
//! amplification, and the reason was the workload: with strictly ascending keys
//! every flush covers a range disjoint from every earlier one, so leveling finds
//! nothing to rewrite and degenerates into tiering. Scatter the same keys and
//! the difference appears.
//!
//! So a benchmark that only inserts sequential keys would conclude the choice of
//! compaction policy does not matter. [`KeyDistribution`] is therefore swept
//! alongside value size and operation mix, not fixed.
//!
//! # Value size is too
//!
//! LSM compaction research is measured on ~100-byte values. This project stores
//! vectors: 512 bytes for SIFT, 3840 for GIST. Write amplification at that size
//! is dominated by recopying value bytes, which is the problem key-value
//! separation exists to solve — so the papers' trends may simply not hold here.
//! Measuring at both sizes is the point.

/// A small, fast, seeded PRNG: xorshift64* .
///
/// Not cryptographic and not trying to be. What it must be is *reproducible*:
/// the same seed gives the same stream on every machine, so a reported number
/// can be re-derived rather than taken on trust.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// A seed of 0 would leave xorshift stuck at 0 forever, so it is nudged.
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[0, 1)`, using the 53 bits an `f64` can represent exactly.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in `[0, bound)`.
    ///
    /// Uses rejection sampling rather than a modulo: `next_u64() % bound` biases
    /// towards small values whenever `bound` does not divide `2^64`, which would
    /// quietly skew every "uniform" benchmark.
    pub fn below(&mut self, bound: u64) -> u64 {
        if bound == 0 {
            return 0;
        }
        let zone = u64::MAX - (u64::MAX % bound) - 1;
        loop {
            let value = self.next_u64();
            if value <= zone {
                return value % bound;
            }
        }
    }
}

/// How keys are chosen from the key space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KeyDistribution {
    /// Ascending, wrapping at the end of the key space. Every flush covers a
    /// range disjoint from the last, which is the best case for compaction and
    /// the case that hides the difference between policies.
    Sequential,
    /// Uniform over the key space. Every flush spans the whole range, so runs
    /// overlap maximally.
    Uniform,
    /// Zipfian: a small set of hot keys takes most of the traffic, as in real
    /// workloads. `theta` controls skew; 0.99 is the YCSB default.
    Zipfian { theta: f64 },
}

impl KeyDistribution {
    /// Short name for CSV output and plot legends.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
            Self::Uniform => "uniform",
            Self::Zipfian { .. } => "zipfian",
        }
    }
}

/// Draws keys according to a [`KeyDistribution`].
#[derive(Debug, Clone)]
pub struct KeyGenerator {
    distribution: KeyDistribution,
    key_count: u64,
    /// Cursor for [`KeyDistribution::Sequential`].
    next_sequential: u64,
    /// Precomputed `zeta(n, theta)` for the Zipfian case; `O(n)` once.
    zeta_n: f64,
    zeta_2: f64,
    alpha: f64,
    eta: f64,
}

impl KeyGenerator {
    pub fn new(distribution: KeyDistribution, key_count: u64) -> Self {
        let key_count = key_count.max(1);
        let mut generator = Self {
            distribution,
            key_count,
            next_sequential: 0,
            zeta_n: 0.0,
            zeta_2: 0.0,
            alpha: 0.0,
            eta: 0.0,
        };

        if let KeyDistribution::Zipfian { theta } = distribution {
            // The standard formulation (Gray et al., as used by YCSB).
            generator.zeta_n = zeta(key_count, theta);
            generator.zeta_2 = zeta(2, theta);
            generator.alpha = 1.0 / (1.0 - theta);
            generator.eta = (1.0 - (2.0 / key_count as f64).powf(1.0 - theta))
                / (1.0 - generator.zeta_2 / generator.zeta_n);
        }
        generator
    }

    pub fn key_count(&self) -> u64 {
        self.key_count
    }

    /// Draw the next key id.
    pub fn next(&mut self, rng: &mut Rng) -> u64 {
        match self.distribution {
            KeyDistribution::Sequential => {
                let id = self.next_sequential;
                self.next_sequential = (self.next_sequential + 1) % self.key_count;
                id
            }
            KeyDistribution::Uniform => rng.below(self.key_count),
            KeyDistribution::Zipfian { theta } => {
                let u = rng.next_f64();
                let uz = u * self.zeta_n;
                if uz < 1.0 {
                    return 0;
                }
                if uz < 1.0 + 0.5f64.powf(theta) {
                    return 1;
                }
                let id = (self.key_count as f64 * (self.eta * u - self.eta + 1.0).powf(self.alpha))
                    as u64;
                id.min(self.key_count - 1)
            }
        }
    }
}

/// `zeta(n, theta) = sum over i in 1..=n of 1/i^theta`.
fn zeta(n: u64, theta: f64) -> f64 {
    (1..=n).map(|i| 1.0 / (i as f64).powf(theta)).sum()
}

/// One operation for the engine to perform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operation {
    Put {
        key: Vec<u8>,
        value: Vec<u8>,
    },
    Get {
        key: Vec<u8>,
    },
    Delete {
        key: Vec<u8>,
    },
    /// A range scan starting at `key`, reading up to `length` entries.
    Scan {
        key: Vec<u8>,
        length: usize,
    },
}

impl Operation {
    /// Short name for per-operation latency breakdowns.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Put { .. } => "put",
            Self::Get { .. } => "get",
            Self::Delete { .. } => "delete",
            Self::Scan { .. } => "scan",
        }
    }
}

/// Encode a key id as 8 big-endian bytes.
///
/// Big-endian so byte order matches numeric order — the engine sorts keys
/// lexicographically, and a little-endian encoding would scramble a "sequential"
/// workload into a scattered one, silently invalidating the whole distribution
/// axis.
pub fn encode_key(id: u64) -> Vec<u8> {
    id.to_be_bytes().to_vec()
}

/// What mix of operations to generate, over what key space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorkloadSpec {
    pub key_count: u64,
    pub value_bytes: usize,
    pub distribution: KeyDistribution,
    /// Proportions; normalised, so they need not sum to exactly 1.
    pub get_ratio: f64,
    pub put_ratio: f64,
    pub delete_ratio: f64,
    pub scan_ratio: f64,
    /// Entries a scan reads. EcoTune's cost model turns on the *long* range
    /// scan, which it defines as longer than `C + 1` entries.
    pub scan_length: usize,
    pub seed: u64,
}

impl Default for WorkloadSpec {
    /// YCSB workload A: an even read/write split over Zipfian keys.
    fn default() -> Self {
        Self {
            key_count: 100_000,
            value_bytes: 100,
            distribution: KeyDistribution::Zipfian { theta: 0.99 },
            get_ratio: 0.5,
            put_ratio: 0.5,
            delete_ratio: 0.0,
            scan_ratio: 0.0,
            scan_length: 100,
            seed: 42,
        }
    }
}

impl WorkloadSpec {
    /// Write-only: the load phase, and the shape write-amplification numbers
    /// come from.
    pub fn write_only(key_count: u64, value_bytes: usize, distribution: KeyDistribution) -> Self {
        Self {
            key_count,
            value_bytes,
            distribution,
            get_ratio: 0.0,
            put_ratio: 1.0,
            delete_ratio: 0.0,
            scan_ratio: 0.0,
            ..Self::default()
        }
    }

    /// Read-only, for measuring read amplification against a settled tree.
    pub fn read_only(key_count: u64, value_bytes: usize, distribution: KeyDistribution) -> Self {
        Self {
            key_count,
            value_bytes,
            distribution,
            get_ratio: 1.0,
            put_ratio: 0.0,
            delete_ratio: 0.0,
            scan_ratio: 0.0,
            ..Self::default()
        }
    }

    /// Scan-heavy: the case EcoTune's cost model is built around, where the
    /// number of sorted runs actually drives I/O.
    pub fn scan_heavy(key_count: u64, value_bytes: usize, distribution: KeyDistribution) -> Self {
        Self {
            key_count,
            value_bytes,
            distribution,
            get_ratio: 0.2,
            put_ratio: 0.2,
            delete_ratio: 0.0,
            scan_ratio: 0.6,
            ..Self::default()
        }
    }

    /// A label for CSV output that captures the axes being swept.
    pub fn label(&self) -> String {
        format!(
            "{}-{}B-g{:.0}p{:.0}d{:.0}s{:.0}",
            self.distribution.name(),
            self.value_bytes,
            self.get_ratio * 100.0,
            self.put_ratio * 100.0,
            self.delete_ratio * 100.0,
            self.scan_ratio * 100.0,
        )
    }
}

/// Generates a reproducible stream of operations.
#[derive(Debug, Clone)]
pub struct Workload {
    spec: WorkloadSpec,
    rng: Rng,
    keys: KeyGenerator,
    /// Cumulative operation weights, for one comparison per draw.
    thresholds: [f64; 4],
}

impl Workload {
    pub fn new(spec: WorkloadSpec) -> Self {
        let total = spec.get_ratio + spec.put_ratio + spec.delete_ratio + spec.scan_ratio;

        // An all-zero mix has no meaningful normalisation. Fall back to writes,
        // so the caller gets a usable workload rather than a division by zero or
        // an arbitrary operation type.
        let thresholds = if total <= 0.0 {
            [0.0, 1.0, 1.0, 1.0]
        } else {
            let get = spec.get_ratio / total;
            let put = get + spec.put_ratio / total;
            let delete = put + spec.delete_ratio / total;
            [get, put, delete, 1.0]
        };

        Self {
            rng: Rng::new(spec.seed),
            keys: KeyGenerator::new(spec.distribution, spec.key_count),
            thresholds,
            spec,
        }
    }

    pub fn spec(&self) -> &WorkloadSpec {
        &self.spec
    }

    /// The next operation.
    ///
    /// [`Workload`] also implements [`Iterator`], which is the usual way to
    /// consume it: `workload.by_ref().take(10_000)`.
    pub fn generate(&mut self) -> Operation {
        let key = encode_key(self.keys.next(&mut self.rng));
        let roll = self.rng.next_f64();

        if roll < self.thresholds[0] {
            Operation::Get { key }
        } else if roll < self.thresholds[1] {
            Operation::Put {
                key,
                // A constant fill byte: value *content* is irrelevant to the
                // storage engine, and generating random bytes would spend more
                // time in the RNG than in the database.
                value: vec![0xAB; self.spec.value_bytes],
            }
        } else if roll < self.thresholds[2] {
            Operation::Delete { key }
        } else {
            Operation::Scan {
                key,
                length: self.spec.scan_length,
            }
        }
    }
}

/// An endless stream of operations. Bound it with [`Iterator::take`].
impl Iterator for Workload {
    type Item = Operation;

    fn next(&mut self) -> Option<Operation> {
        Some(self.generate())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // -----------------------------------------------------------------
    // The RNG
    // -----------------------------------------------------------------

    /// The whole point: a seed must reproduce a stream exactly, or no reported
    /// number can be re-derived.
    #[test]
    fn the_same_seed_gives_the_same_stream() {
        let first: Vec<u64> = (0..100)
            .scan(Rng::new(7), |r, _| Some(r.next_u64()))
            .collect();
        let second: Vec<u64> = (0..100)
            .scan(Rng::new(7), |r, _| Some(r.next_u64()))
            .collect();
        assert_eq!(first, second);

        let different: Vec<u64> = (0..100)
            .scan(Rng::new(8), |r, _| Some(r.next_u64()))
            .collect();
        assert_ne!(first, different, "different seeds must diverge");
    }

    #[test]
    fn a_zero_seed_does_not_stick() {
        let mut rng = Rng::new(0);
        let values: Vec<u64> = (0..10).map(|_| rng.next_u64()).collect();
        assert!(
            values.iter().any(|&v| v != 0),
            "xorshift seeded at zero stays at zero forever"
        );
    }

    #[test]
    fn floats_stay_in_the_unit_interval() {
        let mut rng = Rng::new(1);
        for _ in 0..10_000 {
            let value = rng.next_f64();
            assert!((0.0..1.0).contains(&value), "{value} escaped [0, 1)");
        }
    }

    /// `next_u64() % bound` biases towards small values. Rejection sampling
    /// avoids it, and a badly biased generator would skew every "uniform"
    /// benchmark without anyone noticing.
    #[test]
    fn bounded_draws_are_close_to_uniform() {
        let mut rng = Rng::new(99);
        let buckets = 10u64;
        let draws = 200_000;
        let mut counts = vec![0usize; buckets as usize];
        for _ in 0..draws {
            counts[rng.below(buckets) as usize] += 1;
        }

        let expected = draws as f64 / buckets as f64;
        for (bucket, &count) in counts.iter().enumerate() {
            let deviation = (count as f64 - expected).abs() / expected;
            assert!(
                deviation < 0.05,
                "bucket {bucket} got {count}, expected about {expected}"
            );
        }
    }

    #[test]
    fn a_bound_of_one_or_zero_is_handled() {
        let mut rng = Rng::new(3);
        assert_eq!(rng.below(1), 0);
        assert_eq!(rng.below(0), 0);
    }

    // -----------------------------------------------------------------
    // Key distributions
    // -----------------------------------------------------------------

    #[test]
    fn sequential_keys_ascend_and_wrap() {
        let mut rng = Rng::new(1);
        let mut keys = KeyGenerator::new(KeyDistribution::Sequential, 5);
        let drawn: Vec<u64> = (0..12).map(|_| keys.next(&mut rng)).collect();
        assert_eq!(drawn, vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1]);
    }

    #[test]
    fn uniform_keys_cover_the_space() {
        let mut rng = Rng::new(5);
        let mut keys = KeyGenerator::new(KeyDistribution::Uniform, 1000);
        let mut seen = vec![false; 1000];
        for _ in 0..20_000 {
            seen[keys.next(&mut rng) as usize] = true;
        }
        let covered = seen.iter().filter(|&&s| s).count();
        assert!(covered > 990, "only {covered} of 1000 keys were ever drawn");
    }

    /// The defining property of Zipfian: a small hot set takes most of the
    /// traffic. Without this the "skewed" axis of the benchmark is a lie.
    #[test]
    fn zipfian_keys_concentrate_on_a_hot_set() {
        let mut rng = Rng::new(11);
        let key_count = 10_000u64;
        let mut keys = KeyGenerator::new(KeyDistribution::Zipfian { theta: 0.99 }, key_count);

        let draws = 100_000;
        let mut counts: HashMap<u64, usize> = HashMap::new();
        for _ in 0..draws {
            *counts.entry(keys.next(&mut rng)).or_default() += 1;
        }

        let mut frequencies: Vec<usize> = counts.values().copied().collect();
        frequencies.sort_unstable_by(|a, b| b.cmp(a));

        let hot_set = (key_count as f64 * 0.01) as usize;
        let hot_traffic: usize = frequencies.iter().take(hot_set).sum();
        let share = hot_traffic as f64 / draws as f64;

        assert!(
            share > 0.25,
            "the hottest 1% of keys took only {:.1}% of traffic; that is not skewed",
            share * 100.0
        );
    }

    #[test]
    fn zipfian_keys_stay_inside_the_key_space() {
        let mut rng = Rng::new(13);
        let key_count = 500u64;
        let mut keys = KeyGenerator::new(KeyDistribution::Zipfian { theta: 0.8 }, key_count);
        for _ in 0..50_000 {
            let id = keys.next(&mut rng);
            assert!(id < key_count, "{id} is outside a {key_count}-key space");
        }
    }

    #[test]
    fn a_higher_theta_is_more_skewed() {
        fn hot_share(theta: f64) -> f64 {
            let mut rng = Rng::new(17);
            let key_count = 5_000u64;
            let mut keys = KeyGenerator::new(KeyDistribution::Zipfian { theta }, key_count);
            let draws = 50_000;
            let mut counts: HashMap<u64, usize> = HashMap::new();
            for _ in 0..draws {
                *counts.entry(keys.next(&mut rng)).or_default() += 1;
            }
            let mut frequencies: Vec<usize> = counts.values().copied().collect();
            frequencies.sort_unstable_by(|a, b| b.cmp(a));
            frequencies.iter().take(50).sum::<usize>() as f64 / draws as f64
        }

        assert!(
            hot_share(0.99) > hot_share(0.5),
            "theta = 0.99 should concentrate more than theta = 0.5"
        );
    }

    // -----------------------------------------------------------------
    // Keys
    // -----------------------------------------------------------------

    /// Byte order must match numeric order, or a "sequential" workload becomes
    /// a scattered one and the distribution axis measures nothing.
    #[test]
    fn key_encoding_preserves_numeric_order() {
        let ids = [0u64, 1, 255, 256, 65_535, 65_536, u64::MAX / 2, u64::MAX];
        for pair in ids.windows(2) {
            assert!(
                encode_key(pair[0]) < encode_key(pair[1]),
                "{} should encode below {}",
                pair[0],
                pair[1]
            );
        }
        assert_eq!(encode_key(1).len(), 8);
    }

    // -----------------------------------------------------------------
    // The operation mix
    // -----------------------------------------------------------------

    #[test]
    fn the_operation_mix_matches_the_requested_ratios() {
        let spec = WorkloadSpec {
            key_count: 1000,
            get_ratio: 0.5,
            put_ratio: 0.3,
            delete_ratio: 0.1,
            scan_ratio: 0.1,
            ..WorkloadSpec::default()
        };
        let mut workload = Workload::new(spec);

        let draws = 100_000;
        let mut counts: HashMap<&'static str, usize> = HashMap::new();
        for _ in 0..draws {
            *counts.entry(workload.generate().kind()).or_default() += 1;
        }

        for (kind, expected) in [("get", 0.5), ("put", 0.3), ("delete", 0.1), ("scan", 0.1)] {
            let actual = counts[kind] as f64 / draws as f64;
            assert!(
                (actual - expected).abs() < 0.01,
                "{kind} was {actual:.3}, expected {expected}"
            );
        }
    }

    /// Ratios that do not sum to 1 are normalised rather than rejected, so a
    /// caller can write 2:1:1 without doing arithmetic.
    #[test]
    fn ratios_are_normalised() {
        let spec = WorkloadSpec {
            get_ratio: 2.0,
            put_ratio: 1.0,
            delete_ratio: 1.0,
            scan_ratio: 0.0,
            ..WorkloadSpec::default()
        };
        let mut workload = Workload::new(spec);

        let draws = 40_000;
        let gets = (0..draws)
            .filter(|_| workload.generate().kind() == "get")
            .count();
        let share = gets as f64 / draws as f64;
        assert!(
            (share - 0.5).abs() < 0.02,
            "gets were {share:.3}, expected 0.5"
        );
    }

    #[test]
    fn an_empty_mix_falls_back_to_writes_rather_than_dividing_by_zero() {
        let spec = WorkloadSpec {
            get_ratio: 0.0,
            put_ratio: 0.0,
            delete_ratio: 0.0,
            scan_ratio: 0.0,
            ..WorkloadSpec::default()
        };
        let mut workload = Workload::new(spec);
        assert_eq!(workload.generate().kind(), "put");
    }

    #[test]
    fn a_workload_is_reproducible_from_its_seed() {
        let spec = WorkloadSpec {
            key_count: 500,
            seed: 2024,
            ..WorkloadSpec::default()
        };
        let first: Vec<_> = Workload::new(spec).take(1000).collect();
        let second: Vec<_> = Workload::new(spec).take(1000).collect();
        assert_eq!(first, second);

        let other: Vec<_> = Workload::new(WorkloadSpec { seed: 2025, ..spec })
            .take(1000)
            .collect();
        assert_ne!(first, other);
    }

    #[test]
    fn values_are_the_requested_size() {
        let mut workload = Workload::new(WorkloadSpec::write_only(
            100,
            3840,
            KeyDistribution::Sequential,
        ));
        match workload.generate() {
            Operation::Put { value, .. } => assert_eq!(value.len(), 3840, "a GIST vector"),
            other => panic!("expected a put, got {other:?}"),
        }
    }

    #[test]
    fn the_preset_mixes_are_what_they_claim() {
        assert_eq!(
            WorkloadSpec::write_only(10, 100, KeyDistribution::Uniform).get_ratio,
            0.0
        );
        assert_eq!(
            WorkloadSpec::read_only(10, 100, KeyDistribution::Uniform).put_ratio,
            0.0
        );
        assert!(WorkloadSpec::scan_heavy(10, 100, KeyDistribution::Uniform).scan_ratio > 0.5);
    }

    #[test]
    fn labels_capture_the_swept_axes() {
        let spec = WorkloadSpec {
            value_bytes: 512,
            distribution: KeyDistribution::Uniform,
            get_ratio: 0.5,
            put_ratio: 0.5,
            delete_ratio: 0.0,
            scan_ratio: 0.0,
            ..WorkloadSpec::default()
        };
        assert_eq!(spec.label(), "uniform-512B-g50p50d0s0");
    }
}
