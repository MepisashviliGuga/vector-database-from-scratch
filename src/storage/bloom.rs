//! Bloom filter for SSTable point lookups.
//!
//! # Why an LSM-tree needs this
//!
//! A point lookup must consult every run that could hold the key, newest first.
//! Without a filter, a key that exists in one run — or in none — still costs a
//! block read per run examined. That read amplification is the central cost of
//! the LSM design, and it is what the growth-scheme and compaction papers spend
//! their analysis budget on.
//!
//! A Bloom filter answers "is this key definitely absent?" from memory. It can
//! say *maybe present* about a key it has never seen (a false positive), but it
//! can never say *absent* about a key that was inserted. That asymmetry is
//! exactly what a lookup needs: a negative answer is trustworthy, so the block
//! read can be skipped outright.
//!
//! # Sizing
//!
//! For `n` keys and a target false-positive rate `p`, the standard results are
//!
//! ```text
//! m = -n ln(p) / (ln 2)^2      bits
//! k = (m / n) ln 2             hash functions
//! ```
//!
//! which at `p = 1%` works out to about 9.6 bits and 7 hashes per key.
//!
//! # Hashing
//!
//! `k` independent hashes are simulated from two, using the Kirsch-Mitzenmacher
//! technique: `g_i(x) = h1(x) + i * h2(x)`. That paper ("Less Hashing, Same
//! Performance", ESA 2006) shows the false-positive rate is asymptotically
//! unchanged, so the cost is one hash computation per key instead of `k`.
//!
//! The base hash is FNV-1a passed through the MurmurHash3 64-bit finalizer.
//! FNV-1a alone has poor avalanche in its high bits, which would correlate the
//! two derived hashes and inflate the false-positive rate; the finalizer fixes
//! the bit distribution. [`tests::measured_false_positive_rate_matches_theory`]
//! checks empirically that this holds up.

use std::io;

/// Cap on hash count. Beyond this the filter costs more to probe than it saves,
/// and it only arises from an absurdly small target rate.
const MAX_HASHES: u32 = 30;

/// Header written ahead of the bit array: `num_bits` and `num_hashes`.
const ENCODED_HEADER_BYTES: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BloomFilter {
    bits: Vec<u8>,
    num_bits: u32,
    num_hashes: u32,
}

impl BloomFilter {
    /// Size a filter for `expected_entries` keys at a target false-positive
    /// rate.
    ///
    /// `false_positive_rate` is clamped into `(0, 1)`; the caller passing 0
    /// would otherwise ask for an infinitely large filter.
    pub fn with_capacity(expected_entries: usize, false_positive_rate: f64) -> Self {
        let rate = false_positive_rate.clamp(f64::MIN_POSITIVE, 0.999_999);
        let entries = expected_entries.max(1) as f64;

        let ln2 = std::f64::consts::LN_2;
        let bits = (-entries * rate.ln() / (ln2 * ln2)).ceil();
        // Round up to a whole byte, and keep at least one byte so the filter is
        // always well formed.
        let num_bits = (bits.max(8.0).min(u32::MAX as f64 - 8.0) as u32).div_ceil(8) * 8;

        let num_hashes = (((num_bits as f64 / entries) * ln2).round() as u32).clamp(1, MAX_HASHES);

        Self {
            bits: vec![0u8; (num_bits / 8) as usize],
            num_bits,
            num_hashes,
        }
    }

    pub fn insert(&mut self, key: &[u8]) {
        self.insert_hash(hash_key(key));
    }

    /// Insert by precomputed hash.
    ///
    /// The SSTable writer hashes each key as it streams past, then builds the
    /// filter at `finish` once the true key count is known — so it needs to
    /// insert hashes it computed earlier rather than re-walking the keys.
    pub fn insert_hash(&mut self, hash: u64) {
        for bit in bit_positions(hash, self.num_bits, self.num_hashes) {
            self.bits[(bit / 8) as usize] |= 1 << (bit % 8);
        }
    }

    /// `false` means the key is definitely absent. `true` means it may be
    /// present — the caller must still check.
    pub fn may_contain(&self, key: &[u8]) -> bool {
        self.may_contain_hash(hash_key(key))
    }

    pub fn may_contain_hash(&self, hash: u64) -> bool {
        bit_positions(hash, self.num_bits, self.num_hashes)
            .all(|bit| self.bits[(bit / 8) as usize] & (1 << (bit % 8)) != 0)
    }

    pub fn num_bits(&self) -> u32 {
        self.num_bits
    }

    pub fn num_hashes(&self) -> u32 {
        self.num_hashes
    }

    /// Bits set, over total bits. A saturated filter (near 1.0) has lost most of
    /// its value and signals that the sizing was wrong.
    pub fn fill_ratio(&self) -> f64 {
        let set: u32 = self.bits.iter().map(|byte| byte.count_ones()).sum();
        set as f64 / self.num_bits as f64
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(ENCODED_HEADER_BYTES + self.bits.len());
        bytes.extend_from_slice(&self.num_bits.to_le_bytes());
        bytes.extend_from_slice(&self.num_hashes.to_le_bytes());
        bytes.extend_from_slice(&self.bits);
        bytes
    }

    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < ENCODED_HEADER_BYTES {
            return Err(malformed("bloom filter header is truncated"));
        }
        let num_bits = u32::from_le_bytes(bytes[0..4].try_into().expect("4 bytes"));
        let num_hashes = u32::from_le_bytes(bytes[4..8].try_into().expect("4 bytes"));
        let bits = bytes[ENCODED_HEADER_BYTES..].to_vec();

        if num_bits == 0 || num_hashes == 0 {
            return Err(malformed("bloom filter has no bits or no hash functions"));
        }
        if num_bits as usize != bits.len() * 8 {
            return Err(malformed(
                "bloom filter bit count disagrees with its payload length",
            ));
        }
        Ok(Self {
            bits,
            num_bits,
            num_hashes,
        })
    }
}

/// The `k` bit positions a hash maps to, via `g_i = h1 + i * h2`.
///
/// A free function rather than a method so [`BloomFilter::insert_hash`] can walk
/// the positions while mutating the bit array.
fn bit_positions(hash: u64, num_bits: u32, num_hashes: u32) -> impl Iterator<Item = u32> {
    let h1 = (hash & 0xFFFF_FFFF) as u32;
    // Forcing h2 odd keeps the probe sequence from collapsing: an even h2 paired
    // with an even bit count can only reach half the filter, and h2 == 0 would
    // probe the same bit k times over.
    let h2 = ((hash >> 32) as u32) | 1;

    (0..num_hashes).map(move |i| h1.wrapping_add(i.wrapping_mul(h2)) % num_bits)
}

fn malformed(reason: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("malformed bloom filter: {reason}"),
    )
}

/// FNV-1a, then the MurmurHash3 64-bit finalizer.
///
/// The filter derives two 32-bit hashes from the two halves of this value, so
/// the halves must be independent. FNV-1a's high bits move sluggishly on short
/// keys — its last multiply cannot propagate carries upward far enough — which
/// would correlate the halves. The finalizer's shift-xor-multiply rounds give
/// full avalanche.
pub fn hash_key(key: &[u8]) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    let mut hash = FNV_OFFSET_BASIS;
    for &byte in key {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    fmix64(hash)
}

/// MurmurHash3's 64-bit finalizer.
fn fmix64(mut value: u64) -> u64 {
    value ^= value >> 33;
    value = value.wrapping_mul(0xff51_afd7_ed55_8ccd);
    value ^= value >> 33;
    value = value.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    value ^= value >> 33;
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(i: usize) -> Vec<u8> {
        format!("key{i:08}").into_bytes()
    }

    /// The one property a Bloom filter may never violate. A false negative in an
    /// SSTable filter means a lookup skips a block that holds the key, and the
    /// database silently loses data.
    #[test]
    fn never_reports_a_false_negative() {
        let count = 10_000;
        let mut filter = BloomFilter::with_capacity(count, 0.01);
        for i in 0..count {
            filter.insert(&key(i));
        }
        for i in 0..count {
            assert!(
                filter.may_contain(&key(i)),
                "inserted key {i} reported absent"
            );
        }
    }

    #[test]
    fn no_false_negatives_for_awkward_keys() {
        let keys: Vec<Vec<u8>> = vec![
            Vec::new(),
            vec![0],
            vec![0, 0, 0, 0],
            vec![0xFF; 64],
            (0..=255u8).collect(),
            b"a".to_vec(),
            b"ab".to_vec(),
        ];
        let mut filter = BloomFilter::with_capacity(keys.len(), 0.01);
        for k in &keys {
            filter.insert(k);
        }
        for k in &keys {
            assert!(filter.may_contain(k), "key {k:?} reported absent");
        }
    }

    /// The filter is only worth its memory if it hits roughly the rate it was
    /// sized for. A badly correlated hash shows up here as a rate several times
    /// the target.
    #[test]
    fn measured_false_positive_rate_matches_theory() {
        let inserted = 20_000;
        let target = 0.01;

        let mut filter = BloomFilter::with_capacity(inserted, target);
        for i in 0..inserted {
            filter.insert(&key(i));
        }

        // Probe with keys that were definitely not inserted.
        let probes = 100_000;
        let false_positives = (0..probes)
            .filter(|i| filter.may_contain(format!("absent{i:08}").as_bytes()))
            .count();
        let measured = false_positives as f64 / probes as f64;

        assert!(
            measured < target * 2.0,
            "measured false-positive rate {measured:.4} is far above the {target} target \
             (fill ratio {:.3}, {} hashes) — this usually means the two derived hashes \
             are correlated",
            filter.fill_ratio(),
            filter.num_hashes(),
        );
        // A rate far *below* target means the filter is oversized and wasting
        // memory, which is also a sizing bug worth catching.
        assert!(
            measured > target / 10.0,
            "measured rate {measured:.5} is implausibly low; check the sizing maths"
        );
    }

    #[test]
    fn sizing_follows_the_standard_formulas() {
        // ~9.6 bits and ~7 hashes per key at 1%.
        let filter = BloomFilter::with_capacity(1000, 0.01);
        let bits_per_key = filter.num_bits() as f64 / 1000.0;
        assert!(
            (9.0..=10.5).contains(&bits_per_key),
            "expected ~9.6 bits per key, got {bits_per_key:.2}"
        );
        assert_eq!(filter.num_hashes(), 7);

        // A tighter rate buys more bits and more hashes.
        let tighter = BloomFilter::with_capacity(1000, 0.001);
        assert!(tighter.num_bits() > filter.num_bits());
        assert!(tighter.num_hashes() > filter.num_hashes());
    }

    #[test]
    fn fill_ratio_lands_near_one_half_when_correctly_sized() {
        // Optimal k makes each bit a coin flip; a ratio far from 0.5 means the
        // sizing or the hashing is off.
        let count = 10_000;
        let mut filter = BloomFilter::with_capacity(count, 0.01);
        for i in 0..count {
            filter.insert(&key(i));
        }
        let ratio = filter.fill_ratio();
        assert!(
            (0.4..=0.6).contains(&ratio),
            "fill ratio {ratio:.3} is not near the expected 0.5"
        );
    }

    #[test]
    fn round_trips_through_encoding() {
        let mut filter = BloomFilter::with_capacity(500, 0.01);
        for i in 0..500 {
            filter.insert(&key(i));
        }

        let decoded = BloomFilter::decode(&filter.encode()).expect("decode");
        assert_eq!(decoded, filter);
        for i in 0..500 {
            assert!(decoded.may_contain(&key(i)));
        }
    }

    #[test]
    fn rejects_malformed_encodings() {
        assert!(BloomFilter::decode(&[]).is_err());
        assert!(BloomFilter::decode(&[0, 0, 0]).is_err());

        // Header claims 800 bits but carries only 8 bytes of payload.
        let mut bytes = 800u32.to_le_bytes().to_vec();
        bytes.extend_from_slice(&7u32.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 8]);
        assert!(BloomFilter::decode(&bytes).is_err());
    }

    #[test]
    fn an_empty_filter_rejects_everything() {
        let filter = BloomFilter::with_capacity(0, 0.01);
        assert!(!filter.may_contain(b"anything"));
        assert!(!filter.may_contain(b""));
    }

    #[test]
    fn extreme_rates_do_not_produce_a_degenerate_filter() {
        for rate in [0.0, 1.0, 0.5, 1e-9] {
            let filter = BloomFilter::with_capacity(100, rate);
            assert!(filter.num_bits() > 0);
            assert!(filter.num_bits().is_multiple_of(8));
            assert!((1..=MAX_HASHES).contains(&filter.num_hashes()));
        }
    }

    /// The base hash's two halves must be independent, since the filter treats
    /// them as separate hash functions.
    #[test]
    fn hash_halves_are_independent() {
        // Sequential keys are the worst case for FNV-1a's high bits.
        let hashes: Vec<u64> = (0..2000).map(|i| hash_key(&key(i))).collect();

        let high_bits_set: u32 = hashes.iter().map(|h| (h >> 32).count_ones()).sum();
        let expected = 2000 * 16;
        let deviation = (high_bits_set as i64 - expected as i64).abs();
        assert!(
            deviation < 800,
            "high 32 bits are poorly distributed: {high_bits_set} set vs {expected} expected"
        );

        // A one-byte change must scatter the whole output.
        let a = hash_key(b"key00000001");
        let b = hash_key(b"key00000002");
        let differing = (a ^ b).count_ones();
        assert!(
            (16..=48).contains(&differing),
            "avalanche is weak: only {differing} of 64 bits changed"
        );
    }
}
