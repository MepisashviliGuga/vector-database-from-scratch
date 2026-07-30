//! CRC-32 (IEEE 802.3 / zlib polynomial), table-driven.
//!
//! Used to detect torn and corrupted write-ahead-log records. Hand-rolled rather
//! than pulled from `crc32fast` because this project is meant to be from
//! scratch; it is ~40 lines and validated against the standard check vector.
//!
//! This is the bit-reflected form: the polynomial is stored reversed
//! (`0xEDB88320` rather than `0x04C11DB7`) so the algorithm can shift right,
//! which is what every implementation you can cross-check against does.
//! Producing byte-identical output to `crc32fast`/zlib is the whole point — the
//! log format should be inspectable with standard tools.

/// Bit-reversed IEEE 802.3 polynomial.
const POLYNOMIAL: u32 = 0xEDB8_8320;

/// Per-byte lookup table, built at compile time.
static TABLE: [u32; 256] = build_table();

const fn build_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut byte = 0usize;
    while byte < 256 {
        let mut crc = byte as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ POLYNOMIAL
            } else {
                crc >> 1
            };
            bit += 1;
        }
        table[byte] = crc;
        byte += 1;
    }
    table
}

/// CRC-32 of `data`.
pub fn crc32(data: &[u8]) -> u32 {
    finish(update(0xFFFF_FFFF, data))
}

/// Fold `data` into a running CRC state. State starts at `0xFFFF_FFFF`; pass the
/// result through [`finish`] to get the checksum.
fn update(mut state: u32, data: &[u8]) -> u32 {
    for &byte in data {
        let index = ((state ^ byte as u32) & 0xFF) as usize;
        state = TABLE[index] ^ (state >> 8);
    }
    state
}

fn finish(state: u32) -> u32 {
    state ^ 0xFFFF_FFFF
}

/// Incremental CRC-32, for checksumming a record written in several pieces
/// without first concatenating it into one buffer.
#[derive(Debug, Clone)]
pub struct Crc32 {
    state: u32,
}

impl Default for Crc32 {
    fn default() -> Self {
        Self::new()
    }
}

impl Crc32 {
    pub fn new() -> Self {
        Self { state: 0xFFFF_FFFF }
    }

    pub fn update(&mut self, data: &[u8]) {
        self.state = update(self.state, data);
    }

    pub fn finish(&self) -> u32 {
        finish(self.state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_standard_check_vector() {
        // The canonical CRC-32 check value, which every conforming
        // implementation (zlib, crc32fast, Python's zlib.crc32) agrees on.
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn known_values() {
        assert_eq!(crc32(b""), 0x0000_0000);
        assert_eq!(crc32(b"a"), 0xE8B7_BE43);
        assert_eq!(crc32(b"hello world"), 0x0D4A_1185);
    }

    #[test]
    fn incremental_matches_one_shot() {
        let data = b"the quick brown fox jumps over the lazy dog";
        let mut hasher = Crc32::new();
        hasher.update(&data[..10]);
        hasher.update(&data[10..25]);
        hasher.update(&data[25..]);
        assert_eq!(hasher.finish(), crc32(data));
    }

    #[test]
    fn detects_single_bit_flips() {
        let original = b"a write-ahead log record".to_vec();
        let baseline = crc32(&original);

        for byte_index in 0..original.len() {
            for bit in 0..8 {
                let mut corrupted = original.clone();
                corrupted[byte_index] ^= 1 << bit;
                assert_ne!(
                    crc32(&corrupted),
                    baseline,
                    "flipping bit {bit} of byte {byte_index} went undetected"
                );
            }
        }
    }
}
