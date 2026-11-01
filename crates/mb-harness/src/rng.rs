//! A small seeded random number generator, for the slice order and the
//! example order.
//!
//! SplitMix64, which is short, well studied and good enough for shuffling.
//! The seed is recorded in every result, so any run's schedule can be rebuilt
//! exactly.

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform integer in `0..n`. The modulo bias is below 1e-15 for the
    /// sizes used here, far under anything a benchmark could notice.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Fisher-Yates, in place.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i + 1);
            items.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_gives_the_same_sequence() {
        let (mut a, mut b) = (Rng::new(7), Rng::new(7));
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn a_shuffle_is_a_permutation() {
        let mut v: Vec<usize> = (0..1000).collect();
        Rng::new(1).shuffle(&mut v);
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..1000).collect::<Vec<_>>());
        assert_ne!(v, sorted, "a shuffle of 1000 items should move something");
    }

    #[test]
    fn below_stays_in_range_and_covers_it() {
        let mut r = Rng::new(3);
        let mut seen = [0usize; 4];
        for _ in 0..4000 {
            seen[r.below(4)] += 1;
        }
        for (k, &count) in seen.iter().enumerate() {
            assert!(
                (800..1200).contains(&count),
                "value {k} drawn {count} times of 4000"
            );
        }
    }
}
