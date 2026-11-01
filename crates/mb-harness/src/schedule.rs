//! Which examples run in which cycle, and in which order.
//!
//! Each cycle runs one slice on the ruler and one on the variant, on the
//! **same** examples, in an order chosen at random per cycle. Same examples
//! makes the two slices a paired comparison. Random order stops either side
//! from always running hotter, second, after the other has warmed the chip.
//!
//! A pure function of its inputs and the seed, so a schedule can be rebuilt
//! exactly from a result file.

use crate::rng::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Ruler,
    Variant,
}

impl Role {
    pub fn name(self) -> &'static str {
        match self {
            Role::Ruler => "ruler",
            Role::Variant => "variant",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cycle {
    pub index: usize,
    /// The examples both slices of this cycle run, in this order.
    pub examples: Vec<usize>,
    /// Which role goes first.
    pub order: [Role; 2],
}

/// Plan `cycles` cycles of `per_slice` examples each.
///
/// Examples come from a seeded shuffle of `0..example_count`, taken in turn
/// and wrapping round if the run needs more than there are.
pub fn plan(example_count: usize, cycles: usize, per_slice: usize, seed: u64) -> Vec<Cycle> {
    assert!(example_count > 0 && per_slice > 0, "nothing to schedule");
    let mut rng = Rng::new(seed);
    let mut deck: Vec<usize> = (0..example_count).collect();
    rng.shuffle(&mut deck);

    let mut next = 0usize;
    let mut out = Vec::with_capacity(cycles);
    for index in 0..cycles {
        let examples = (0..per_slice)
            .map(|_| {
                let e = deck[next % deck.len()];
                next += 1;
                e
            })
            .collect();
        let order = if rng.below(2) == 0 {
            [Role::Ruler, Role::Variant]
        } else {
            [Role::Variant, Role::Ruler]
        };
        out.push(Cycle {
            index,
            examples,
            order,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_gives_the_same_schedule() {
        assert_eq!(plan(7600, 30, 20, 5), plan(7600, 30, 20, 5));
        assert_ne!(plan(7600, 30, 20, 5), plan(7600, 30, 20, 6));
    }

    #[test]
    fn every_cycle_runs_both_roles_once() {
        for c in plan(100, 50, 3, 1) {
            assert_ne!(
                c.order[0], c.order[1],
                "cycle {} runs one role twice",
                c.index
            );
        }
    }

    #[test]
    fn the_order_is_balanced_over_many_cycles() {
        let ruler_first = plan(100, 1000, 1, 2)
            .iter()
            .filter(|c| c.order[0] == Role::Ruler)
            .count();
        assert!(
            (430..570).contains(&ruler_first),
            "the ruler went first in {ruler_first} of 1000 cycles"
        );
    }

    #[test]
    fn no_example_repeats_until_the_deck_runs_out() {
        let all: Vec<usize> = plan(60, 3, 20, 4)
            .into_iter()
            .flat_map(|c| c.examples)
            .collect();
        let mut sorted = all.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..60).collect::<Vec<_>>());
    }
}
