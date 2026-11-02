//! Rung 8 allocates nothing in a forward pass.
//!
//! A counting global allocator wraps the system one. It counts only on the
//! thread that switches it on, so the test harness's own threads cannot add
//! to the count.
//!
//! What it cannot see: Accelerate's own memory. `cblas_sgemm` is C and calls
//! the system `malloc` directly, not Rust's allocator. So this asserts that
//! *our* code allocates nothing, which is the claim rung 8 makes.
//!
//! Memory: one test function, one model load (hard rule 8).

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use mb_model::fused::Workspace;
use mb_model::model::ModernBert;

struct Counting;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    // `const` so that reading it never allocates, inside the allocator.
    static COUNTING: Cell<bool> = const { Cell::new(false) };
}

// SAFETY: every call goes straight to `System`, which upholds the
// `GlobalAlloc` contract. The counter is a side effect only.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.with(Cell::get) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.with(Cell::get) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Run `f` and return how many allocations it made on this thread.
fn allocations_in(f: impl FnOnce()) -> usize {
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    COUNTING.with(|c| c.set(true));
    f();
    COUNTING.with(|c| c.set(false));
    ALLOCATIONS.load(Ordering::Relaxed) - before
}

/// Business, World, Sports, Tech.
const VERBALIZER_IDS: [u32; 4] = [10518, 3645, 15000, 18447];

/// A deterministic prompt of `len` ids from the ordinary vocabulary, with
/// the mask at the end.
fn prompt(len: usize) -> Vec<u32> {
    let mut ids: Vec<u32> = (0..len as u32).map(|i| 1000 + (i * 7919) % 40000).collect();
    ids[len - 1] = 50284;
    ids
}

#[test]
fn a_forward_pass_allocates_nothing() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model = ModernBert::load(root.join("weights/ModernBERT-Large-Instruct"))
        .expect("the model must load");

    // The workspace starts too small for 150 tokens, so the first call grows
    // it. That one allocates, which shows the counter works.
    let mut ws = Workspace::new(model.config(), 8);
    let mut logits = [0.0f32; 4];
    let long = prompt(150);
    let grew = allocations_in(|| {
        model
            .logits_for_tokens_fused(&long, 149, &VERBALIZER_IDS, &mut ws, &mut logits)
            .expect("runs");
    });
    assert!(
        grew > 0,
        "growing the workspace must allocate, or the counter is broken"
    );
    let first = logits;

    // The same prompt again, and a shorter one with a local window that
    // fits inside the sequence: nothing.
    let short = prompt(97);
    for (ids, label) in [(&long, "the same 150 tokens"), (&short, "97 tokens")] {
        let n = allocations_in(|| {
            model
                .logits_for_tokens_fused(ids, ids.len() - 1, &VERBALIZER_IDS, &mut ws, &mut logits)
                .expect("runs");
        });
        println!("{label}: {n} allocations");
        assert_eq!(n, 0, "{label}: the forward pass made {n} allocations");
    }

    // A reused workspace must not leak state from one call into the next.
    model
        .logits_for_tokens_fused(&long, 149, &VERBALIZER_IDS, &mut ws, &mut logits)
        .expect("runs");
    assert_eq!(
        logits, first,
        "the same prompt gave different logits on a reused workspace"
    );
}
