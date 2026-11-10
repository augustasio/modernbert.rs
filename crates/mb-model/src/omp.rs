//! Rung 19: our threads, from the OpenMP pool that oneDNN and MKL also use.
//!
//! Through rung 18 the work outside the matmuls runs on threads spawned for
//! each piece of work (`fused/threads.rs`), while oneDNN and MKL keep a second
//! set of threads in libgomp's pool. After each matmul those spin-wait for more
//! work, and so compete with the spawned threads for the same cores. At 32
//! threads on 16 cores that cost 4x (0.86 s against 0.21 s an example, with the
//! spinning turned off). PyTorch runs all of its work on the one pool, so the
//! spinning threads are the ones about to get the next piece of work.
//!
//! This is an FFI boundary, like `accelerate.rs` and `onednn.rs`, and all of
//! its `unsafe` is here.

use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};

// `GOMP_parallel` is what GCC compiles `#pragma omp parallel` into.
#[link(name = "gomp")]
extern "C" {
    fn GOMP_parallel(
        task: extern "C" fn(*mut c_void),
        data: *mut c_void,
        num_threads: u32,
        flags: u32,
    );
}

struct Job<'a> {
    work: &'a (dyn Fn(usize) + Sync),
    next: AtomicUsize,
    count: usize,
}

/// Every thread of the team runs this: claim the next index until none is left.
/// Claiming instead of using the thread's number means a team smaller than
/// `count`, which libgomp may give us, still finishes every index.
extern "C" fn run_claimed(data: *mut c_void) {
    // SAFETY: `data` is the `&Job` from `parallel_for`, which outlives the team
    // because `GOMP_parallel` returns only after every thread has finished.
    let job = unsafe { &*(data as *const Job) };
    loop {
        let index = job.next.fetch_add(1, Ordering::Relaxed);
        if index >= job.count {
            return;
        }
        (job.work)(index);
    }
}

/// Run `work(0)` to `work(count - 1)` on the OpenMP pool, once each.
pub fn parallel_for(count: usize, work: &(dyn Fn(usize) + Sync)) {
    if count <= 1 {
        if count == 1 {
            work(0);
        }
        return;
    }
    let job = Job {
        work,
        next: AtomicUsize::new(0),
        count,
    };
    // SAFETY: `run_claimed` only reads `job`, through atomics and a `Sync`
    // closure, and `GOMP_parallel` blocks until the team is done.
    unsafe {
        GOMP_parallel(
            run_claimed,
            &job as *const Job as *mut c_void,
            count as u32,
            0,
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_index_runs_exactly_once() {
        let hits: Vec<AtomicUsize> = (0..64).map(|_| AtomicUsize::new(0)).collect();
        parallel_for(hits.len(), &|i| {
            hits[i].fetch_add(1, Ordering::Relaxed);
        });
        assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
    }
}
