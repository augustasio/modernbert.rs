//! Score every AG News test example and write the four verbalizer logits.
//!
//! This is the Rust side of Tier 3: the logits go to a `.npy` file that the
//! Python statistics compare against rung 0, example by example. It is not a
//! benchmark. Timing belongs to the harness and its interleaved protocol, so
//! the wall clock printed here is only progress information.
//!
//! The token ids come from `data/agnews_test_prompt_ids.tsv`, written once by
//! `tools/export_prompt_ids.py`, so this rung reads byte-identical ids to
//! every other rung (hard rule 1). Rust's standard library has no SHA-256,
//! so this binary records the expected digest and the file size in a sidecar
//! `.json`, and the Python comparison recomputes the hash and checks it.
//!
//! ```text
//! cargo run --profile eval -p mb-model --features accelerate --bin eval_agnews -- \
//!     weights/ModernBERT-Large-Instruct data/agnews_test_prompt_ids.tsv \
//!     results/rung7_logits.npy
//! ```
//!
//! Rung 11 reads `MB_THREADS` for its thread count, 1 if unset.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use mb_model::fused::Workspace;
use mb_model::model::ModernBert;
use mb_model::ops::backend;

/// `[MASK]`, where the verbalizer logits are read.
const MASK_ID: u32 = 50284;

/// The four verbalizers, in rung 0's order: Business, World, Sports, Tech.
/// The `.npy` columns follow this order, so it must match rung 0 exactly.
const VERBALIZER_IDS: [u32; 4] = [10518, 3645, 15000, 18447];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!(
            "usage: {} <weights dir> <prompt ids .tsv> <out .npy>",
            args[0]
        );
        std::process::exit(2);
    }
    let (weights, ids_path, out_path) = (&args[1], Path::new(&args[2]), Path::new(&args[3]));

    let examples = read_prompt_ids(ids_path);
    let digest = expected_digest(ids_path);
    eprintln!(
        "{} examples, matmul backend {}, loading the model",
        examples.len(),
        backend()
    );

    let model = ModernBert::load(weights).unwrap_or_else(|e| {
        eprintln!("cannot load the model: {e}");
        std::process::exit(1);
    });

    let mut workspace = Workspace::new(model.config(), 512);
    // Rung 11 splits its work across `MB_THREADS` threads, 1 if unset.
    // Earlier rungs ignore it. The logits do not depend on it.
    let threads = match std::env::var("MB_THREADS") {
        Ok(t) => t.parse().unwrap_or_else(|_| {
            eprintln!("MB_THREADS is {t:?}, not a thread count");
            std::process::exit(2);
        }),
        Err(_) => 1,
    };
    workspace.set_threads(threads);
    let start = Instant::now();
    let mut logits = Vec::with_capacity(examples.len() * VERBALIZER_IDS.len());
    for (n, ids) in examples.iter().enumerate() {
        // The last [MASK], as the reference harness takes. Each prompt has
        // exactly one, so first and last agree.
        let mask = ids
            .iter()
            .rposition(|&id| id == MASK_ID)
            .unwrap_or_else(|| panic!("example {n} has no [MASK]"));
        // Rung 8 builds run the fused path, so its logits are the ones the
        // benchmark times.
        let row = if cfg!(feature = "fused") {
            let mut row = [0.0f32; 4];
            model
                .logits_for_tokens_fused(ids, mask, &VERBALIZER_IDS, &mut workspace, &mut row)
                .unwrap_or_else(|e| panic!("example {n}: {e}"));
            row.to_vec()
        } else {
            model
                .logits_for_tokens(ids, mask, &VERBALIZER_IDS)
                .unwrap_or_else(|e| panic!("example {n}: {e}"))
        };
        logits.extend_from_slice(&row);

        if (n + 1) % 100 == 0 || n + 1 == examples.len() {
            let done = (n + 1) as f64;
            let per = start.elapsed().as_secs_f64() / done;
            eprintln!(
                "{:5} / {}  {:.3} s per example, about {:.0} min left",
                n + 1,
                examples.len(),
                per,
                per * (examples.len() as f64 - done) / 60.0
            );
        }
    }

    write_npy(out_path, &logits, examples.len(), VERBALIZER_IDS.len());
    write_sidecar(
        out_path,
        ids_path,
        &digest,
        examples.len(),
        start.elapsed().as_secs_f64(),
    );
    eprintln!("wrote {} and its .json", out_path.display());
}

/// One line per example: index, gold label, space-separated token ids.
fn read_prompt_ids(path: &Path) -> Vec<Vec<u32>> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let mut out = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        let mut fields = line.split('\t');
        let index: usize = fields
            .next()
            .and_then(|f| f.parse().ok())
            .unwrap_or_else(|| panic!("line {line_no}: no index"));
        assert_eq!(
            index,
            out.len(),
            "line {line_no}: examples must be in order"
        );
        let _label = fields.next();
        let ids: Vec<u32> = fields
            .next()
            .unwrap_or_else(|| panic!("line {line_no}: no ids"))
            .split(' ')
            .map(|t| {
                t.parse()
                    .unwrap_or_else(|_| panic!("line {line_no}: bad id {t:?}"))
            })
            .collect();
        out.push(ids);
    }
    out
}

/// The SHA-256 that `tools/export_prompt_ids.py` recorded next to the ids.
/// Refuse to run without it, so a result can always be traced to its input.
fn expected_digest(path: &Path) -> String {
    let digest_path = path.with_extension("sha256");
    match std::fs::read_to_string(&digest_path) {
        Ok(d) => d.trim().to_string(),
        Err(_) => {
            eprintln!(
                "{} is missing. Regenerate the ids with tools/export_prompt_ids.py.",
                digest_path.display()
            );
            std::process::exit(1);
        }
    }
}

/// What this run was, next to its logits: the backend, the input it read,
/// and how long it took. The comparison script checks the digest.
fn write_sidecar(out: &Path, ids_path: &Path, digest: &str, n: usize, seconds: f64) {
    let bytes = std::fs::metadata(ids_path).map(|m| m.len()).unwrap_or(0);
    let json = format!(
        "{{\n  \"backend\": \"{}\",\n  \"prompt_ids_file\": \"{}\",\n  \
         \"prompt_ids_bytes\": {bytes},\n  \"prompt_ids_sha256_expected\": \"{digest}\",\n  \
         \"examples\": {n},\n  \"columns\": [\"Business\", \"World\", \"Sports\", \"Tech\"],\n  \
         \"wall_clock_seconds\": {seconds:.1},\n  \"note\": \"progress timing only, not a benchmark\"\n}}\n",
        backend(),
        ids_path.display(),
    );
    let path = out.with_extension("json");
    std::fs::write(&path, json).unwrap_or_else(|e| panic!("cannot write {}: {e}", path.display()));
}

/// Write a 2-D little-endian f32 array in NumPy's `.npy` format, version 1.0.
///
/// The format is a magic string, a header length, a Python dict literal
/// padded so the data starts on a 64-byte boundary, then the raw values.
fn write_npy(path: &Path, data: &[f32], rows: usize, cols: usize) {
    assert_eq!(data.len(), rows * cols);
    let dict = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
    // magic (6) + version (2) + header length (2) = 10 bytes before the dict.
    let unpadded = 10 + dict.len() + 1;
    let padding = (64 - unpadded % 64) % 64;
    let header = format!("{dict}{}\n", " ".repeat(padding));

    let mut file = std::fs::File::create(path)
        .unwrap_or_else(|e| panic!("cannot create {}: {e}", path.display()));
    let header_len = u16::try_from(header.len()).expect("the header is short");
    let mut bytes = Vec::with_capacity(10 + header.len() + data.len() * 4);
    bytes.extend_from_slice(b"\x93NUMPY\x01\x00");
    bytes.extend_from_slice(&header_len.to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    file.write_all(&bytes)
        .unwrap_or_else(|e| panic!("cannot write {}: {e}", path.display()));
}
