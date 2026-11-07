//! Reading `model.safetensors` by hand.
//!
//! The format is simple enough to read directly, which is the point: eight
//! bytes of little-endian header length, that many bytes of JSON describing
//! each tensor's dtype, shape and byte range, then the raw values.
//!
//! All 173 tensors in this checkpoint are bf16. There is no fp32 download for
//! ModernBERT-Large-Instruct, so fp32 means upcasting here, and every rung of
//! the ladder must upcast the same way or the numerical comparison means
//! nothing. bf16 is the top 16 bits of an f32, so the upcast is exact: shift
//! left by 16 and reinterpret. No rounding, no precision choice to get wrong.
//!
//! **Three ways to read, and the cheap ones exist for a reason.** This
//! checkpoint is 791 MB on disk and 1.58 GB once it is f32, so a careless
//! test suite can ask for many gigabytes at once: `cargo test` runs its test
//! functions in parallel across every core, and a helper that loads the whole
//! file per test multiplies that by ten. It did, on a 16 GB machine, and the
//! machine died. So:
//!
//! * [`Header`] reads names, shapes and dtypes and no values at all. Every
//!   structural check wants this.
//! * [`Checkpoint::load_one`] reads one tensor's bytes by seeking to its
//!   range. Use it to check a single tensor.
//! * [`Checkpoint::from_file`] reads everything, and only the forward pass
//!   needs that. A test that calls it should load once for the whole binary,
//!   not once per test function.
//!
//! The file is read into owned memory rather than mapped. That is a
//! measurement decision, not a performance one: mapped pages count toward RSS
//! only once touched and can be evicted, so a mapping variant looks
//! artificially lean beside one that reads. mmap gets measured separately.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use serde_json::Value;

/// A problem with the file. The message names the tensor, because a 791 MB
/// file and "unexpected end of input" together help nobody.
#[derive(Debug)]
pub struct TensorError(String);

impl TensorError {
    pub fn new(message: impl Into<String>) -> Self {
        TensorError(message.into())
    }
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for TensorError {}

/// What the header says about one tensor: everything but its values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Byte range within the payload, which starts after the header.
    pub start: usize,
    pub end: usize,
}

impl TensorInfo {
    /// How many values the shape describes.
    pub fn count(&self) -> usize {
        self.shape.iter().product()
    }
}

/// One tensor's values, upcast to f32, with the shape the file gave.
pub struct RawTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl RawTensor {
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// The header alone: names, shapes, dtypes and byte ranges, no values.
///
/// Reading this costs the first 18 KB of the file, so a structural test is
/// effectively free. Prefer it over [`Checkpoint`] whenever the values are
/// not actually needed.
pub struct Header {
    tensors: HashMap<String, TensorInfo>,
    /// Where the payload begins, for [`Checkpoint::load_one`].
    payload_start: u64,
}

impl Header {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TensorError> {
        let path = path.as_ref();
        let mut file = File::open(path)
            .map_err(|e| TensorError(format!("cannot open {}: {e}", path.display())))?;

        let mut length = [0u8; 8];
        file.read_exact(&mut length).map_err(|e| {
            TensorError(format!(
                "{}: cannot read the 8-byte header length: {e}",
                path.display()
            ))
        })?;
        let header_len = u64::from_le_bytes(length) as usize;

        let mut json = vec![0u8; header_len];
        file.read_exact(&mut json).map_err(|e| {
            TensorError(format!(
                "{}: the header claims {header_len} bytes but reading them failed: {e}",
                path.display()
            ))
        })?;

        Ok(Header {
            tensors: parse_entries(&json)?,
            payload_start: 8 + header_len as u64,
        })
    }

    pub fn get(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Total number of values across every tensor.
    pub fn parameter_count(&self) -> usize {
        self.tensors.values().map(TensorInfo::count).sum()
    }

    /// Fetch a tensor's description and check its shape, naming both when it
    /// is wrong.
    pub fn require(&self, name: &str, shape: &[usize]) -> Result<&TensorInfo, TensorError> {
        let info = self.tensors.get(name).ok_or_else(|| {
            TensorError(format!(
                "the checkpoint has no tensor {name:?}. Note that the encoder tensors are \
                 prefixed `model.` while the head tensors are not."
            ))
        })?;
        if info.shape != shape {
            return Err(TensorError(format!(
                "{name}: shape is {:?}, expected {shape:?}",
                info.shape
            )));
        }
        Ok(info)
    }
}

/// Every tensor in the file, with its values.
pub struct Checkpoint {
    tensors: HashMap<String, RawTensor>,
}

impl Checkpoint {
    /// Read one tensor's values, seeking straight to its byte range.
    ///
    /// This touches the header plus that tensor, so checking `decoder.bias`
    /// costs 100 KB rather than 791 MB.
    pub fn load_one(path: impl AsRef<Path>, name: &str) -> Result<RawTensor, TensorError> {
        let path = path.as_ref();
        let header = Header::from_file(path)?;
        let info = header
            .get(name)
            .ok_or_else(|| TensorError(format!("the checkpoint has no tensor {name:?}")))?;

        let mut file = File::open(path)
            .map_err(|e| TensorError(format!("cannot open {}: {e}", path.display())))?;
        file.seek(SeekFrom::Start(header.payload_start + info.start as u64))
            .map_err(|e| TensorError(format!("{name}: cannot seek to its values: {e}")))?;

        let mut raw = vec![0u8; info.end - info.start];
        file.read_exact(&mut raw)
            .map_err(|e| TensorError(format!("{name}: cannot read its values: {e}")))?;

        let data = decode(&info.dtype, &raw, info.count())
            .map_err(|e| TensorError(format!("{name} (shape {:?}): {e}", info.shape)))?;
        Ok(RawTensor {
            shape: info.shape.clone(),
            data,
        })
    }

    /// Read only the named tensors, seeking to each one's range.
    ///
    /// Between [`Self::load_one`] and [`Self::from_file`]: use it when a
    /// handful of tensors are needed and the rest are not. Loading just the
    /// embedding table and its norm costs 206 MB rather than 1.58 GB.
    pub fn load_subset(path: impl AsRef<Path>, names: &[&str]) -> Result<Self, TensorError> {
        let path = path.as_ref();
        let header = Header::from_file(path)?;
        let mut file = File::open(path)
            .map_err(|e| TensorError(format!("cannot open {}: {e}", path.display())))?;

        let mut tensors = HashMap::with_capacity(names.len());
        for &name in names {
            let info = header
                .get(name)
                .ok_or_else(|| TensorError(format!("the checkpoint has no tensor {name:?}")))?;
            file.seek(SeekFrom::Start(header.payload_start + info.start as u64))
                .map_err(|e| TensorError(format!("{name}: cannot seek to its values: {e}")))?;
            let mut raw = vec![0u8; info.end - info.start];
            file.read_exact(&mut raw)
                .map_err(|e| TensorError(format!("{name}: cannot read its values: {e}")))?;
            let data = decode(&info.dtype, &raw, info.count())
                .map_err(|e| TensorError(format!("{name} (shape {:?}): {e}", info.shape)))?;
            tensors.insert(
                name.to_string(),
                RawTensor {
                    shape: info.shape.clone(),
                    data,
                },
            );
        }
        Ok(Checkpoint { tensors })
    }

    /// Read the whole file. 1.58 GB of f32 for this checkpoint, so a test
    /// that calls this should share one instance across the binary.
    ///
    /// It reads one tensor at a time rather than the whole file at once, so
    /// the 791 MB raw buffer is never held alongside the decoded f32. Peak is
    /// the decoded 1.58 GB plus the largest single tensor's raw bytes.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TensorError> {
        let path = path.as_ref();
        let header = Header::from_file(path)?;
        let names: Vec<&str> = header.names().collect();
        Self::load_subset(path, &names)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TensorError> {
        if bytes.len() < 8 {
            return Err(TensorError(format!(
                "the file is {} bytes, too short to hold a header length",
                bytes.len()
            )));
        }
        let header_len = u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes")) as usize;
        let header_end = 8 + header_len;
        if bytes.len() < header_end {
            return Err(TensorError(format!(
                "the header claims {header_len} bytes but the file holds only {}",
                bytes.len() - 8
            )));
        }

        let entries = parse_entries(&bytes[8..header_end])?;

        // Offsets in the header are relative to the end of the header, not
        // the start of the file.
        let payload = &bytes[header_end..];

        let mut tensors = HashMap::with_capacity(entries.len());
        for (name, info) in entries {
            if info.end > payload.len() {
                return Err(TensorError(format!(
                    "{name}: byte range {}..{} is outside the {} byte payload",
                    info.start,
                    info.end,
                    payload.len()
                )));
            }
            let data = decode(&info.dtype, &payload[info.start..info.end], info.count())
                .map_err(|e| TensorError(format!("{name} (shape {:?}): {e}", info.shape)))?;
            tensors.insert(
                name,
                RawTensor {
                    shape: info.shape,
                    data,
                },
            );
        }

        Ok(Checkpoint { tensors })
    }

    pub fn get(&self, name: &str) -> Option<&RawTensor> {
        self.tensors.get(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Hand a tensor's values to the model, checking its shape as
    /// [`Self::require`] does.
    ///
    /// Rung 13 (feature `owned-weights`) moves the values out of the
    /// checkpoint, so the model's copy is the only one, and a loaded
    /// checkpoint is empty. Before rung 13 the values are cloned and the
    /// checkpoint keeps its own copy until it drops. That is two f32 copies
    /// of the weights at once, 3.1 GiB at the peak, and macOS's allocator
    /// then keeps about 0.9 GiB of the freed copy cached in the process
    /// (`results/memory/t4_quiet/SUMMARY.md`).
    pub fn take(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>, TensorError> {
        let tensor = self.require(name, shape)?;
        if !cfg!(feature = "owned-weights") {
            return Ok(tensor.data.clone());
        }
        let tensor = self
            .tensors
            .remove(name)
            .expect("require has just found it");
        Ok(tensor.data)
    }

    /// Fetch a tensor and check its shape, naming both when it is wrong.
    pub fn require(&self, name: &str, shape: &[usize]) -> Result<&RawTensor, TensorError> {
        let tensor = self.tensors.get(name).ok_or_else(|| {
            TensorError(format!(
                "the checkpoint has no tensor {name:?}. Note that the encoder tensors are \
                 prefixed `model.` while the head tensors are not."
            ))
        })?;
        if tensor.shape != shape {
            return Err(TensorError(format!(
                "{name}: shape is {:?}, expected {shape:?}",
                tensor.shape
            )));
        }
        Ok(tensor)
    }
}

/// Read the header JSON into one [`TensorInfo`] per tensor.
fn parse_entries(json: &[u8]) -> Result<HashMap<String, TensorInfo>, TensorError> {
    let header: Value = serde_json::from_slice(json)
        .map_err(|e| TensorError(format!("the header is not valid JSON: {e}")))?;
    let entries = header
        .as_object()
        .ok_or_else(|| TensorError("the header is not a JSON object".into()))?;

    let mut out = HashMap::with_capacity(entries.len());
    for (name, entry) in entries {
        // Written by the producer, not a tensor.
        if name == "__metadata__" {
            continue;
        }

        let dtype = entry
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| TensorError(format!("{name}: no dtype")))?
            .to_string();

        let shape: Vec<usize> = entry
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| TensorError(format!("{name}: no shape")))?
            .iter()
            .map(|n| n.as_u64().map(|v| v as usize))
            .collect::<Option<_>>()
            .ok_or_else(|| TensorError(format!("{name}: a shape entry is not a number")))?;

        let offsets = entry
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| TensorError(format!("{name}: no data_offsets")))?;
        if offsets.len() != 2 {
            return Err(TensorError(format!(
                "{name}: data_offsets has {} entries, expected 2",
                offsets.len()
            )));
        }
        let start = offsets[0].as_u64().unwrap_or(0) as usize;
        let end = offsets[1].as_u64().unwrap_or(0) as usize;
        if end < start {
            return Err(TensorError(format!(
                "{name}: data_offsets {start}..{end} runs backwards"
            )));
        }

        out.insert(
            name.clone(),
            TensorInfo {
                dtype,
                shape,
                start,
                end,
            },
        );
    }
    Ok(out)
}

fn decode(dtype: &str, raw: &[u8], count: usize) -> Result<Vec<f32>, String> {
    match dtype {
        "BF16" => upcast_bf16(raw, count),
        "F32" => read_f32(raw, count),
        other => Err(format!(
            "dtype {other} is not supported; this checkpoint should be entirely BF16"
        )),
    }
}

/// bf16 is the top 16 bits of an f32, so widening is a shift. Exact, with no
/// rounding mode to choose.
fn upcast_bf16(raw: &[u8], count: usize) -> Result<Vec<f32>, String> {
    if raw.len() != count * 2 {
        return Err(format!(
            "{count} bf16 values need {} bytes, the range holds {}",
            count * 2,
            raw.len()
        ));
    }
    let mut out = Vec::with_capacity(count);
    for pair in raw.chunks_exact(2) {
        let bits = u16::from_le_bytes([pair[0], pair[1]]);
        out.push(f32::from_bits((bits as u32) << 16));
    }
    Ok(out)
}

fn read_f32(raw: &[u8], count: usize) -> Result<Vec<f32>, String> {
    if raw.len() != count * 4 {
        return Err(format!(
            "{count} f32 values need {} bytes, the range holds {}",
            count * 4,
            raw.len()
        ));
    }
    let mut out = Vec::with_capacity(count);
    for quad in raw.chunks_exact(4) {
        out.push(f32::from_le_bytes([quad[0], quad[1], quad[2], quad[3]]));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A checkpoint holding one f32 tensor, `w`, of shape `[2]`.
    fn one_tensor() -> Checkpoint {
        let header = r#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&1.5f32.to_le_bytes());
        bytes.extend_from_slice(&(-2.0f32).to_le_bytes());
        Checkpoint::from_bytes(&bytes).expect("the synthetic checkpoint must parse")
    }

    #[test]
    fn take_moves_the_tensor_out_only_in_rung_13() {
        let mut checkpoint = one_tensor();
        assert_eq!(
            checkpoint.take("w", &[2]).expect("the shape is right"),
            [1.5, -2.0]
        );
        // Rung 13 leaves nothing behind; before it the checkpoint keeps its copy.
        assert_eq!(checkpoint.is_empty(), cfg!(feature = "owned-weights"));
    }

    #[test]
    fn take_checks_the_shape_and_keeps_the_tensor_when_it_is_wrong() {
        let mut checkpoint = one_tensor();
        let message = match checkpoint.take("w", &[3]) {
            Ok(_) => panic!("a wrong shape must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(
            message.contains("w: shape is [2], expected [3]"),
            "{message}"
        );
        assert!(checkpoint.get("w").is_some());
    }
}
