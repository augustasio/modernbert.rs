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
//! The file is read into owned memory rather than mapped. That is a
//! measurement decision, not a performance one: mapped pages count toward RSS
//! only once touched and can be evicted, so a mapping variant looks
//! artificially lean beside one that reads. mmap gets measured separately.

use std::collections::HashMap;
use std::path::Path;

use serde_json::Value;

/// A problem with the file. The message names the tensor, because a 791 MB
/// file and "unexpected end of input" together help nobody.
#[derive(Debug)]
pub struct TensorError(String);

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for TensorError {}

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

/// Every tensor in the file, by name.
pub struct Checkpoint {
    tensors: HashMap<String, RawTensor>,
}

impl Checkpoint {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TensorError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .map_err(|e| TensorError(format!("cannot read {}: {e}", path.display())))?;
        Self::from_bytes(&bytes)
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

        let header: Value = serde_json::from_slice(&bytes[8..header_end])
            .map_err(|e| TensorError(format!("the header is not valid JSON: {e}")))?;
        let entries = header
            .as_object()
            .ok_or_else(|| TensorError("the header is not a JSON object".into()))?;

        // Offsets in the header are relative to the end of the header, not
        // the start of the file.
        let payload = &bytes[header_end..];

        let mut tensors = HashMap::with_capacity(entries.len());
        for (name, entry) in entries {
            // Written by the producer, not a tensor.
            if name == "__metadata__" {
                continue;
            }

            let dtype = entry
                .get("dtype")
                .and_then(Value::as_str)
                .ok_or_else(|| TensorError(format!("{name}: no dtype")))?;

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
            if end < start || end > payload.len() {
                return Err(TensorError(format!(
                    "{name}: byte range {start}..{end} is outside the {} byte payload",
                    payload.len()
                )));
            }

            let count: usize = shape.iter().product();
            let raw = &payload[start..end];
            let data = match dtype {
                "BF16" => upcast_bf16(raw, count)
                    .map_err(|e| TensorError(format!("{name} (shape {shape:?}): {e}")))?,
                "F32" => upcast_f32(raw, count)
                    .map_err(|e| TensorError(format!("{name} (shape {shape:?}): {e}")))?,
                other => {
                    return Err(TensorError(format!(
                        "{name}: dtype {other} is not supported; this checkpoint should be \
                         entirely BF16"
                    )))
                }
            };

            tensors.insert(name.clone(), RawTensor { shape, data });
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

fn upcast_f32(raw: &[u8], count: usize) -> Result<Vec<f32>, String> {
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
