//! Read a 2-D little-endian f32 array from NumPy's `.npy` format.
//!
//! Only what the harness needs: rung 0's logits, `[7600, 4]`, C order. Any
//! other layout is an error rather than a guess.

use std::path::Path;

pub struct Array2 {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Array2 {
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }
}

pub fn read(path: &Path) -> Result<Array2, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(format!("{} is not a .npy file", path.display()));
    }
    let (header_len, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
        v => return Err(format!("unsupported .npy version {v}")),
    };
    let header = std::str::from_utf8(&bytes[start..start + header_len])
        .map_err(|_| "the .npy header is not text".to_string())?;

    if !header.contains("'descr': '<f4'") {
        return Err(format!(
            "expected little-endian f32 ('<f4'), header is {header}"
        ));
    }
    if !header.contains("'fortran_order': False") {
        return Err("expected C order, found Fortran order".to_string());
    }
    let shape_text = header
        .split("'shape': (")
        .nth(1)
        .and_then(|rest| rest.split(')').next())
        .ok_or_else(|| format!("no shape in header {header}"))?;
    let dims: Vec<usize> = shape_text
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().map_err(|_| format!("bad dimension {s:?}")))
        .collect::<Result<_, _>>()?;
    let [rows, cols] = dims[..] else {
        return Err(format!("expected a 2-D array, shape is {dims:?}"));
    };

    let payload = &bytes[start + header_len..];
    if payload.len() != rows * cols * 4 {
        return Err(format!(
            "shape {rows}x{cols} needs {} bytes, the file has {}",
            rows * cols * 4,
            payload.len()
        ));
    }
    let data = payload
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok(Array2 { rows, cols, data })
}
