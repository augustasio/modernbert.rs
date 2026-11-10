//! The least HTTP/1.1 that rung 21 needs: one message with a `Content-Length`
//! body, read or written on a kept-alive connection. Written here, with no
//! library, because the server's whole job is a few lines of framing, and a
//! framework would hide the costs this rung exists to measure.
//!
//! Trap: both ends must set `TCP_NODELAY`. Without it, Nagle's algorithm holds
//! a small reply back until the peer's delayed ACK, up to 40 ms on Linux, which
//! is a fifth of the latency budget spent doing nothing.

use std::io::{BufRead, Write};

/// One request or response: its first line and its body.
pub struct Message {
    pub start_line: String,
    pub body: Vec<u8>,
}

/// Read one message. `Ok(None)` when the peer closed the connection cleanly
/// between messages.
pub fn read_message(reader: &mut impl BufRead) -> Result<Option<Message>, String> {
    let mut start_line = String::new();
    if reader
        .read_line(&mut start_line)
        .map_err(|e| format!("read: {e}"))?
        == 0
    {
        return Ok(None);
    }
    let mut length = 0usize;
    loop {
        let mut header = String::new();
        if reader
            .read_line(&mut header)
            .map_err(|e| format!("read: {e}"))?
            == 0
        {
            return Err("the connection closed inside the headers".to_string());
        }
        let header = header.trim_end();
        if header.is_empty() {
            break;
        }
        if let Some((name, value)) = header.split_once(':') {
            if name.eq_ignore_ascii_case("content-length") {
                length = value
                    .trim()
                    .parse()
                    .map_err(|_| format!("bad Content-Length {value:?}"))?;
            }
        }
    }
    let mut body = vec![0u8; length];
    reader
        .read_exact(&mut body)
        .map_err(|e| format!("read body: {e}"))?;
    Ok(Some(Message {
        start_line: start_line.trim_end().to_string(),
        body,
    }))
}

/// Write one message with a JSON body, in a single write so it leaves as one
/// segment.
pub fn write_message(writer: &mut impl Write, start_line: &str, body: &[u8]) -> Result<(), String> {
    let mut out = format!(
        "{start_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        body.len()
    )
    .into_bytes();
    out.extend_from_slice(body);
    writer
        .write_all(&out)
        .and_then(|_| writer.flush())
        .map_err(|e| format!("write: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufReader;

    #[test]
    fn a_written_message_reads_back() {
        let mut wire = Vec::new();
        write_message(&mut wire, "POST /classify HTTP/1.1", b"{\"ids\":[1,2]}").unwrap();
        write_message(&mut wire, "HTTP/1.1 200 OK", b"{}").unwrap();
        let mut reader = BufReader::new(&wire[..]);
        let first = read_message(&mut reader).unwrap().unwrap();
        assert_eq!(first.start_line, "POST /classify HTTP/1.1");
        assert_eq!(first.body, b"{\"ids\":[1,2]}");
        let second = read_message(&mut reader).unwrap().unwrap();
        assert_eq!(second.start_line, "HTTP/1.1 200 OK");
        assert!(read_message(&mut reader).unwrap().is_none());
    }
}
