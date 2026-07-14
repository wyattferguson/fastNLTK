#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        // CCG category parsing must never panic on arbitrary input
        let _ = fastnltk::ccg::parse_category(s);
    }
});
