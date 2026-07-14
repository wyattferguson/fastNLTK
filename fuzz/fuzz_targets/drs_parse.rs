#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        // DRS parser must never panic on arbitrary input
        let _ = fastnltk::drt::DRS::from_string(s);
    }
});
