//! NLTK data file loading.
//!
//! Resolves nltk_data paths and loads serialized models (pickle, bincode).
//! Compatible with NLTK's data directory structure.

use once_cell::sync::Lazy;
use std::path::PathBuf;

/// Search paths for nltk_data, computed once at first use.
static DATA_SEARCH_PATHS: Lazy<Vec<PathBuf>> = Lazy::new(|| {
    let mut paths = Vec::new();

    // 1. NLTK_DATA env var
    if let Ok(val) = std::env::var("NLTK_DATA") {
        paths.push(PathBuf::from(val));
    }

    // 2. User home directory
    if let Some(home) = dirs_data_dir() {
        paths.push(home);
    }

    // 3. Common system paths
    if cfg!(windows) {
        paths.push(PathBuf::from(r"C:\nltk_data"));
    } else {
        paths.push(PathBuf::from("/usr/share/nltk_data"));
        paths.push(PathBuf::from("/usr/local/share/nltk_data"));
    }

    paths
});

#[cfg(not(windows))]
fn dirs_data_dir() -> Option<PathBuf> {
    std::env::var("HOME").ok().map(|home| {
        let mut p = PathBuf::from(home);
        p.push("nltk_data");
        p
    })
}

#[cfg(windows)]
fn dirs_data_dir() -> Option<PathBuf> {
    std::env::var("USERPROFILE").ok().map(|home| {
        let mut p = PathBuf::from(home);
        p.push("nltk_data");
        p
    })
}

/// Find an NLTK resource file by name.
pub fn find_resource(name: &str) -> Result<PathBuf, String> {
    for base in DATA_SEARCH_PATHS.iter() {
        let candidate = base.join(name);
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    Err(format!("NLTK resource not found: {name}"))
}

/// Find a directory for an NLTK resource (e.g., tagger directory).
pub fn find_resource_dir(name: &str) -> Result<PathBuf, String> {
    for base in DATA_SEARCH_PATHS.iter() {
        let candidate = base.join(name);
        if candidate.is_dir() {
            return Ok(candidate);
        }
    }
    Err(format!("NLTK resource directory not found: {name}"))
}

/// Resolve nltk_data to a bincode cache path.
pub fn bincode_cache_path(resource_name: &str) -> PathBuf {
    let sanitized = resource_name.replace('/', "_").replace('.', "_");
    let mut cache = std::env::temp_dir();
    cache.push("fastnltk_cache");
    cache.push(format!("{}.bin", sanitized));
    cache
}
