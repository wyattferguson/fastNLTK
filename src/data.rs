//! NLTK data file loading.

use pyo3::exceptions::PyLookupError;
use pyo3::prelude::*;
use std::path::PathBuf;
use std::sync::LazyLock;

/// Search paths for `nltk_data`, computed once at first use.
static DATA_SEARCH_PATHS: LazyLock<Vec<PathBuf>> = LazyLock::new(|| {
    let mut paths = Vec::new();

    // 1. NLTK_DATA env var
    if let Ok(val) = std::env::var("NLTK_DATA") {
        for p in std::env::split_paths(&val) {
            paths.push(p);
        }
    }

    // 2. User home directory
    if let Some(home) = dirs_data_dir() {
        paths.push(home);
    }

    // 3. Common system paths + sys.prefix-based paths
    if cfg!(windows) {
        paths.push(PathBuf::from(r"C:\nltk_data"));
        if let Ok(appdata) = std::env::var("APPDATA") {
            paths.push(PathBuf::from(appdata).join("nltk_data"));
        }
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
    std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .ok()
        .map(|home| {
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

/// Resolve `nltk_data` to a bincode cache path.
#[must_use]
pub fn bincode_cache_path(resource_name: &str) -> PathBuf {
    let sanitized = resource_name.replace(['/', '.'], "_");
    let mut cache = std::env::temp_dir();
    cache.push("fastnltk_cache");
    cache.push(format!("{sanitized}.bin"));
    cache
}

// ── Python-visible functions ───────────────────────────────────────────────

/// `find(name)` — resolve an NLTK resource name to an absolute path.
#[pyfunction]
fn find(name: &str) -> PyResult<String> {
    find_resource(name)
        .map(|p| p.to_string_lossy().to_string())
        .map_err(|e| PyLookupError::new_err(e))
}

/// Register all data functions with the Python module.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find, m)?)?;
    Ok(())
}
