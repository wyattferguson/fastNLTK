//! Association measures — Rust wrappers around nltk_metrics::association.
//!
//! BigramAssocMeasures, TrigramAssocMeasures, QuadgramAssocMeasures
//! with PMI, chi-square, likelihood ratio, Dice, etc.
//!
//! NLTK equivalent: nltk.metrics.association

use pyo3::prelude::*;

#[pyclass(name = "BigramAssocMeasures", module = "fastnltk._rust")]
pub struct BigramAssocMeasures;

#[pymethods]
impl BigramAssocMeasures {
    #[staticmethod]
    fn pmi(_count: f64, n: f64, n_ii: f64, n_ix: f64, n_xi: f64) -> f64 {
        // PMI = log( (n_ii * N) / (n_ix * n_xi) ) / log(2)
        let expected = (n_ix * n_xi) / n;
        if expected <= 0.0 || n_ii <= 0.0 {
            return 0.0;
        }
        (n_ii / expected).log2()
    }

    #[staticmethod]
    fn chi_sq(_count: f64, n: f64, n_ii: f64, n_ix: f64, n_xi: f64) -> f64 {
        let n_oi = n_ix - n_ii;
        let n_io = n_xi - n_ii;
        let n_oo = n - n_ii - n_oi - n_io;
        if n_oi < 0.0 || n_io < 0.0 || n_oo < 0.0 {
            return 0.0;
        }
        let num = n * (n_ii * n_oo - n_io * n_oi).powi(2);
        let den = n_ix * n_xi * (n - n_ix) * (n - n_xi);
        if den == 0.0 {
            return 0.0;
        }
        num / den
    }

    #[staticmethod]
    fn likelihood_ratio(_count: f64, n: f64, n_ii: f64, n_ix: f64, n_xi: f64) -> f64 {
        let n_oi = n_ix - n_ii;
        let n_io = n_xi - n_ii;
        let n_oo = n - n_ii - n_oi - n_io;
        if n_oi < 0.0 || n_io < 0.0 || n_oo < 0.0 {
            return 0.0;
        }
        let mut ll = 0.0;
        // log-likelihood = 2 * sum of (observed * log(observed / expected))
        if n_ii > 0.0 {
            let e_ii = (n_ix * n_xi) / n;
            ll += n_ii * (n_ii / e_ii).ln();
        }
        if n_oi > 0.0 {
            let e_oi = (n_ix * (n - n_xi)) / n;
            ll += n_oi * (n_oi / e_oi).ln();
        }
        if n_io > 0.0 {
            let e_io = ((n - n_ix) * n_xi) / n;
            ll += n_io * (n_io / e_io).ln();
        }
        if n_oo > 0.0 {
            let e_oo = ((n - n_ix) * (n - n_xi)) / n;
            ll += n_oo * (n_oo / e_oo).ln();
        }
        2.0 * ll
    }

    #[staticmethod]
    fn dice(n_ii: f64, n_ix: f64, n_xi: f64) -> f64 {
        if n_ix + n_xi == 0.0 {
            return 0.0;
        }
        2.0 * n_ii / (n_ix + n_xi)
    }
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BigramAssocMeasures>()?;
    Ok(())
}
