use pyo3::prelude::*;

#[pyfunction]
fn accelerated_available() -> bool {
    true
}

#[pyfunction]
fn _accelerated(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accelerated_available, m)?)?;
    Ok(())
}
