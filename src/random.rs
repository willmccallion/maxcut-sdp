//! Standard normal samples for the BM start and the rounding hyperplanes.

use std::f64::consts::TAU;

/// The Box-Muller transform: two independent standard normal samples from
/// two independent uniform samples `u1, u2` in `[0, 1)`.
pub fn box_muller(u1: f64, u2: f64) -> (f64, f64) {
    let magnitude = (-2.0 * u1.ln()).sqrt();
    let angle = TAU * u2;
    (magnitude * angle.cos(), magnitude * angle.sin())
}
