//! Model servers for asset-studio: a **catalog** of the known Python model
//! servers and a **typed HTTP client** to talk to them once they're running.
//!
//! Model servers are *separate Python processes* (each a `uv`/`python` server
//! from the reference `model_servers/`), started by the
//! [`asset-studio-launcher`](../asset_studio_launcher/index.html) and reached
//! over plain HTTP on a stable port. This crate does not embed a Python
//! interpreter; it only knows *what* servers exist (`catalog`) and *how* to
//! call one (`client`).
pub mod client;
pub mod catalog;

pub use catalog::{all_servers, find_server, ModelServer};
pub use client::ModelClient;
