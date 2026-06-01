//! Retina Module
//!
//! Biological model of the retina converting light to spike trains.
//! Uses the same ribbon synapse mechanism as the cochlea.
//!
//! Architecture:
//! ```text
//! Light → Photoreceptors → Bipolar cells → Ganglion cells → Spikes
//!              ↓               ↓
//!         (cone/rod)      (ON/OFF)
//!              ↓               ↓
//!      Horizontal cells   Amacrine cells
//!      (lateral inhib)    (temporal)
//! ```

pub mod photoreceptor;
pub mod ganglion;
pub mod model;

pub use photoreceptor::{Photoreceptor, PhotoreceptorType, ConeType};
pub use ganglion::{GanglionCell, GanglionCellType, ReceptiveField};
pub use model::{RetinaModel, RetinaConfig, RetinaOutput};
