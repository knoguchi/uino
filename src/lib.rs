//! Uino (初乃): Biologically-inspired sensory processing.
//!
//! Predictive coding from periphery to cortex.
//! See uino_再設計計画書.md for the design plan.

pub mod ribbon_synapse;
pub mod spike_generator;
pub mod retina;
pub mod cortex;

pub use ribbon_synapse::{RibbonSynapse, RibbonSynapseConfig, PowerLawState};

pub use retina::{RetinaModel, RetinaConfig, RetinaOutput};
pub use retina::ganglion::{GanglionCell, GanglionCellType, ReceptiveField};
pub use retina::photoreceptor::{Photoreceptor, PhotoreceptorType, ConeType};

pub use cortex::{
    CorticalModel, CorticalConfig, CorticalOutput, CorticalModelBuilder,
    ThalamicRelay, A1Core, Belt, ThetaGammaOscillator, RewardModule,
    STRF,
};
