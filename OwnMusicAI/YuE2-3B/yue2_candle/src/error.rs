#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("config.json: {0}")]
    Config(#[from] serde_json::Error),
    #[error("{0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, EngineError>;
