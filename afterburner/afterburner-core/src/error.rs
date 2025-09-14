pub type AbResult<T> = Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Shape Missmatch")]
    ShapeMissmatch,
    #[error("Invalid Dimension")]
    InvalidDimension,
    #[error("Invalid Parameter")]
    InvalidParameter,
}
