use std::error::Error;
use std::fmt;
use std::fmt::Formatter;

#[derive(Debug, Clone)]
pub enum TensorError {
    InvalidShapeError,
    BroadcastError,
    OperationError,
    DimensionError,
    IndexError,
}

impl Error for TensorError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::InvalidShapeError => write!(f, "invaid shape values"),
            TensorError::BroadcastError => write!(f, "tensors cannot be broadcasted"),
            TensorError::OperationError => write!(f, "operation cannot be applied"),
            TensorError::DimensionError => write!(f, "wrong dimension"),
            TensorError::IndexError => write!(f, "wrong logical indexes"),
        }
    }
}
