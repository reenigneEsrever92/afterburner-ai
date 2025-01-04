pub trait Backend: Clone {}

#[derive(Clone)]
pub struct NoBackend;

impl Backend for NoBackend {}
