mod features;
mod gn;
mod mutations;
mod output;
mod plddt;
mod structures;
mod variants;


use std::error::Error;

pub use self::output::Output;

pub fn deserialize(path: &str) -> Result<self::output::Output, Box<dyn Error>> {
    let bytes = std::fs::read(path)?;
    Ok(serde_pickle::from_slice(&bytes, Default::default())?)
}
