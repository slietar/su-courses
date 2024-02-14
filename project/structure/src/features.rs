use std::{error::Error, fs::File, io::BufReader};
use serde::{Deserialize, Serialize};


#[derive(Debug, Deserialize)]
struct UniProtEntry {
    features: Vec<UniProtFeature>,
}

#[derive(Debug, Deserialize)]
struct UniProtFeature {
    description: String,
    location: UniProtLocation,

    #[serde(rename = "type")]
    feature_type: String,
}

#[derive(Debug, Deserialize)]
struct UniProtLocation {
    start: UniProtPosition,
    end: UniProtPosition,
}

#[derive(Debug, Deserialize)]
struct UniProtPosition {
    value: usize,
}

#[derive(Debug, Serialize)]
pub struct Domain {
    name: String,
    range: (usize, usize),
}


pub fn process_domains(path: &str) -> Result<Vec<Domain>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let entry: UniProtEntry = serde_json::from_reader(reader)?;
    let domains = entry.features.iter()
        .filter(|feature| feature.feature_type == "Domain")
        .map(|feature| {
            Domain {
                name: feature.description.clone(),
                range: (feature.location.start.value, feature.location.end.value)
            }
        })
        .collect::<Vec<_>>();

    Ok(domains)
}
