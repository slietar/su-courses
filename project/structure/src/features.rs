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
    #[serde(flatten)]
    pub kind: DomainKind,

    pub name: String,
    pub range: (usize, usize),
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind")]
pub enum DomainKind {
    EGFLike,
    EGFLikeCalciumBinding,
    TB,
}


pub fn process_domains(path: &str) -> Result<Vec<Domain>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let entry: UniProtEntry = serde_json::from_reader(reader)?;
    let domains = entry.features.iter()
        .filter(|feature| feature.feature_type == "Domain")
        .map(|feature| {
            let kind = if feature.description.ends_with("; calcium-binding") {
                DomainKind::EGFLikeCalciumBinding
            } else if feature.description.starts_with("TB ") {
                DomainKind::TB
            } else {
                DomainKind::EGFLike
            };

            Domain {
                kind,
                name: feature.description.clone(),
                range: (feature.location.start.value, feature.location.end.value)
            }
        })
        .collect::<Vec<_>>();

    Ok(domains)
}
