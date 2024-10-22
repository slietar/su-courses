use std::{error::Error, fmt::Display, fs::File, io::BufReader};
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

#[derive(Debug, Deserialize, Serialize)]
pub struct Domain {
    #[serde(flatten)]
    pub kind: DomainKind,
    pub global_index: usize,

    pub name: String,
    pub number: usize,

    pub start_position: usize,
    pub end_position: usize,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(tag = "kind")]
pub enum DomainKind {
    #[serde(rename = "EGF")]
    EGFLike,

    #[serde(rename = "EGFCB")]
    EGFLikeCalciumBinding,

    TB,
}

impl Display for DomainKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DomainKind::EGFLike => write!(f, "EGF"),
            DomainKind::EGFLikeCalciumBinding => write!(f, "EGFCB"),
            DomainKind::TB => write!(f, "TB"),
        }
    }
}


pub const DOMAIN_KINDS: &'static [&str; 3] = &[
    "EGF",
    "EGFCB",
    "TB",
];


pub fn process_domains(path: &str) -> Result<Vec<Domain>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let entry: UniProtEntry = serde_json::from_reader(reader)?;
    let domains = entry.features.iter()
        .filter(|feature| feature.feature_type == "Domain")
        .enumerate()
        .map(|(global_index, feature)| {
            let name = &feature.description;

            let (kind, raw_number) = if name.ends_with("; calcium-binding") {
                (DomainKind::EGFLikeCalciumBinding, &name["EGF-like ".len()..(name.len() - "; calcium-binding".len())])
            } else if name.starts_with("TB ") {
                (DomainKind::TB, &name["TB ".len()..])
            } else {
                (DomainKind::EGFLike, &name["EGF-like ".len()..])
            };

            let number = raw_number.parse()?;

            Ok(Domain {
                kind,
                global_index,

                name: format!("{} {}", &kind, number),
                number,

                start_position: feature.location.start.value,
                end_position: feature.location.end.value,
            })
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;

    Ok(domains)
}
