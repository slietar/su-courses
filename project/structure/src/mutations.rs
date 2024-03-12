use std::{error::Error, fs::File, io::BufReader};

use serde::{Deserialize, Serialize};


#[derive(Debug, Deserialize)]
struct RawMutation {
    #[serde(rename = "Clinique")]
    effects: String,

    #[serde(rename = "Mutation c.")]
    mutation: String,

    #[serde(rename = "Néomutation?")]
    neomutation: Option<String>,

    #[serde(rename = "nucléotide")]
    position: usize,

    #[serde(rename = "p.")]
    residue: String,
}

#[derive(Debug, Serialize)]
pub struct Mutation {
    effects: u32,
    genomic_position: usize, // Starting at 1
    name: String,
    neomutation: bool,
    position: usize, // Starting at 1

    reference_aa: char,
    alternate_aa: char,

    reference_nucleotides: String,
    alternate_nucleotides: String,
}

#[derive(Debug)]
#[allow(non_snake_case)]
pub struct MutationEffects {
    AAA: bool,
    Ectopia: bool,
    MFSClassic: bool,
    MFSFull: bool,
    MFSWithoutEye: bool,
    PVM: bool,
    SK: bool,
    TAA: bool,
}

impl std::convert::From<MutationEffects> for u32 {
    fn from(effects: MutationEffects) -> u32 {
        return (1 << 0) * (effects.AAA as u32)
            + (1 << 1) * (effects.Ectopia as u32)
            + (1 << 2) * (effects.MFSClassic as u32)
            + (1 << 3) * (effects.MFSFull as u32)
            + (1 << 4) * (effects.MFSWithoutEye as u32)
            + (1 << 5) * (effects.PVM as u32)
            + (1 << 6) * (effects.SK as u32)
            + (1 << 7) * (effects.TAA as u32);
    }
}


pub const EFFECT_LABELS: &'static [&str; 8] = &[
    "AAA",
    "Ectopia",
    "Classic MFS",
    "Full MFS",
    "MFS without eye",
    "PVM",
    "SK",
    "TAA",
];


pub fn process_mutations(path: &str) -> Result<Vec<Mutation>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut csv_reader = csv::Reader::from_reader(reader);

    let mut mutations = Vec::new();

    for result in csv_reader.deserialize() {
        let raw_mutation: RawMutation = result?;
        let raw_residue = raw_mutation.residue.trim();
        let mutation_chars = raw_mutation.mutation.chars().collect::<Vec<_>>();
        let residue_chars = raw_residue.chars().collect::<Vec<_>>();

        let raw_effects = raw_mutation.effects.to_lowercase();

        let effects = MutationEffects {
            AAA: raw_effects.contains("aaa"),
            MFSClassic: raw_effects.contains("classique"),
            Ectopia: raw_effects.contains("ectopie") && !raw_effects.contains("sans ectopie"),
            MFSFull: raw_effects.contains("mfs complet"),
            MFSWithoutEye: raw_effects.contains("sans oeuil") || raw_effects.contains("sans oeil") || raw_effects.contains("sans œil"),
            PVM: raw_effects.contains("pvm"),
            SK: raw_effects.contains("sk"),
            TAA: raw_effects.contains("taa"),
        };

        if let Ok(residue) = raw_residue[1..(raw_residue.len() - 1)].parse::<usize>() {
            let mutation = Mutation {
                alternate_aa: residue_chars[residue_chars.len() - 1],
                alternate_nucleotides: mutation_chars[mutation_chars.len() - 1].to_string(),
                effects: effects.into(),
                name: raw_mutation.mutation,
                neomutation: raw_mutation.neomutation.is_some(),
                genomic_position: raw_mutation.position,
                reference_aa: residue_chars[0],
                reference_nucleotides: mutation_chars[mutation_chars.len() - 3].to_string(),
                position: residue,
            };

            mutations.push(mutation);
        }
    }

    Ok(mutations)
}
