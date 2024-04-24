use std::{error::Error, fs::File, io::BufReader};

use serde::{Deserialize, Serialize};


#[derive(Debug, Deserialize)]
struct RawMutation {
    #[serde(rename = "Cardio")]
    effect_cardio: Option<usize>,

    #[serde(rename = "cutané")]
    effect_cutaneous: Option<String>,

    #[serde(rename = "neuro")]
    effect_neuro: Option<String>,

    #[serde(rename = "Ophtalmo")]
    effect_ophtalmo: Option<usize>,

    #[serde(rename = "Pneumothorax")]
    effect_pneumothorax: Option<String>,

    #[serde(rename = "Sévère ou jeune")]
    effect_severe: Option<String>,

    #[serde(rename = "SK")]
    effect_sk: Option<usize>,

    // #[serde(rename = "Clinique")]
    // effects: String,

    #[serde(rename = "Mutation c.")]
    mutation: String,

    #[serde(rename = "Néomutation?")]
    neomutation: Option<String>,

    #[serde(rename = "nucléotide")]
    position: usize,

    #[serde(rename = "p.")]
    residue: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Mutation {
    effect_cardio: usize,
    effect_cutaneous: bool,
    effect_ophtalmo: usize,
    effect_neuro: bool,
    effect_pneumothorax: bool,
    effect_severe: bool,
    effect_sk: usize,

    genomic_position: usize, // Starting at 1
    name: String,
    neomutation: bool,
    position: usize, // Starting at 1

    reference_aa: char,
    alternate_aa: char,

    reference_nucleotides: String,
    alternate_nucleotides: String,
}

pub fn process_mutations() -> Result<Vec<Mutation>, Box<dyn Error>> {
    let file = File::open("../sources/hospital/mutations.csv")?;
    let reader = BufReader::new(file);
    let mut csv_reader = csv::Reader::from_reader(reader);

    let mut mutations = Vec::new();

    for row in csv_reader.deserialize() {
        let raw_mutation: RawMutation = row?;
        let raw_residue = raw_mutation.residue.trim();
        let mutation_chars = raw_mutation.mutation.chars().collect::<Vec<_>>();
        let residue_chars = raw_residue.chars().collect::<Vec<_>>();

        if let Ok(residue) = raw_residue[1..(raw_residue.len() - 1)].parse::<usize>() {
            let mutation = Mutation {
                alternate_aa: residue_chars[residue_chars.len() - 1],
                alternate_nucleotides: mutation_chars[mutation_chars.len() - 1].to_string(),
                effect_cardio: raw_mutation.effect_cardio.unwrap_or(0),
                effect_cutaneous: raw_mutation.effect_cutaneous.is_some(),
                effect_neuro: raw_mutation.effect_neuro.is_some(),
                effect_ophtalmo: raw_mutation.effect_ophtalmo.unwrap_or(0),
                effect_pneumothorax: raw_mutation.effect_pneumothorax.is_some(),
                effect_severe: raw_mutation.effect_severe.is_some(),
                effect_sk: raw_mutation.effect_sk.unwrap_or(0),
                genomic_position: raw_mutation.position,
                name: raw_mutation.mutation,
                neomutation: raw_mutation.neomutation.is_some(),
                position: residue,
                reference_aa: residue_chars[0],
                reference_nucleotides: mutation_chars[mutation_chars.len() - 3].to_string(),
            };

            mutations.push(mutation);
        }
    }

    Ok(mutations)
}
