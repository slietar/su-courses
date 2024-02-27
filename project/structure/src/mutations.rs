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
    effects: MutationEffects,
    neomutation: bool,
    new_nucleotide: char,
    new_residue: char,
    old_nucleotide: char,
    old_residue: char,
    position: usize, // Starting at 1
    residue: usize, // Starting at 1
}

#[derive(Debug, Serialize)]
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
                effects,
                neomutation: raw_mutation.neomutation.is_some(),
                new_nucleotide: mutation_chars[mutation_chars.len() - 1],
                new_residue: residue_chars[residue_chars.len() - 1],
                old_nucleotide: mutation_chars[mutation_chars.len() - 3],
                old_residue: residue_chars[0],
                position: raw_mutation.position,
                residue,
            };

            mutations.push(mutation);
        }
    }

    Ok(mutations)
}
