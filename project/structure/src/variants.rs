use std::{collections::HashMap, error::Error, fs::File, io::BufReader};

use serde::{Deserialize, Serialize};


#[derive(Debug, Deserialize)]
struct RawVariant {
    // #[serde(rename = "Position")]
    // genome_position: usize,

    #[serde(rename = "Allele Frequency")]
    frequency: f32,

    #[serde(rename = "Protein Consequence")]
    protein_effect: Option<String>,

    // #[serde(flatten, rename = "ClinVar Clinical Significance")]
    #[serde(flatten)]
    clinical_effect: Option<RawClinicalEffect>,

    // #[serde(rename = "Reference")]
    // reference: String,

    // #[serde(rename = "Alternate")]
    // alternate: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "ClinVar Clinical Significance")]
enum RawClinicalEffect {
    #[serde(rename = "Benign")]
    Benign,

    #[serde(rename = "Benign/Likely benign")]
    BenignLikelyBenign,

    #[serde(rename = "Conflicting interpretations of pathogenicity")]
    Conflicting,

    #[serde(rename = "Likely benign")]
    LikelyBenign,

    #[serde(rename = "Likely pathogenic")]
    LikelyPathogenic,

    #[serde(rename = "Pathogenic")]
    Pathogenic,

    #[serde(rename = "Pathogenic/Likely pathogenic")]
    PathogenicLikelyPathogenic,

    #[serde(rename = "Uncertain significance")]
    Uncertain,
}


#[derive(Debug, Serialize)]
pub struct Variant {
    alternate_residue: Option<char>,
    clinical_effect: Option<isize>,
    frequency: f32,
    protein_position: usize,
    reference_residue: char,
}

#[derive(Debug)]
pub struct VariantData {
    pub pathogenicity_labels: Vec<&'static str>,
    pub variants: Vec<Variant>,
}


pub fn process_variants(path: &str) -> Result<VariantData, Box<dyn Error>> {
    let amino_acids = HashMap::from([
        ("Ala", 'A'),
        ("Arg", 'R'),
        ("Asn", 'N'),
        ("Asp", 'D'),
        ("Cys", 'C'),
        ("Gln", 'Q'),
        ("Glu", 'E'),
        ("Gly", 'G'),
        ("His", 'H'),
        ("Ile", 'I'),
        ("Leu", 'L'),
        ("Lys", 'K'),
        ("Met", 'M'),
        ("Phe", 'F'),
        ("Pro", 'P'),
        ("Ser", 'S'),
        ("Thr", 'T'),
        ("Trp", 'W'),
        ("Tyr", 'Y'),
        ("Val", 'V'),
    ]);

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut csv_reader = csv::Reader::from_reader(reader);

    let mut variants = Vec::new();

    for result in csv_reader.deserialize() {
        let raw_variant: RawVariant = result?;

        if let Some(protein_effect) = raw_variant.protein_effect {
            // let chars = protein_effect.chars().collect::<Vec<_>>();
            // chars.slice(0, 3)

            let reference_long = &protein_effect[2..5];
            let reference_short = amino_acids.get(reference_long);

            let alternate_long = &protein_effect[(protein_effect.len() - 3)..];
            let alternate_short = amino_acids.get(alternate_long);

            let opt_protein_position: Option<usize> = protein_effect[5..(protein_effect.len() - 3)].parse().ok();

            // eprintln!("{:?} {:?} {:?}", reference_long, alternate_long, opt_protein_position);

            let (reference_residue, alternate_residue, protein_position) = match (reference_short, alternate_long, alternate_short, opt_protein_position) {
                (Some(&reference), "del", _, Some(position)) => {
                    (reference, None, position)
                },
                (Some(&reference), _, Some(&alternate), Some(position)) => {
                    (reference, Some(alternate), position)
                },
                _ => {
                    continue;
                }
            };

            // eprintln!("{:?} {:?} {:?}", reference_residue, alternate_residue, protein_position);

            // if let Some(reference_short) = amino_acids.get(reference_long) {
            //     let alternate_short = amino_acids.get(alternate_long);

            //     if alternate_long == "del" {
            //     }
            // }

            let clinical_effect = match raw_variant.clinical_effect {
                Some(RawClinicalEffect::Benign)
                    => Some(7),
                Some(RawClinicalEffect::BenignLikelyBenign)
                    => Some(6),
                Some(RawClinicalEffect::LikelyBenign)
                    => Some(5),
                Some(RawClinicalEffect::Conflicting)
                    => Some(4),
                Some(RawClinicalEffect::LikelyPathogenic)
                    => Some(3),
                Some(RawClinicalEffect::PathogenicLikelyPathogenic)
                    => Some(2),
                Some(RawClinicalEffect::Pathogenic)
                    => Some(1),
                Some(RawClinicalEffect::Uncertain) | None
                    => Some(0),
            };

            variants.push(Variant {
                alternate_residue,
                clinical_effect,
                frequency: raw_variant.frequency,
                protein_position,
                reference_residue,
            });
        }
    }

    // eprintln!("{:#?}", variants);

    Ok(VariantData {
        pathogenicity_labels: [
            "Uncertain significance",
            "Pathogenic",
            "Pathogenic/Likely pathogenic",
            "Likely pathogenic",
            "Conflicting interpretations of pathogenicity",
            "Benign/Likely benign",
            "Likely benign",
            "Benign",
        ].to_vec(),
        variants,
    })
}
