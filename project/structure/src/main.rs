use std::{error::Error, fs::File, io::BufReader};

use pdbtbx::*;
use serde::Deserialize;


// fn extract_region(residues: &[&Residue], name: &str, start: usize, end: usize) -> Chain {
//     Chain::from_iter(name, residues[(start - 1)..(end - 1)].iter().map(|residue| { (*residue).clone() })).unwrap()
// }


#[derive(Debug, Deserialize)]
struct UniProtEntry {
    features: Vec<UniProtFeature>
}

#[derive(Debug, Deserialize)]
struct UniProtFeature {
    description: String,
    location: UniProtLocation,

    #[serde(rename = "type")]
    feature_type: String
}

#[derive(Debug, Deserialize)]
struct UniProtLocation {
    start: UniProtPosition,
    end: UniProtPosition
}

#[derive(Debug, Deserialize)]
struct UniProtPosition {
    value: usize
}

// #[derive(Debug)]
// struct Region {
//     name: String,
//     range: (usize, usize)
// }

fn process_pdb() -> Result<(), Box<dyn Error>> {
    let (input_pdb, _errors) = pdbtbx::open(
        "../drive/FBN1_AlphaFold.pdb",
        StrictnessLevel::Medium
    ).unwrap();

    let residues = input_pdb.residues().collect::<Vec<_>>();

    let mut output_pdb = PDB::new();

    let file = File::open("./data/features.json")?;
    let reader = BufReader::new(file);

    let entry: UniProtEntry = serde_json::from_reader(reader)?;

    let mut chains = Vec::new();
    let mut orphan_residues = Vec::new();
    let mut current_position = 0;

    for (index, feature) in entry.features.iter().enumerate() {
        if feature.feature_type != "Domain" {
            continue;
        }

        orphan_residues.extend(residues[current_position..(feature.location.start.value - 1)].iter().map(|residue| { (*residue).clone() }));

        let chain_residues = residues[(feature.location.start.value - 1)..(feature.location.end.value - 1)].iter().map(|residue| { (*residue).clone() });
        let chain = Chain::from_iter(format!("A{}", index), chain_residues).ok_or("Failed to create chain")?;

        chains.push(chain);
        current_position = feature.location.end.value - 1;
    }

    orphan_residues.extend(residues[current_position..].iter().map(|residue| { (*residue).clone() }));

    let orphan_chain = Chain::from_iter("B", orphan_residues.into_iter()).ok_or("Failed to create orphan chain")?;
    chains.push(orphan_chain);

    let model = Model::from_iter(0, chains.into_iter());
    output_pdb.add_model(model);

    pdbtbx::save(&output_pdb, "output/structure.pdb", pdbtbx::StrictnessLevel::Loose).unwrap();

    Ok(())
}




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
}

#[derive(Debug)]
struct Mutation {
    effects: MutationEffects,
    neomutation: bool,
    new_nucleotide: char,
    old_nucleotide: char,
    position: usize,
}

#[derive(Debug)]
struct MutationEffects {
    AAA: bool,
    Ectopia: bool,
    MFSClassic: bool,
    MFSFull: bool,
    MFSWithoutEye: bool,
    PVM: bool,
    SK: bool,
    TAA: bool,
}

#[derive(Debug, Deserialize)]
struct UniProtCoordinates {
    #[serde(rename = "gnCoordinate")]
    gn_coordinate: Vec<UniProtGnCoordinate>,
    sequence: String,
}

#[derive(Debug, Deserialize)]
struct UniProtGnCoordinate {
    #[serde(rename = "genomicLocation")]
    genomic_location: UniProtGenomicLocation,
}

#[derive(Debug, Deserialize)]
struct UniProtGenomicLocation {
    exon: Vec<UniProtExon>,
}

#[derive(Debug, Deserialize)]
struct UniProtExon {
    #[serde(rename = "genomeLocation")]
    genomic_location: UniProtGNLocation,

    #[serde(rename = "proteinLocation")]
    protein_location: UniProtGNLocation,
}

#[derive(Debug, Deserialize)]
struct UniProtGNLocation {
    begin: UniProtGNPosition,
    end: UniProtGNPosition,
}

#[derive(Debug, Deserialize)]
struct UniProtGNPosition {
    position: usize,
}

fn process_mutations() -> Result<(), Box<dyn Error>> {
    let file = File::open("./data/mutations.csv")?;
    let reader = BufReader::new(file);
    let mut csv_reader = csv::Reader::from_reader(reader);

    let mut index = 0;

    for result in csv_reader.deserialize() {
        let raw_mutation: RawMutation = result?;
        let mutation_chars = raw_mutation.mutation.chars().collect::<Vec<_>>();

        println!("{:?}", raw_mutation);

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

        let mutation = Mutation {
            effects,
            neomutation: raw_mutation.neomutation.is_some(),
            new_nucleotide: mutation_chars[mutation_chars.len() - 1],
            old_nucleotide: mutation_chars[mutation_chars.len() - 3],
            position: raw_mutation.position
        };

        eprintln!("{:#?}", mutation);

        index += 1;

        if index > 10 {
            break;
        }
    }

    Ok(())
}


#[derive(Debug)]
struct Exon {
    genomic_range: (usize, usize),
    protein_range: (usize, usize),
}

fn process_coordinates() -> Result<(), Box<dyn Error>> {
    let file = File::open("./data/coordinates.json")?;
    let reader = BufReader::new(file);

    let coords: UniProtCoordinates = serde_json::from_reader(reader)?;

    eprintln!("{:#?}", coords);

    let exons = coords.gn_coordinate.first().ok_or("No coordinates found")?.genomic_location.exon.iter().map(|raw_exon| {
        Exon {
            genomic_range: (
                raw_exon.genomic_location.begin.position - 1,
                raw_exon.genomic_location.end.position - 1
            ),
            protein_range: (
                raw_exon.protein_location.begin.position - 1,
                raw_exon.protein_location.end.position - 1
            )
        }
    }).collect::<Vec<_>>();

    eprintln!("{:#?}", exons);

    Ok(())
}


fn main() -> Result<(), Box<dyn Error>> {
    // process_mutations()?;
    process_coordinates()?;

    Ok(())
}
