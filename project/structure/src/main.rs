use std::{error::Error, fs::File};
use serde::Serialize;
use serde_pickle::SerOptions;
// use pdbtbx::*;

mod features;
mod gn;
mod mutations;
mod variants;


/* fn process_pdb() -> Result<(), Box<dyn Error>> {
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
} */


#[derive(Debug, Serialize)]
struct Output {
    domains: Vec<self::features::Domain>,
    effect_labels: &'static [&'static str],
    exons: Vec<self::gn::Exon>,
    mutations: Vec<self::mutations::Mutation>,
    pathogenicity_labels: Vec<&'static str>,
    sequence: Vec<char>,
    variants: Vec<self::variants::Variant>,
}


fn main() -> Result<(), Box<dyn Error>> {
    let domains = self::features::process_domains("./data/features.json")?;
    let mutations = self::mutations::process_mutations("./data/mutations.csv")?;
    let protein_data = self::gn::process_coordinates("./data/coordinates.json")?;
    let variant_data = self::variants::process_variants("./data/variants.csv")?;

/*
    let mut output = String::new();
    output += "[";

    for domain in &domains {
        let (start, end) = &domain.range;
        output += "'";
        output += &String::from_iter(&protein.sequence[(start - 1)..(*end)]);
        output += "', ";
    }

    output += "]";
    println!("{}", output); */

    let mut writer = File::create("./output/data.pkl")?;
    let output = Output {
        domains,
        effect_labels: self::mutations::EFFECT_LABELS,
        exons: protein_data.exons,
        mutations,
        pathogenicity_labels: variant_data.pathogenicity_labels,
        sequence: protein_data.sequence,
        variants: variant_data.variants,
    };

    serde_pickle::to_writer(&mut writer, &output, SerOptions::new())?;

    Ok(())
}
