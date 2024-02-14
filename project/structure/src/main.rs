use std::{error::Error, fs::File, io::BufReader};
use serde::Serialize;
use serde_pickle::SerOptions;
use pdbtbx::*;

mod features;
mod gn;
mod mutations;


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
    length: usize,
    mutations: Vec<self::mutations::Mutation>,
}


fn main() -> Result<(), Box<dyn Error>> {
    // process_mutations()?;
    // process_coordinates()?;
    // process_pdb()?;

    // eprintln!("{:#?}", prot);

    let prot = self::gn::process_coordinates("./data/coordinates.json")?;
    let mutations = self::mutations::process_mutations("./data/mutations.csv")?;
    let domains = self::features::process_domains("./data/features.json")?;

    let mut writer = File::create("./output/data.pkl")?;
    let output = Output {
        domains,
        length: prot.sequence.len(),
        mutations,
    };

    serde_pickle::to_writer(&mut writer, &output, SerOptions::new())?;

    Ok(())
}
