use std::{error::Error, fs::File};
use serde_pickle::SerOptions;

mod features;
mod gn;
mod mutations;
mod output;
mod plddt;
mod structures;
mod variants;


fn main() -> Result<(), Box<dyn Error>> {
    let domains = self::features::process_domains("./data/features.json")?;
    let mutations = self::mutations::process_mutations()?;
    let plddt = self::plddt::process_plddt()?;
    let protein_data = self::gn::process_coordinates("./data/coordinates.json")?;
    let structures = self::structures::process_structures()?;
    let variant_data = self::variants::process_variants("./data/variants.csv")?;

    let mut writer = File::create("./output/data.pkl")?;
    let output = self::output::Output {
        domain_kinds: self::features::DOMAIN_KINDS.map(|kind| kind.to_string()).to_vec(),
        domains,
        effect_labels: self::mutations::EFFECT_LABELS.map(|label| label.to_string()).to_vec(),
        exons: protein_data.exons,
        mutations,
        pathogenicity_labels: variant_data.pathogenicity_labels.iter().map(|kind| kind.to_string()).collect::<Vec<_>>(),
        plddt,
        sequence: protein_data.sequence,
        structures,
        variants: variant_data.variants,
    };

    serde_pickle::to_writer(&mut writer, &output, SerOptions::new())?;

    Ok(())
}
