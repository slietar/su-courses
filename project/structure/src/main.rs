use std::{error::Error, fs::File};
use serde::Serialize;
use serde_pickle::SerOptions;

mod features;
mod gn;
mod mutations;
mod plddt;
mod structures;
mod variants;


#[derive(Debug, Serialize)]
struct Output {
    domain_kinds: &'static [&'static str],
    domains: Vec<self::features::Domain>,
    effect_labels: &'static [&'static str],
    exons: Vec<self::gn::Exon>,
    mutations: Vec<self::mutations::Mutation>,
    pathogenicity_labels: Vec<&'static str>,
    plddt: Vec<f64>,
    sequence: Vec<char>,
    structures: Vec<self::structures::ExperimentalStructure>,
    variants: Vec<self::variants::Variant>,
}


fn main() -> Result<(), Box<dyn Error>> {
    let domains = self::features::process_domains("./data/features.json")?;
    let mutations = self::mutations::process_mutations("./data/mutations.csv")?;
    let plddt = self::plddt::process_plddt()?;
    let protein_data = self::gn::process_coordinates("./data/coordinates.json")?;
    let structures = self::structures::process_structures()?;
    let variant_data = self::variants::process_variants("./data/variants.csv")?;

    let mut writer = File::create("./output/data.pkl")?;
    let output = Output {
        domain_kinds: self::features::DOMAIN_KINDS,
        domains,
        effect_labels: self::mutations::EFFECT_LABELS,
        exons: protein_data.exons,
        mutations,
        pathogenicity_labels: variant_data.pathogenicity_labels,
        plddt,
        sequence: protein_data.sequence,
        structures,
        variants: variant_data.variants,
    };

    serde_pickle::to_writer(&mut writer, &output, SerOptions::new())?;

    Ok(())
}
