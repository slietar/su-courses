use serde::{Deserialize, Serialize};


#[derive(Debug, Deserialize, Serialize)]
pub struct Output {
    pub domain_kinds: Vec<String>,
    pub domains: Vec<super::features::Domain>,
    pub effect_labels: Vec<String>,
    pub exons: Vec<super::gn::Exon>,
    pub mutations: Vec<super::mutations::Mutation>,
    pub pathogenicity_labels: Vec<String>,
    pub plddt: Vec<f64>,
    pub sequence: Vec<char>,
    pub structures: Vec<super::structures::ExperimentalStructure>,
    pub variants: Vec<super::variants::Variant>,
}
