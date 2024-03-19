use pdbtbx::Chain;
use serde::Deserialize;


#[derive(Debug, Deserialize)]
pub struct Domain {
    #[serde(flatten)]
    pub kind: DomainKind,

    pub name: String,
    pub number: usize,

    pub start_position: usize,
    pub end_position: usize,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind")]
pub enum DomainKind {
    #[serde(rename = "EGF")]
    EGFLike,

    #[serde(rename = "EGFCB")]
    EGFLikeCalciumBinding,

    TB,
}

#[derive(Debug, Deserialize)]
struct Output {
    domains: Vec<Domain>,
}


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data: Output = {
        let bytes = std::fs::read("../structure/output/data.pkl")?;
        serde_pickle::from_slice(&bytes, Default::default())?
    };

    for (domain_index, domain) in data.domains.iter().enumerate() {
        let domain_number = format!("{:04}", domain_index);

        let offset = domain.start_position - (if domain_index > 0 {
            data.domains[domain_index - 1].start_position
        } else {
            0
        });

        let (input_pdb, _) = pdbtbx::open(
            &format!("../esmfold-output/contextualized/domains/{}.pdb", domain_number),
            pdbtbx::StrictnessLevel::Medium
        ).or_else(|_| Err("Failed to load PDB"))?;

        let input_model = input_pdb.model(0).ok_or("Failed to get model")?;
        let chain = input_model.chain(0).ok_or("Failed to get chain")?;
        let residues = chain.residues().skip(offset).take(domain.end_position - domain.start_position + 1).cloned().collect::<Vec<_>>();
        let new_chain = Chain::from_iter("A", residues.into_iter()).ok_or("Failed to create chain")?;

        let mut output_pdb = pdbtbx::PDB::new();

        let output_model = pdbtbx::Model::from_iter(0, [new_chain].into_iter());
        output_pdb.add_model(output_model);

        std::fs::create_dir_all("output")?;
        pdbtbx::save(&output_pdb, format!("output/{}.pdb", domain_number), pdbtbx::StrictnessLevel::Loose).or_else(|_| Err("Failed to save PDB"))?;
    }

    Ok(())
}
