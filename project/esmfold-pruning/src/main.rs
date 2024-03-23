use std::{error::Error, fs::File, path::{Path, PathBuf}};

use pdbtbx::Chain;
use serde::Deserialize;
use serde_pickle::SerOptions;


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


#[derive(Debug)]
struct Spec {
    input_path: &'static str,
    kind: SpecKind,
    name: &'static str,
}

#[derive(Debug)]
enum SpecKind {
    Contextualized {
        pruned_name: &'static str,
    },
    Global,
    Isolated,
}


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data: Output = {
        let bytes = std::fs::read("../structure/output/data.pkl")?;
        serde_pickle::from_slice(&bytes, Default::default())?
    };

    let root_output_path = Path::new("../output/structures");

    for spec in [
        Spec {
            input_path: "../drive/FBN1_AlphaFold.pdb",
            kind: SpecKind::Global,
            name: "alphafold-global",
        },
        Spec {
            input_path: "../sources/esmfold-contextualized/domains/{}.pdb",
            kind: SpecKind::Contextualized {
                pruned_name: "esmfold-pruned",
            },
            name: "esmfold-contextualized",
        },
        Spec {
            input_path: "../sources/esmfold-isolated/domains/{}/structure.pdb",
            kind: SpecKind::Isolated,
            name: "esmfold-isolated",
        },
        Spec {
            input_path: "../sources/alphafold-contextualized/{}/main_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb",
            kind: SpecKind::Contextualized {
                pruned_name: "alphafold-pruned",
            },
            name: "alphafold-contextualized",
        },
    ] {
        let output_dir_path = root_output_path.join(spec.name);
        std::fs::create_dir_all(&output_dir_path)?;

        match spec.kind {
            SpecKind::Contextualized { .. } | SpecKind::Isolated => {
                let mut domains_plddt = Vec::with_capacity(data.domains.len());

                struct PrunedInfo {
                    output_dir_path: PathBuf,
                    plddt: Vec<Vec<f64>>,
                }

                let mut pruned_info = if let SpecKind::Contextualized { pruned_name } = spec.kind {
                    let path = root_output_path.join(pruned_name);
                    std::fs::create_dir_all(&path)?;

                    Some(PrunedInfo {
                        output_dir_path: path,
                        plddt: Vec::with_capacity(data.domains.len()),
                    })
                } else {
                    None
                };


                for (domain_index, domain) in data.domains.iter().enumerate() {
                    let domain_number = format!("{:04}", domain_index);
                    let input_path = spec.input_path.replace("{}", &domain_number);
                    let output_name = &format!("{}.pdb", domain_number);
                    let output_path = output_dir_path.join(output_name);

                    // eprintln!("{}", &input_path);
                    // eprintln!("{}", &output_path.to_string_lossy());

                    if !Path::new(&input_path).exists() {
                        eprintln!("{}: Skipping domain {:02} because input file does not exist", spec.name, domain_index);

                        domains_plddt.push(vec![]);

                        if let Some(pruned_info) = &mut pruned_info {
                            pruned_info.plddt.push(vec![]);
                        }

                        continue;
                    }

                    let (input_pdb, _) = pdbtbx::open(
                        &input_path,
                        pdbtbx::StrictnessLevel::Medium,
                    ).or_else(|_| Err("Failed to load PDB"))?;

                    if let Some(pruned_info) = &mut pruned_info {
                        // let pruned_output_path = output_dir_path.join(pruned_name).join(&format!("{}.pdb", domain_number));
                        // pdbtbx::save(&input_pdb, &pruned_output_path.to_string_lossy(), pdbtbx::StrictnessLevel::Loose).or_else(|_| Err("Failed to save PDB"))?;

                        let input_model = input_pdb.model(0).ok_or("Failed to get model")?;
                        let chain = input_model.chain(0).ok_or("Failed to get chain")?;

                        let offset = domain.start_position - (if domain_index > 0 {
                            data.domains[domain_index - 1].start_position
                        } else {
                            0
                        });

                        let residues = chain.residues().skip(offset).take(domain.end_position - domain.start_position + 1).cloned().collect::<Vec<_>>();

                        pruned_info.plddt.push(get_plddt(residues.iter())?);

                        let new_chain = Chain::from_iter("A", residues.into_iter()).ok_or("Failed to create chain")?;

                        let mut output_pdb = pdbtbx::PDB::new();
                        let output_model = pdbtbx::Model::from_iter(0, [new_chain].into_iter());
                        output_pdb.add_model(output_model);

                        pdbtbx::save(&output_pdb, &pruned_info.output_dir_path.join(output_name).to_string_lossy(), pdbtbx::StrictnessLevel::Loose).or_else(|_| Err("Failed to save PDB"))?;
                    }

                    // let residues = match spec.kind {
                    //     SpecKind::Contextualized => {
                    //         chain.residues().skip(offset).take(domain.end_position - domain.start_position + 1).cloned().collect::<Vec<_>>()
                    //     },
                    //     SpecKind::Isolated => {
                    //         chain.residues().cloned().collect::<Vec<_>>()
                    //     },
                    //     _ => unreachable!(),
                    // };

                    // let residues = chain.residues().skip(offset).take(domain.end_position - domain.start_position + 1).cloned().collect::<Vec<_>>();

                    domains_plddt.push(get_plddt(input_pdb.residues())?);

                    // let new_chain = Chain::from_iter("A", residues.into_iter()).ok_or("Failed to create chain")?;

                    // let mut output_pdb = pdbtbx::PDB::new();
                    // let output_model = pdbtbx::Model::from_iter(0, [new_chain].into_iter());
                    // output_pdb.add_model(output_model);

                    pdbtbx::save(&input_pdb, &output_path.to_string_lossy(), pdbtbx::StrictnessLevel::Loose).or_else(|_| Err("Failed to save PDB"))?;
                }

                let mut writer = File::create(output_dir_path.join("plddt.pkl"))?;
                serde_pickle::to_writer(&mut writer, &domains_plddt, SerOptions::new())?;

                if let Some(pruned_info) = pruned_info {
                    let mut writer = File::create(pruned_info.output_dir_path.join("plddt.pkl"))?;
                    serde_pickle::to_writer(&mut writer, &pruned_info.plddt, SerOptions::new())?;
                }
            },
            SpecKind::Global => {
                let (input_pdb, _) = pdbtbx::open(
                    spec.input_path,
                    pdbtbx::StrictnessLevel::Medium,
                ).or_else(|_| Err("Failed to load PDB"))?;

                let plddt = get_plddt(input_pdb.residues())?;

                pdbtbx::save(&input_pdb, &output_dir_path.join("structure.pdb").to_string_lossy(), pdbtbx::StrictnessLevel::Loose).or_else(|_| Err("Failed to save PDB"))?;

                let mut writer = File::create(output_dir_path.join("plddt.pkl"))?;
                serde_pickle::to_writer(&mut writer, &plddt, SerOptions::new())?;
            },
        }
    }

    Ok(())
}


fn get_plddt<'a, T: Iterator<Item = &'a pdbtbx::Residue>>(residues: T) -> Result<Vec<f64>, Box<dyn Error>> {
    Ok(residues.map(|residue| {
        residue.atom(0).and_then(|atom| {
            Some(atom.b_factor())
        }).ok_or("Failed to get atom")
    }).collect::<Result<Vec<_>, _>>()?)
}
