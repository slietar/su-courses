use pdbtbx::*;


pub type Plddt = Vec<f64>;

pub fn process_plddt() -> Result<Plddt, Box<dyn std::error::Error>> {
    let (pdb, _) = pdbtbx::open(
        "../drive/FBN1_AlphaFold.pdb",
        StrictnessLevel::Medium
    ).or_else(|_| Err("Failed to load PDB"))?;

    let model = pdb.model(0).ok_or("Failed to get model")?;

    let plddt = model.residues().map(|residue| {
        residue.atom(0).and_then(|atom| {
            Some(atom.b_factor())
        }).ok_or("Failed to get atom")
    }).collect::<Result<Vec<_>, _>>()?;

    Ok(plddt)
}
