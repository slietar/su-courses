use std::{error::Error, fs::File, io::BufReader};

use serde::{Deserialize, Serialize};


#[derive(Debug, Deserialize)]
struct UniProtEntry {
    #[serde(rename = "uniProtKBCrossReferences")]
    references: Vec<Reference>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "database")]
enum Reference {
    PDB {
        id: String,
        properties: Vec<PDBProperty>,
    },

    AGR,
    Antibodypedia,
    Bgee,
    BioGRID,
    BioMuta,
    BMRB,
    CCDS,
    CDD,
    ChiTaRS,
    CORUM,
    CPTAC,
    CPTC,
    CTD,
    DIP,
    DisGeNET,
    DMDM,
    DNASU,
    eggNOG,
    ELM,
    EMBL,
    Ensembl,
    EPD,
    EvolutionaryTrace,
    ExpressionAtlas,
    Gene3D,
    GeneCards,
    GeneID,
    GeneReviews,
    GeneTree,
    Genevisible,
    GeneWiki,
    GenomeRNAi,
    GlyConnect,
    GlyCosmos,
    GlyGen,
    GO,
    HGNC,
    HOGENOM,
    HPA,
    InParanoid,
    IntAct,
    InterPro,
    iPTMnet,
    jPOST,
    KEGG,
    MalaCards,
    MassIVE,
    MaxQB,
    MIM,
    MINT,
    neXtProt,
    OMA,
    OpenTargets,
    Orphanet,
    OrthoDB,
    PANTHER,
    PathwayCommons,
    PaxDb,
    PDBsum,
    PeptideAtlas,
    Pfam,
    PharmGKB,
    Pharos,
    PhosphoSitePlus,
    PhylomeDB,
    PIR,
    PIRSF,
    PRO,
    PROSITE,
    Proteomes,
    ProteomicsDB,
    Pumba,
    Reactome,
    RefSeq,
    RNAct,
    SASBDB,
    SignaLink,
    SIGNOR,
    SMART,
    SMR,
    STRING,
    SUPFAM,
    SwissPalm,
    TreeFam,
    UCSC,
    VEuPathDB,

    #[serde(rename = "MANE-Select")]
    MANESelect,

    #[serde(rename = "BioGRID-ORCS")]
    BioGRIDORCS,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "key", content = "value")]
enum PDBProperty {
    Chains(String),
    Method(String),
    Resolution(String),
}


#[derive(Debug, Serialize)]
pub struct ExperimentalStructure {
    pub id: String,

    pub start_position: usize,
    pub end_position: usize,
}


pub fn process_structures() -> Result<Vec<ExperimentalStructure>, Box<dyn Error>> {
    let file = File::open("./data/entry.json")?;
    let reader = BufReader::new(file);
    let entry: UniProtEntry = serde_json::from_reader(reader)?;

    let mut structures = Vec::new();

    for reference in &entry.references {
        if let Reference::PDB { id, properties } = reference {
            let mut position_range = None;

            for property in properties {
                match property {
                    PDBProperty::Chains(chains) => {
                        let (_, raw_range) = chains.split_once('=').ok_or("Invalid chains property")?;
                        let (raw_start, raw_end) = raw_range.split_once('-').ok_or("Invalid range")?;

                        position_range = Some((
                            raw_start.parse::<usize>()?,
                            raw_end.parse::<usize>()?,
                        ));

                        break;
                    },
                    _ => (),
                }
            }

            if let Some((start, end)) = position_range {
                structures.push(ExperimentalStructure {
                    id: id.clone(),
                    start_position: start,
                    end_position: end,
                });
            }
        }
    }

    Ok(structures)
}
