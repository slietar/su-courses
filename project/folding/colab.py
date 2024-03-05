#@title ##run **ESMFold**
%%time
from string import ascii_uppercase, ascii_lowercase
import hashlib, re, os
import numpy as np
import torch
from jax.tree_util import tree_map
import matplotlib.pyplot as plt
from scipy.special import softmax
import gc
import pickle
from tqdm import tqdm

sequences = ['IVPICRHSCGDGFCSRPNMCTCPSGQIAPSCG', 'SIQHCNIRCMNGGSCSDDHCLCQKGYIGTHCG', 'QPVCESGCLNGGRCVAPNRCACTYGFTGPQCE', 'GPCFTVISNQMCQGQLSGIVCTKTLCCATVGRAWGHPCEMCPAQPHPCRRGFI', 'DVDECQAIPGLCQGGNCINTVGSFECKCPAGHKLNEVSQKCE', 'DIDECSTIPGICEGGECTNTVSSYFCKCPPGFYTSPDGTRCI', 'GYCYTALTNGRCSNQLPQSITKMQCCCDAGRCWSPGVTVAPEMCPIRATEDFNKLC', 'VTDYCQLVRYLCQNGRCIPTPGSYRCECNKGFQLDLRGECI', 'DVDECEKNPCAGGECINNQGSYTCQCRAGYQSTLTRTECR', 'DIDECLQNGRICNNGRCINTDGSFHCVCNAGFHVTRDGKNCE', 'DMDECSIRNMCLNGMCINEDGSFKCICKPGFQLASDGRYCK', 'DINECETPGICMNGRCVNTDGSYRCECFPGLAVGLDGRVCV', 'STCYGGYKRGQCIKPLFGAVTKSECCCASTEYAFGEPCQPCPAQNSAEYQALC', 'DINECALDPDICPNGICENLRGTYKCICNSGYEVDSTGKNCV', 'DINECVLNSLLCDNGQCRNTPGSFVCTCPKGFIYKPDLKTCE', 'DIDECESSPCINGVCKNSPGSFICECSSESTLDPTKTICI', 'GTCWQTVIDGRCEININGATLKSQCCSSLGAAWGSPCTLCQVDPICGKGYSR', 'DIDECEVFPGVCKNGLCVNTRGSFKCQCPSGMTLDATGRICL', 'ETCFLRYEDEECTLPIAGRHRMDACCCSVGAAWGTEECEECPMRNTPEYEELC', 'DINECKMIPSLCTHGKCRNTIGSFKCRCDSGFALDSEERNCT', 'DIDECRISPDLCGRGQCVNTPGDFECKCDEGYESGFMMMKNCM', 'DIDECQRDPLLCRGGVCHNTEGSYRCECPPGHQLSPNISACI', 'DINECELSAHLCPNGRCVNLIGKYQCACNPGYHSTPDRLFCV', 'DIDECSIMNGGCETFCTNSEGSYECSCQPGFALMPDQRSCT', 'DIDECEDNPNICDGGQCTNIPGEYRCLCYDGFMASEDMKTCV', 'DVNECDLNPNICLSGTCENTKGSFICHCDMGYSGKKGKTGCT', 'DINECEIGAHNCGKHAVCTNTAGSFKCSCSPGWIGDGIKCT', 'DLDECSNGTHMCSQHADCKNTMGSYRCLCKEGYTGDGFTCT', 'DLDECSENLNLCGNGQCLNAPGGYRCECDMGFVPSADGKACE', 'DIDECSLPNICVFGTCHNLPGLFRCECEIGYELDRSGGNCT', 'DVNECLDPTTCISGNCVNTPGSYICDCPPDFELNPTRVGCV', 'GNCYLDIRPRGDNGDTACSNEIGVGVSKASCCCSLGKAWGTPCEMCPAVNTSEYKILC', 'DIDECQELPGLCQGGKCINTFGSFQCRCPTGYYLNEDTRVCD', 'DVNECETPGICGPGTCYNTVGNYTCICPPDYMQVNGGNNCM', 'SLCYRNYYADNQTCDGELLFNMTKKMCCCSYNIGRAWNKPCEQCPIPSTDEFATLC', 'DIDECREIPGVCENGVCINMVGSFRCECPVGFFYNDKLLVCE', 'DIDECQNGPVCQRNAECINTAGSYRCDCKPGYRFTSTGQCN', 'DRNECQEIPNICSHGQCIDTVGSFYCLCHTGFKTNDDQTMCL', 'DINECERDACGNGTCRNTIGSFNCRCNHGFILSHNNDCI', 'DVDECASGNGNLCRNGQCINTVGSFQCQCNEGYEVAPDGRTCV', 'DINECLLEPRKCAPGTCQNLDGSYRCICPPGYSLQNEKCE', 'DIDECVEEPEICALGTCSNTEGSFKCLCPEGFSLSSSGRRCQ', 'SYCYAKFEGGKCSSPKSRNHSKQECCCALKGEGWGDPCELCPTEPDEAFRQIC', 'DMDECKEPDVCKHGQCINTDGSYRCECPFGYILAGNECV', 'DTDECSVGNPCGNGTCKNVIGGFECTCEEGFEPGPMMTCE', 'DINECAQNPLLCAFRCVNTYGSYECKCPVGYVLREDRRMCK', 'DEDECEEGKHDCTEKQMECKNLIGTYMCICGPGYQRRPDGEGCV', 'DENECQTKPGICENGRCLNTRGSYTCECNDGFTASPNQDECL', 'GYCFTEVLQNMCQIGSSNRNPVTKSECCCDGGRGWGPHCEICPFQGTVAFKKLC', 'DIDECKVIHDVCRNGECVNDRGSYHCICKTGYTPDITGTSCV', 'DLNECNQAPKPCNFICKNTEGSYQCSCPKGYILQEDGRSCK', 'DLDECATKQHNCQFLCVNTIGGFTCKCPPGFTQHHTSCI', 'DNNECTSDINLCGSKGICQNTPGSFTCECQRGFSLDQTGSSCE', 'DVDECEGNHRCQHGCQNIIGGYRCSCPQGYLQHYQWNQCV', 'DENECLSAHICGGASCHNTLGSYKCMCPAGFQYEQFSGGCQ', 'DINECGSAQAPCSYGCSNTEGGYLCGCPPGYFRIGQGHCV',]
metadata = []
metadata_keys = {'max_predicted_aligned_error', 'predicted_aligned_error', 'aligned_confidence_probs', 'ptm', 'plddt'}

def parse_output(output):
  pae = (output["aligned_confidence_probs"][0] * np.arange(64)).mean(-1) * 31
  plddt = output["plddt"][0,:,1]

  bins = np.append(0,np.linspace(2.3125,21.6875,63))
  sm_contacts = softmax(output["distogram_logits"],-1)[0]
  sm_contacts = sm_contacts[...,bins<8].sum(-1)
  xyz = output["positions"][-1,0,:,1]
  mask = output["atom37_atom_exists"][0,:,1] == 1
  o = {"pae":pae[mask,:][:,mask],
       "plddt":plddt[mask],
       "sm_contacts":sm_contacts[mask,:][:,mask],
       "xyz":xyz[mask]}
  return o

def get_hash(x): return hashlib.sha1(x.encode()).hexdigest()
alphabet_list = list(ascii_uppercase+ascii_lowercase)

for sequence_index, sequence in tqdm(list(enumerate(sequences))):
  jobname = "test" #@param {type:"string"}
  jobname = re.sub(r'\W+', '', jobname)[:50]

  #sequence = "IVPICRHSCGDGFCSRPNMCTCPSGQIAPSCG" #@param {type:"string"}
  sequence = re.sub("[^A-Z:]", "", sequence.replace("/",":").upper())
  sequence = re.sub(":+",":",sequence)
  sequence = re.sub("^[:]+","",sequence)
  sequence = re.sub("[:]+$","",sequence)
  copies = 1 #@param {type:"integer"}
  if copies == "" or copies <= 0: copies = 1
  sequence = ":".join([sequence] * copies)
  num_recycles = 3 #@param ["0", "1", "2", "3", "6", "12", "24"] {type:"raw"}
  chain_linker = 25

  #ID = jobname+"_"+get_hash(sequence)[:5]
  ID = f"{sequence_index:04}"
  seqs = sequence.split(":")
  lengths = [len(s) for s in seqs]
  length = sum(lengths)
  # print("length",length)

  u_seqs = list(set(seqs))
  if len(seqs) == 1: mode = "mono"
  elif len(u_seqs) == 1: mode = "homo"
  else: mode = "hetero"

  if "model" not in dir() or model_name != model_name_:
    if "model" in dir():
      # delete old model from memory
      del model
      gc.collect()
      if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = torch.load(model_name)
    model.eval().cuda().requires_grad_(False)
    model_name_ = model_name

  # optimized for Tesla T4
  if length > 700:
    model.set_chunk_size(64)
  else:
    model.set_chunk_size(128)

  torch.cuda.empty_cache()
  output = model.infer(sequence,
                      num_recycles=num_recycles,
                      chain_linker="X"*chain_linker,
                      residue_index_offset=512)

  pdb_str = model.output_to_pdb(output)[0]
  output = tree_map(lambda x: x.cpu().numpy(), output)
  output_filtered = { k: v for k, v in output.items() if k in metadata_keys }
  metadata.append(output_filtered)

  #ptm = output["ptm"][0]
  #plddt = output["plddt"][0,...,1].mean()
  O = parse_output(output)
  #print(f'ptm: {ptm:.3f} plddt: {plddt:.3f}')
  os.system(f"mkdir -p output/{ID}")
  #prefix = f"output/{ID}/ptm{ptm:.3f}_r{num_recycles}_default"
  np.savetxt(f"output/{ID}/pae.txt",O["pae"],"%.3f")
  with open(f"output/{ID}/structure.pdb","w") as out:
    out.write(pdb_str)

with open('metadata.pkl', 'wb') as file:
  pickle.dump(metadata, file)
