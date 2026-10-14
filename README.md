# Distance-AF Multimer

Distance-AF Multimer refines AlphaFold-Multimer models of protein complexes with distance constraints between residues. Given an AlphaFold-Multimer model, its embedding, and a set of intra- or inter-chain CA–CA distances, it fine-tunes the structure module on the target. For each target, we run refinement directly from the AlphaFold-Multimer model and separately with each generated start structure. The resulting models form a candidate set for ranking with FTDMP and LZerD rank sum.

Copyright (C) 2026 Kai Ling, Yuanyuan Zhang, Zicong Zhang, Tsukasa Nakamura, Daisuke Kihara, and Purdue University.

License: GPL v3 (see [LICENSE](LICENSE)). Please contact us if you are interested in a different license, for example for commercial use.

Contact: Daisuke Kihara (dkihara@purdue.edu)

For technical problems or questions, please contact Kai Ling (ling59@purdue.edu).

## Installation

### Requirements
 + Linux with an NVIDIA GPU. Step 1 (AlphaFold-Multimer) and Step 3 (refinement) run on the GPU; Step 2 (start structure generation) runs on the CPU.
 + [conda](https://docs.conda.io/) (Anaconda or Miniconda).
 + For Step 1: the AlphaFold v2.2 model parameters and genetic databases, which need several terabytes of disk space.

### Setup
1. Clone this repository and enter it:
   ```
   git clone https://github.com/kiharalab/Distance-AFM.git
   cd Distance-AFM
   ```
2. Create the two conda environments:
   ```
   conda env create -f environment.yml             # dist-af: Step 2 and Step 3
   conda env create -f environment_embedding.yml   # dist-af-embedding: Step 1
   ```
   `dist-af-embedding` also provides the MSA tools and `ptxas` (package `cuda-nvcc`), which JAX needs to run AlphaFold on the GPU.
3. For Step 1, download the AlphaFold v2.2 model parameters (`alphafold_params_2022-03-02`, which contain the `multimer_v2` models) and the genetic databases with `scripts/download_all_data.sh <DATA_DIR>` of [AlphaFold v2.2.4](https://github.com/google-deepmind/alphafold/tree/v2.2.4).

Activate the environment of a step before running it: `conda activate dist-af-embedding` for Step 1 and `conda activate dist-af` for Step 2 and Step 3. The pretrained Distance-AF Multimer weights are part of the repository (`model_dir/multimer/model_state_dict.pt`) and are loaded automatically.

## Usage

Distance-AF Multimer runs in **4 steps**. Run the commands below in Bash from the repository root.

| Step | What it does | Script | Environment | Output used later |
|---|---|---|---|---|
| 1 | Runs AlphaFold-Multimer to get the embedding and the initial model | `run_alphafold_v3_msa.py`, `run_alphafold_v3.py` | `dist-af-embedding` | `result_model_1_multimer_v2_pred_0.pkl`, `unrelaxed_model_1_multimer_v2_pred_0.pdb` |
| 2 | Generates start structures by re-docking the chains using the distance constraints | `train_utils/generate_start_structures.py` | `dist-af` | `<TARGET>_start_0.pdb`, `<TARGET>_start_1.pdb`, ... |
| 3 | Runs independent refinements directly from AFM and with every start structure | `main.py` | `dist-af` | One `best_checkpoint/<TARGET>_best_pred.pdb` per run |
| 4 | Ranks all refined candidates for the target | FTDMP + LZerD rank sum | See the linked tools in Step 4 | Ranked candidate models |

`example/2N97` contains example inputs, a recorded start-structure run, and a separate [best result](example/2N97/best_result/2N97_best_pred.pdb); see the [example guide](example/2N97/README.md).

### Input files
 + **FASTA file** (Step 1): one entry per chain, for example `example/2N97/input/2N97.fasta`.
 + **Target file** (Step 3): a text file whose first line is the target name; the results are written to `<output_dir>/<target name>/`. For example, `example/2N97/input/2N97` contains `2N97`.
 + **Distance constraint file** (Step 2 and Step 3): one constraint per line, `i,j,d`, meaning that the CA–CA distance between residue `i` and residue `j` is `d` Å. Residues are numbered from 1 consecutively over all chains, in the order of the chains in the initial model; in 2N97, each chain has 98 residues, so residue 99 is the first residue of the second chain. The more constraints you provide, the better. From `example/2N97/input/dist_constraint.txt`:
   ```
   21,148,10.04
   89,187,47.79
   ```
   With `--dist_range=1` in Step 3, each line gives a lower and an upper bound instead: `i,j,lower,upper` (use `-inf` or `inf` for an open bound).
 + **Domain window file** (Step 3): one domain per line, `start,end`, in the same residue numbering; typically one line per chain. `example/2N97/input/window.txt`:
   ```
   1,98
   99,196
   ```

### Step 1: Generate the AlphaFold-Multimer embedding and the initial model
This step runs AlphaFold-Multimer v2.2 in the `dist-af-embedding` environment. `run_alphafold_v3_msa.py` runs the MSA and template search on the CPU and writes `features.pkl`; `run_alphafold_v3.py` reads `features.pkl`, runs AlphaFold-Multimer on the GPU and saves the embedding.
```
conda activate dist-af-embedding
DATA_DIR=/path/to/alphafold_data     # directory given to download_all_data.sh
AF_FLAGS="--fasta_paths=example/2N97/input/2N97.fasta \
  --output_dir=example/2N97/output/embedding \
  --data_dir=$DATA_DIR \
  --model_preset=multimer \
  --db_preset=full_dbs \
  --uniref90_database_path=$DATA_DIR/uniref90/uniref90.fasta \
  --mgnify_database_path=$DATA_DIR/mgnify/mgy_clusters_2018_12.fa \
  --bfd_database_path=$DATA_DIR/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
  --uniclust30_database_path=$DATA_DIR/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
  --uniprot_database_path=$DATA_DIR/uniprot/uniprot.fasta \
  --pdb_seqres_database_path=$DATA_DIR/pdb_seqres/pdb_seqres.txt \
  --template_mmcif_dir=$DATA_DIR/pdb_mmcif/mmcif_files \
  --obsolete_pdbs_path=$DATA_DIR/pdb_mmcif/obsolete.dat \
  --max_template_date=2024-10-01 \
  --use_precomputed_msas=True \
  --use_gpu_relax=False \
  --num_multimer_predictions_per_model=1"

# 1.1 MSA and template search (CPU)
CUDA_VISIBLE_DEVICES=-1 python run_alphafold_v3_msa.py $AF_FLAGS
# 1.2 AlphaFold-Multimer inference (GPU)
CUDA_VISIBLE_DEVICES=0 python run_alphafold_v3.py $AF_FLAGS
```
 + The MSA tools (`jackhmmer`, `hhblits`, `hhsearch`, `hmmsearch`, `hmmbuild`, `kalign`) come with the `dist-af-embedding` environment; use `--<tool>_binary_path` to use other binaries.
 + The output is written to `<output_dir>/<FASTA file name without extension>/`, for example `example/2N97/output/embedding/2N97/`:
   + `result_model_1_multimer_v2_pred_0.pkl`: the embedding, used as `--emd_file` in Step 3.
   + `unrelaxed_model_1_multimer_v2_pred_0.pdb`: the initial model, used as `--pdb_file` in Step 2 and as `--initial_pdb` in Step 3.
 + AlphaFold also writes the results of `model_2` to `model_5`; Distance-AF Multimer uses `model_1`, prediction 0.

### Step 2: Generate start structures
The commands below use the Step 1 output included in `example/2N97/input/`. The generator repositions and rotates chains using the distance constraints and writes candidate start structures. A native or experimental structure is not required.
```bash
conda activate dist-af
python train_utils/generate_start_structures.py \
  --target=2N97 \
  --pdb_file=example/2N97/input/unrelaxed_model_1_multimer_v2_pred_0.pdb \
  --dist_constraints_file=example/2N97/input/dist_constraint.txt \
  --output_dir=example/2N97/output/start_structures
```

| Option | Description |
|---|---|
| `--target` | Target name, used in the output file names |
| `--pdb_file` | Initial model from Step 1 |
| `--dist_constraints_file` | Distance constraint file |
| `--output_dir` | Directory for the start structures |
| `--search_step_degree` | Rotation search step in degrees (default: 30) |
| `--candidates_per_iteration` | Number of candidates kept at each chain-assembly iteration (default: 3) |
| `--use_multiple_constraints` | Use the 3 shortest constraints instead of only the shortest one |
| `--no_translation` | Only rotate the chains |
| `--output_rotation_pdbs` | Also write additional rotation variants |

The generated structures are written as `<TARGET>_start_0.pdb`, `<TARGET>_start_1.pdb`, ... in `--output_dir`. The number of structures depends on the target and the available orientations. Use every generated start structure in a separate Step 3 run. Keep the original AFM model as `--initial_pdb`; pass the generated structure as `--start_pdb`.

For a finer rotation search or more candidates, set `--search_step_degree` and `--candidates_per_iteration`, for example:
```bash
python train_utils/generate_start_structures.py \
  --target=2N97 \
  --pdb_file=example/2N97/input/unrelaxed_model_1_multimer_v2_pred_0.pdb \
  --dist_constraints_file=example/2N97/input/dist_constraint.txt \
  --search_step_degree=15 \
  --candidates_per_iteration=5 \
  --output_dir=example/2N97/output/start_structures_fine
```
Use a separate output directory for each generation setting. If `--output_rotation_pdbs` is enabled, the additional rotation PDBs can also be used as `--start_pdb`, with a distinct refinement output directory for each one.

### Step 3: Refine directly from AFM and from every start structure
Each target has multiple independent refinement runs: one directly from the AFM model, plus one for each generated start structure. All runs use the same AFM reference model, embedding, distance constraints, and domain windows. Give every run its own output directory.

Define the shared arguments:
```bash
conda activate dist-af
TARGET=2N97
INPUT_DIR=example/2N97/input
OUTPUT_DIR=example/2N97/output
COMMON_ARGS=(
  --target_file="$INPUT_DIR/$TARGET"
  --emd_file="$INPUT_DIR/result_model_1_multimer_v2_pred_0.pkl"
  --initial_pdb="$INPUT_DIR/unrelaxed_model_1_multimer_v2_pred_0.pdb"
  --dist_info="$INPUT_DIR/dist_constraint.txt"
  --window_info="$INPUT_DIR/window.txt"
  --config_preset=model_1_multimer_v3
  --dist_weight=0.5 --loose_dist=1
  --epochs=20000 --val_epochs=1000
  --device_id=0
)
```

**3a. Refine directly from AFM**, without `--start_pdb`:
```bash
python main.py "${COMMON_ARGS[@]}" \
  --output_dir="$OUTPUT_DIR/runs/afm_direct"
```

**3b. Refine with each generated start structure**:
```bash
for start_pdb in "$OUTPUT_DIR"/start_structures/"${TARGET}"_start_*.pdb; do
  [ -f "$start_pdb" ] || continue
  run_name=$(basename "$start_pdb" .pdb)
  python main.py "${COMMON_ARGS[@]}" \
    --start_pdb="$start_pdb" \
    --coarse_tuning_epochs=3000 \
    --output_dir="$OUTPUT_DIR/runs/$run_name"
done
```
These commands run sequentially on the selected GPU. If Step 2 used a different output directory, change the loop's start-structure directory accordingly. Repeat Step 3b for any additional rotation variants you choose to refine.

| Option | Default | Description |
|---|---|---|
| `--target_file` | required | Target file |
| `--emd_file` | required | Embedding from Step 1 (`result_model_1_multimer_v2_pred_0.pkl`) |
| `--initial_pdb` | required | Initial model from Step 1 (`unrelaxed_model_1_multimer_v2_pred_0.pdb`) |
| `--dist_info` | required | Distance constraint file |
| `--window_info` | required | Domain window file |
| `--output_dir` | `./example_output` | Results are written to `<output_dir>/<target name>/` |
| `--config_preset` | `model_1_multimer_v3` | Model configuration (multimer presets only) |
| `--epochs` | 20000 | Number of training epochs |
| `--val_epochs` | 100 | Save a checkpoint and a model every N epochs |
| `--dist_weight` | 0.5 | Weight of the distance loss; larger values penalize violations more |
| `--loose_dist` | 1 | Lower the weight of the distance loss once the constraints are roughly satisfied |
| `--dist_range` | 0 | 1: the constraint file gives lower and upper bounds instead of distances |
| `--start_pdb` | None | Start structure from Step 2 |
| `--coarse_tuning_epochs` | 3000 | Number of coarse tuning epochs, trained towards the start structure |
| `--weight_file` | None | Weight file to load instead of `model_dir/multimer/model_state_dict.pt` |
| `--model_dir` | None | Resume from a checkpoint directory |
| `--device_id` | 0 | GPU (CUDA device index) to use |

 + With a start structure, the first `--coarse_tuning_epochs` epochs pull the model towards the start structure (coarse tuning) and the remaining epochs fit the distance constraints (fine tuning). Without a start structure, all epochs fit the distance constraints.
 + Training stops early once the 10-epoch moving-average loss has not improved by more than 0.1% for 1000 consecutive epochs (checked only after epoch `--coarse_tuning_epochs`). With a start structure, the run stops if the loss is still 100 or more at the end of coarse tuning.

### Output
```
<output_dir>/<target name>/
├── best_checkpoint/
│   ├── <target name>_best_pred.pdb    # candidate from this run: the lowest moving-average loss
│   ├── best_results.txt               # best epoch, loss, and RMSD to the initial model
│   └── best_checkpoint.pth, model_state_dict.pt, optimizer.pt
├── checkpoint-<E>-<E>/                # every --val_epochs epochs
│   ├── <target name>_pred_epoch<E>.pdb
│   ├── rmsd.txt                       # RMSD to the initial model
│   └── checkpoint.pth, model_state_dict.pt, optimizer.pt
├── loss_log.txt                       # loss terms of every epoch
└── DONE.out                           # written when training finishes
```
 + Each checkpoint directory stores the full model and optimizer state (about 160 MB), so choose `--val_epochs` with disk space in mind.
 + The constraint and window files are cached as `<target name>_constraint*.pt` and `<target name>_domain_window.pt` in the output directory and reused when present, so use a new `--output_dir` when you change the constraint or window file.

### Step 4: Rank the candidate models
Collect `best_checkpoint/<TARGET>_best_pred.pdb` from **every refinement run for the same target**, including the AFM-direct run and all runs with different start structures. Each run contributes one candidate; keep distinct file names when collecting them.

Rank this complete candidate set using [FTDMP](https://github.com/kliment-olechnovic/ftdmp) together with [LZerD rank sum](https://lzerd.kiharalab.org/about/howtointerpret/). See their official pages for usage.

## Example
The [2N97 example guide](example/2N97/README.md) describes the supplied inputs, recorded run, and separate [best result](example/2N97/best_result/2N97_best_pred.pdb). The best result comes from refinement directly from AFM, without a generated start structure. Step 1 outputs are included, so you can begin with Step 2 and run both refinement routes in Step 3 before ranking all candidates in Step 4.
