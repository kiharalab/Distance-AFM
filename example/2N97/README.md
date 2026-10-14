# 2N97 example

This example is a homodimer with two chains of 98 residues and 12 distance constraints. The AFM model and embedding are included in `input/`, so Step 1 can be skipped.

## Best result

The example's best result is provided separately in [`best_result/`](best_result/):

- [`2N97_best_pred.pdb`](best_result/2N97_best_pred.pdb): the model from refinement directly from AFM, without a generated start structure.
- [`best_results.txt`](best_result/best_results.txt): the corresponding training record.

## Files

```text
example/2N97/
├── input/
│   ├── 2N97.fasta
│   ├── 2N97
│   ├── dist_constraint.txt
│   ├── window.txt
│   ├── result_model_1_multimer_v2_pred_0.pkl
│   └── unrelaxed_model_1_multimer_v2_pred_0.pdb
├── best_result/
│   ├── 2N97_best_pred.pdb
│   └── best_results.txt
└── expected_output/
    ├── start_structures/
    │   ├── 2N97_start_0.pdb
    │   ├── 2N97_start_1.pdb
    │   └── 2N97_start_2.pdb
    └── start_0/2N97/
        ├── best_checkpoint/
        │   ├── 2N97_best_pred.pdb
        │   └── best_results.txt
        ├── loss_log.txt
        └── DONE.out
```

`expected_output/start_0/` contains a recorded refinement using `2N97_start_0.pdb`. Its `best_checkpoint` is the candidate from that individual run. The separate `best_result/` directory contains the example's AFM-direct result.

## Run the example

Run commands from the repository root after `conda activate dist-af`.

1. Generate start structures with the command and options in [Step 2](../../README.md#step-2-generate-start-structures). Use `example/2N97/input/` as the input and `example/2N97/output/start_structures/` as the output directory.
2. Follow [Step 3](../../README.md#step-3-refine-directly-from-afm-and-from-every-start-structure) to run one refinement directly from AFM and one with each generated start structure. Use the same input model, embedding, constraints, and domain windows, and a separate output directory for each run.
3. Collect every run's `best_checkpoint/2N97_best_pred.pdb` and rank the complete set with [FT-DMP](https://github.com/kliment-olechnovic/ftdmp) and [LZerD rank sum](https://lzerd.kiharalab.org/about/howtointerpret/), as described in [Step 4](../../README.md#step-4-rank-the-candidate-models).

For the three supplied start structures, the candidate set consists of the AFM-direct refinement and the three separate start-structure refinements. Include results from any additional starts or independent runs in the same target's candidate set.

To run only the AFM-direct route used for the supplied best result:

```bash
python main.py \
  --target_file=example/2N97/input/2N97 \
  --emd_file=example/2N97/input/result_model_1_multimer_v2_pred_0.pkl \
  --initial_pdb=example/2N97/input/unrelaxed_model_1_multimer_v2_pred_0.pdb \
  --dist_info=example/2N97/input/dist_constraint.txt \
  --window_info=example/2N97/input/window.txt \
  --output_dir=example/2N97/output/runs/afm_direct \
  --config_preset=model_1_multimer_v3 \
  --dist_weight=0.5 --loose_dist=1 \
  --epochs=20000 --val_epochs=1000 \
  --device_id=0
```
