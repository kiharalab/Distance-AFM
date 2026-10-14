from model.openfold_multimer.structure_module import StructureModule
from model.openfold.loss import (
    sidechain_loss,
    compute_renamed_ground_truth,
    find_structural_violations,
    violation_loss,
    supervised_chi_loss,
    lddt_loss,
    backbone_loss,
    compute_plddt,
    compute_tm,
)
from model.openfold.feats import atom14_to_atom37
from model.aux_heads import (
    PerResidueLDDTCaPredictor,
    ExperimentallyResolvedHead,
    TMScoreHead,
    ContactHead,
    KDHead,
    ContactBlock2D,
    BasicBlock2D,
    multimer_backbone_loss,
)
import torch
import torch.nn as nn


class AlphaFold_Multimer(nn.Module):
    def __init__(self, args):
        super(AlphaFold_Multimer, self).__init__()
        self.structure_module = StructureModule(
            trans_scale_factor=args.point_scale,
            no_blocks=args.ipa_depth,
            no_heads_ipa=12,
            c_ipa=16,
            dropout_rate=0.1,
        )  # no_heads_ipa=24, c_ipa=64
        self.plddt = PerResidueLDDTCaPredictor()
        self.experimentally_resolved = ExperimentallyResolvedHead()
        # self.tm = TMScoreHead()

        self.args = args

        tmp_block = 5
        self.contact_res = ContactBlock2D()
        self.residual_blocks = nn.ModuleList([])
        for i in range(tmp_block):
            self.residual_blocks.append(BasicBlock2D())

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

        self.kd_blocks = KDHead(args, num_blocks=5)

    def forward_training(
        self,
        embedding,
        single_repr,
        aatype,
        batch_gt,
        batch_gt_frames,
        resolution,
        representation=None,
        emb2=None,
    ):

        output_bb, outputs = self.structure_module(
            single_repr, embedding, aatype=aatype, mask=batch_gt_frames["seq_mask"]
        )

        pred_frames = torch.stack(output_bb)
        return outputs, pred_frames

    def forward_testing(self, pair, single, aatype, protein_test):
        single_repr, embedding, contact_emb = self.kd_blocks(
            pair, single=single, batch_gt=protein_test
        )
        output_bb, outputs = self.structure_module(
            single_repr, embedding, aatype=aatype, mask=None
        )

        target = protein_test["target"]

        lddt = self.plddt(outputs["single"])
        plddt = compute_plddt(lddt)

        docking_score = 0  # TM/pTM/ipTM docking score not yet enabled
        return outputs["positions"], plddt, docking_score

    def forward(
        self,
        embedding,
        single_repr,
        aatype,
        batch_gt,
        batch_gt_frames,
        resolution,
        representation=None,
        training=True,
        protein_test=None,
    ):
        if training:
            return self.forward_training(
                embedding,
                single_repr,
                aatype,
                batch_gt,
                batch_gt_frames,
                resolution,
                representation=representation,
            )
        else:
            return self.forward_testing(
                embedding, single_repr, aatype, protein_test=protein_test
            )
