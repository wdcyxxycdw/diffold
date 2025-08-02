"""
使用 Biotite 的 RNA 结构评估 + 自定义 TM-score + Clash score
-----------------------------------------------------------
- RMSD、lDDT : Biotite 官方实现
- TM-score   : 纯 NumPy，可自定义 d0
- Clash score: 纯 NumPy，自动补氢后按 MolProbity 定义
"""

import os, tempfile, logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import gemmi                          # 自动补氢

try:
    import biotite.structure as struc
    import biotite.structure.io as bio
    from diffold.diffold_to_pdb import diffold_coords_to_pdb
    BIOTITE_AVAILABLE = True
except ImportError:
    BIOTITE_AVAILABLE = False

logger = logging.getLogger(__name__)


class RNAEvaluationMetrics:
    # ---------- 初始化 ----------
    def __init__(
        self,
        *,
        tm_selection: str = "P",              # 'P'|'all_heavy'|'all'
        tm_d0_mode: str = "rna_casp",         # 'rna_casp'|'protein'|'custom'
        tm_d0_custom: Optional[float] = None,
    ):
        self.tm_selection = tm_selection
        self.tm_d0_mode  = tm_d0_mode
        self.tm_d0_custom = tm_d0_custom
        self.reset()
        self._check_biotite()

    def _check_biotite(self):
        if not BIOTITE_AVAILABLE:
            logger.warning("⚠️  Biotite 未安装，将无法计算结构指标；请 `pip install biotite`")
        else:
            logger.info("✅ Biotite 可用：RMSD / lDDT / 自定义 TM-score / Clash score 将启用")

    def reset(self):
        self.total_loss = 0.0
        self.total_samples = 0
        self.batch_count = 0
        self.loss_components = defaultdict(float)

        self.rmsd_values  = []
        self.lddt_scores  = []
        self.tm_scores    = []
        self.clash_scores = []

        self.confidence_scores = []

    # ---------- 外部调用 ----------
    def update(
        self,
        loss: float,
        batch_size: int,
        loss_breakdown: Dict[str, float] = None,
        predicted_coords: torch.Tensor = None,
        target_coords: torch.Tensor = None,
        confidence_scores: torch.Tensor = None,
        sequences: List[str] = None,
    ):
        self.total_loss += loss * batch_size
        self.total_samples += batch_size
        self.batch_count += 1

        if loss_breakdown:
            for k, v in loss_breakdown.items():
                self.loss_components[k] += v * batch_size

        if predicted_coords is not None and target_coords is not None:
            self._update_structure_metrics(predicted_coords, target_coords, sequences)

        if confidence_scores is not None:
            scores = (
                confidence_scores.plddt
                if hasattr(confidence_scores, "plddt")
                else confidence_scores
            )
            self.confidence_scores.extend(scores.flatten().tolist())

    # ---------- 结构指标 ----------
    def _update_structure_metrics(
        self,
        predicted_coords: torch.Tensor,
        target_coords: torch.Tensor,
        sequences: List[str] = None,
    ):
        if not BIOTITE_AVAILABLE:
            return

        pred_np   = predicted_coords.detach().cpu().numpy()
        target_np = target_coords.detach().cpu().numpy()
        for i in range(pred_np.shape[0]):
            seq = sequences[i] if sequences and i < len(sequences) else None
            met = self._compute_metrics_single(pred_np[i], target_np[i], seq)
            if not met:
                continue
            self.rmsd_values.append(met["rmsd"])
            self.lddt_scores.append(met["lddt"])
            self.tm_scores.append(met["tm_score"])
            self.clash_scores.append(met["clash"])

    def _compute_metrics_single(
        self,
        pred_coords: np.ndarray,
        target_coords: np.ndarray,
        sequence: str = None,
    ) -> Optional[Dict[str, float]]:
        # --- 写临时 PDB ---
        with tempfile.NamedTemporaryFile(suffix="_pred.pdb", delete=False) as fp:
            pred_pdb = fp.name
        with tempfile.NamedTemporaryFile(suffix="_ref.pdb", delete=False) as fr:
            ref_pdb = fr.name
        try:
            diffold_coords_to_pdb(pred_coords, sequence or "", pred_pdb)
            diffold_coords_to_pdb(target_coords, sequence or "", ref_pdb)

            # ---------- RMSD / lDDT 用原始重原子 ----------
            ref  = bio.load_structure(ref_pdb)
            pred = bio.load_structure(pred_pdb)

            rmsd_val  = struc.rmsd(pred, ref)
            lddt_val  = struc.lddt(ref, pred) * 100.0

            # ---------- TM-score ----------
            ref_xyz, pred_xyz = self._tm_select_xyz(ref, pred, self.tm_selection)
            P_aln, Qc = self._kabsch(pred_xyz, ref_xyz)
            L   = ref_xyz.shape[0]
            d0  = self._get_d0(L, self.tm_d0_mode, self.tm_d0_custom)
            tm  = self._tm_score(P_aln, Qc, d0)

            # ---------- Clash score：先给预测结构补氢 ----------
            pred_h_path = self._add_hydrogens_gemmi(pred_pdb)
            pred_h = bio.load_structure(pred_h_path)
            clash  = self._clash_score(pred_h)

            return {"rmsd": rmsd_val, "lddt": lddt_val, "tm_score": tm, "clash": clash}

        except Exception as e:
            logger.warning(f"单样本结构指标计算失败: {e}")
            return None
        finally:
            for f in (pred_pdb, ref_pdb):
                try:
                    os.unlink(f)
                except Exception:
                    pass

    # ----------  TM-score 工具函数 ----------
    def _tm_select_xyz(self, ref, pred, sel="P") -> Tuple[np.ndarray, np.ndarray]:
        if sel == "P":
            m_ref = ref.atom_name == "P"
            m_pred = pred.atom_name == "P"
        elif sel == "all_heavy":
            m_ref = ref.element != "H"
            m_pred = pred.element != "H"
        else:  # "all"
            m_ref = np.ones(ref.array_length(), bool)
            m_pred = np.ones(pred.array_length(), bool)

        keys_ref  = list(
            zip(ref.chain_id[m_ref], ref.res_id[m_ref], ref.atom_name[m_ref])
        )
        keys_pred = list(
            zip(pred.chain_id[m_pred], pred.res_id[m_pred], pred.atom_name[m_pred])
        )
        inter = sorted(set(keys_ref).intersection(keys_pred))
        if not inter:
            raise ValueError("TM 选择为空，无可对应原子")

        idx_ref = {k: i for i, k in enumerate(keys_ref)}
        idx_pred = {k: i for i, k in enumerate(keys_pred)}
        ref_xyz  = ref.coord[m_ref][[idx_ref[k] for k in inter]]
        pred_xyz = pred.coord[m_pred][[idx_pred[k] for k in inter]]
        return ref_xyz, pred_xyz

    def _kabsch(self, P, Q):
        Pc = P - P.mean(0, keepdims=True)
        Qc = Q - Q.mean(0, keepdims=True)
        V, _, Wt = np.linalg.svd(Pc.T @ Qc)
        R = V @ Wt
        if np.linalg.det(R) < 0:
            V[:, -1] *= -1
            R = V @ Wt
        return Pc @ R, Qc

    def _get_d0(self, L, mode, custom):
        if mode == "custom":
            if custom is None:
                raise ValueError("tm_d0_custom 未指定")
            d0 = float(custom)
        elif mode == "protein":
            d0 = 1.24 * (max(L - 15, 1)) ** (1 / 3) - 1.8
        else:  # 'rna_casp'
            d0 = 0.39 * (max(L - 5, 1)) ** (1 / 3) - 1.0
        return max(d0, 0.5)

    def _tm_score(self, P_aln, Qc, d0):
        dist2 = np.sum((P_aln - Qc) ** 2, axis=1)
        return float(np.mean(1.0 / (1.0 + dist2 / (d0 * d0))))

    # ----------  Clash score 工具函数 ----------
    def _add_hydrogens_gemmi(self, pdb_path: str) -> str:
        """Gemmi 加氢，返回新文件路径"""
        model = gemmi.read_structure(pdb_path)
        model.add_hydrogens()               # pH≈7 规则
        tmp = tempfile.NamedTemporaryFile(suffix="_H.pdb", delete=False)
        model.write_pdb(tmp.name)
        return tmp.name

    def _clash_score(self, structure) -> float:
        """
        MolProbity all-atom clashscore: 重叠 ≥0.4 Å 的原子对 ×1000 / 原子总数
        简化实现：不排除共价键、忽略 1-3/1-4 对，足够衡量几何质量趋势
        """
        coords = structure.coord
        elem   = structure.element

        cell = struc.CellList(coords, cell_size=5.0)
        pairs = cell.get_atom_pairs(5.0)
        if pairs.size == 0:
            return 0.0

        vdw = np.vectorize(struc.info.vdw_radius_single)(elem)
        sum_r = vdw[pairs[:, 0]] + vdw[pairs[:, 1]] - 0.4
        dists = struc.distance(coords[pairs[:, 0]], coords[pairs[:, 1]])
        n_clash = int(np.count_nonzero(dists < sum_r))
        return 1000.0 * n_clash / coords.shape[0]

    # ---------- 汇总 ----------
    def compute_metrics(self) -> Dict[str, float]:
        if self.total_samples == 0:
            return {}

        m = {
            "avg_loss": self.total_loss / self.total_samples,
            "batch_count": self.batch_count,
            "total_samples": self.total_samples,
            "structure_type": "RNA",
            "metrics_method": "Biotite+Custom(TM)+All-Atom Clash",
        }
        for k, v in self.loss_components.items():
            m[f"avg_{k}"] = v / self.total_samples

        if self.rmsd_values:
            m.update(
                avg_rmsd=float(np.mean(self.rmsd_values)),
                median_rmsd=float(np.median(self.rmsd_values)),
                std_rmsd=float(np.std(self.rmsd_values)),
                rmsd_method="Biotite",
            )
        if self.lddt_scores:
            arr = np.array(self.lddt_scores)
            m.update(
                avg_lddt=float(arr.mean()),
                median_lddt=float(np.median(arr)),
                std_lddt=float(arr.std()),
                lddt_high_quality_ratio=float((arr >= 70).mean()),
                lddt_good_quality_ratio=float((arr >= 50).mean()),
                lddt_method="Biotite",
            )
        if self.tm_scores:
            m.update(
                avg_tm=float(np.mean(self.tm_scores)),
                median_tm=float(np.median(self.tm_scores)),
                std_tm=float(np.std(self.tm_scores)),
                tm_selection=self.tm_selection,
                tm_d0_mode=self.tm_d0_mode,
                **({"tm_d0_custom": float(self.tm_d0_custom)}
                   if self.tm_d0_mode == "custom" else {}),
            )
        if self.clash_scores:
            m.update(
                avg_clash=float(np.mean(self.clash_scores)),
                median_clash=float(np.median(self.clash_scores)),
                std_clash=float(np.std(self.clash_scores)),
            )
        if self.confidence_scores:
            m.update(
                avg_confidence=float(np.mean(self.confidence_scores)),
                median_confidence=float(np.median(self.confidence_scores)),
            )
        return m
