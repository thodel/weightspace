from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
import numpy as np

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"


def top_deletion_count(confusions, top_k=10):
    total_del = 0
    for item in confusions[:top_k]:
        if isinstance(item, dict):
            ref = item.get("ref", item.get("reference", ""))
            hyp = item.get("hyp", item.get("hypothesis", ""))
            cnt = item.get("count", 0)
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            ref, hyp, cnt = item[0], item[1], item[2]
        else:
            continue
        if hyp == "" or hyp == "∅":
            total_del += int(cnt)
    return total_del


def try_blank_bias(s, target_val, label, hyp_text, hyp_pred, best_cer, best_collapsed):
    cur_val = float(s.subject.tensors()["O_18.lin.bias"][0])
    if abs(cur_val - target_val) < 1e-5 or s.budget_left <= 0:
        return False, best_cer, best_collapsed
    h = s.hypothesis(hyp_text, hyp_pred, "set_rows")
    s.edit(
        "set_rows",
        ["O_18.lin.bias"],
        hypothesis=h,
        index=[0],
        values=np.array([target_val], dtype=np.float32),
    )
    m = s.measure(label)
    if m.cer < best_cer and m.collapsed <= best_collapsed:
        kept = s.keep(
            f"CER improved {best_cer:.4f} -> {m.cer:.4f}, collapsed {best_collapsed} -> {m.collapsed}"
        )
        if kept:
            return True, m.cer, m.collapsed
        return False, best_cer, best_collapsed
    else:
        s.revert(
            f"No improvement over kept state (CER {m.cer:.4f} vs {best_cer:.4f}, collapsed {m.collapsed} vs {best_collapsed})"
        )
        return False, best_cer, best_collapsed


def main():
    s = open_session(".")
    parent = KrakenSubject(PARENT)
    pT = parent.tensors()

    # Pass 1: C0 baseline
    m0 = s.measure("C0")
    best_cer = m0.cer
    best_collapsed = m0.collapsed
    c0_del = top_deletion_count(m0.confusions)

    c0_blank_bias = float(s.subject.tensors()["O_18.lin.bias"][0])
    parent_blank_bias = float(pT["O_18.lin.bias"][0])
    s.note(
        f"C0 CER={m0.cer:.4f}, WER={m0.wer:.4f}, collapsed={m0.collapsed}, "
        f"top10_del={c0_del}, c0_blank_bias={c0_blank_bias:.4f}, parent_blank_bias={parent_blank_bias:.4f}"
    )

    # Pass 2 (H1): subtract 2.0 from O_18.lin.bias[0]
    kept_h1, best_cer, best_collapsed = try_blank_bias(
        s,
        c0_blank_bias - 2.0,
        "H1_blank_bias_minus_2.0",
        "H1: Subtract 2.0 from CTC blank bias O_18.lin.bias[0] (+3.232 -> +1.232) to reverse planted blank bias boost causing deletions and 14 collapsed lines",
        "CER decreases by 0.020 to 0.035 (from 0.0789 to ~0.045-0.059) and collapsed lines drop from 14 to 0-2",
        best_cer,
        best_collapsed,
    )

    # Pass 3 (H2): test exact parent blank bias (+1.119, i.e. delta -2.113 from C0)
    kept_h2, best_cer, best_collapsed = try_blank_bias(
        s,
        parent_blank_bias,
        "H2_blank_bias_parent",
        "H2: Set O_18.lin.bias[0] to exact parent blank bias (+1.119, -2.113 from C0)",
        "CER decreases by 0.0005 to 0.0030 if exact parent blank bias is superior to -2.0 shift",
        best_cer,
        best_collapsed,
    )

    if kept_h1 or kept_h2:
        # Pass 4 (H3): test -2.5 from C0
        kept_h3, best_cer, best_collapsed = try_blank_bias(
            s,
            c0_blank_bias - 2.5,
            "H3_blank_bias_minus_2.5",
            "H3: Lower O_18.lin.bias[0] to C0 - 2.5 (+0.732) to test if planted offset was +2.5",
            "CER decreases by 0.001 to 0.008 if planted offset was +2.5",
            best_cer,
            best_collapsed,
        )

        # Pass 5 (H4a): test -1.5 from C0
        kept_h4a, best_cer, best_collapsed = try_blank_bias(
            s,
            c0_blank_bias - 1.5,
            "H4a_blank_bias_minus_1.5",
            "H4a: Set O_18.lin.bias[0] to C0 - 1.5 (+1.732) to test if planted offset was +1.5",
            "CER decreases by 0.001 to 0.005 if planted offset was +1.5",
            best_cer,
            best_collapsed,
        )

        # Pass 6 (H4b): fine bracket around current best blank bias (-0.25 or +0.25)
        cur_best_bias = float(s.subject.tensors()["O_18.lin.bias"][0])
        if kept_h3:
            probe_val = c0_blank_bias - 3.0
            probe_label = "H4b_blank_bias_minus_3.0"
        elif kept_h4a:
            probe_val = c0_blank_bias - 1.25
            probe_label = "H4b_blank_bias_minus_1.25"
        else:
            probe_val = cur_best_bias - 0.25
            probe_label = "H4b_blank_bias_fine_minus_0.25"

        _, best_cer, best_collapsed = try_blank_bias(
            s,
            probe_val,
            probe_label,
            f"H4b: Fine-bracket O_18.lin.bias[0] at {probe_val:.4f}",
            "CER decreases by 0.0005 to 0.0020 if true optimum lies slightly lower",
            best_cer,
            best_collapsed,
        )
    else:
        # Branch if -2.0 and parent bias (-2.113) failed: try smaller (-1.0, -1.5) and larger (-3.0) shifts
        for delta, lbl in [(-1.0, "H3_alt_minus_1.0"), (-1.5, "H4_alt_minus_1.5"), (-3.0, "H4b_alt_minus_3.0")]:
            _, best_cer, best_collapsed = try_blank_bias(
                s,
                c0_blank_bias + delta,
                lbl,
                f"Alternative blank bias shift {delta:+.1f} on O_18.lin.bias[0]",
                "CER decreases by 0.010 to 0.030",
                best_cer,
                best_collapsed,
            )

    # Pass 7 (H5): test transplanting C_3.co.bias from parent (rel L2 1.0871, cos 0.6793)
    if s.budget_left > 0:
        h5 = s.hypothesis(
            "H5: Transplant C_3.co.bias from parent to test whether its 1.0871 rel L2 divergence is a secondary defect rather than benign small-norm drift",
            "CER decreases by 0.001 to 0.005 if C_3.co.bias was corrupted",
            "transplant",
        )
        s.edit(
            "transplant",
            ["C_3.co.bias"],
            hypothesis=h5,
            donor=pT["C_3.co.bias"],
        )
        m5 = s.measure("H5_transplant_C3_bias")
        if m5.cer < best_cer and m5.collapsed <= best_collapsed:
            kept = s.keep(f"CER improved {best_cer:.4f} -> {m5.cer:.4f}")
            if kept:
                best_cer = m5.cer
                best_collapsed = m5.collapsed
        else:
            s.revert(f"C_3.co.bias transplant did not improve CER ({m5.cer:.4f} vs {best_cer:.4f})")

    final_bias = float(s.subject.tensors()["O_18.lin.bias"][0])
    s.note(
        f"Finished session: best_cer={best_cer:.4f}, best_collapsed={best_collapsed}, "
        f"final O_18.lin.bias[0]={final_bias:.4f} (delta={final_bias - c0_blank_bias:+.4f})"
    )
    s.freeze()


if __name__ == "__main__":
    main()
