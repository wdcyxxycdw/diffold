from pymol import cmd
import re
from collections import defaultdict

def _resi_key(resi):
    # sort like 12 < 12A < 13
    m = re.match(r"^\s*(\d+)\s*([A-Za-z]?)\s*$", str(resi))
    if m:
        return (int(m.group(1)), m.group(2))
    return (10**9, str(resi))

def _find_atom(atoms_dict, chain, resi, candidates):
    """Return (name, (x,y,z)) for the first existing atom in candidates list."""
    a = atoms_dict.get((chain, resi), {})
    for nm in candidates:
        if nm in a:
            return nm, a[nm]
    return None, None

def _collect_atoms(obj):
    """Build {(chain,resi): {atom_name: (x,y,z), ...}} for nucleic polymer."""
    md = cmd.get_model(f"{obj} and polymer.nucleic and not elem H")
    byres = defaultdict(dict)
    for at in md.atom:
        key = (at.chain, at.resi)
        byres[key][at.name] = at.coord
    return byres

def _chains_and_resis(atoms_dict, restrict_chain=None):
    chains = defaultdict(set)
    for (ch, resi) in atoms_dict.keys():
        if restrict_chain and ch != restrict_chain:
            continue
        chains[ch].add(resi)
    # order residues
    ordered = {ch: sorted(list(resis), key=_resi_key) for ch, resis in chains.items()}
    return ordered

def check_rna_backbone(obj, chain=None, cutoff=2.2, draw=1):
    """
    Check RNA backbone continuity via O3'(i)–P(i+1) and report missing key atoms.
    Usage:
      check_rna_backbone("8uyg_A_best", cutoff=2.2, draw=1)
    """
    # Accept common naming variants (prime vs asterisk; OP1 vs O1P)
    O3_candidates = ["O3'", "O3*"]
    O5_candidates = ["O5'", "O5*"]
    P_candidates  = ["P"]
    OP1_candidates= ["OP1", "O1P"]
    OP2_candidates= ["OP2", "O2P"]

    required_per_residue = [
        ("P", P_candidates),
        ("O5'", O5_candidates),
        ("C5'", ["C5'","C5*"]),
        ("C4'", ["C4'","C4*"]),
        ("C3'", ["C3'","C3*"]),
        ("O3'", O3_candidates),
    ]

    atoms = _collect_atoms(obj)
    order = _chains_and_resis(atoms, restrict_chain=chain)

    total_breaks = 0
    total_missing = 0
    report_lines = []
    miss_lines = []

    # clear previous distance objects if any
    try:
        cmd.delete(f"{obj}_backbone_break*")
    except:
        pass

    for ch, reslist in order.items():
        # Missing-atom audit
        for resi in reslist:
            missing_here = []
            for tag, cand in required_per_residue:
                nm, pos = _find_atom(atoms, ch, resi, cand)
                if pos is None:
                    missing_here.append(tag)
            if missing_here:
                total_missing += len(missing_here)
                miss_lines.append(f"[Missing] chain {ch} resi {resi}: " +
                                  ", ".join(missing_here))

        # Continuity via O3'(i)–P(i+1)
        for i in range(len(reslist)-1):
            r_i   = reslist[i]
            r_ip1 = reslist[i+1]

            nmO3, xyzO3 = _find_atom(atoms, ch, r_i, O3_candidates)
            nmP , xyzP  = _find_atom(atoms, ch, r_ip1, P_candidates)

            flag = None
            if xyzO3 is None or xyzP is None:
                flag = "MISSING_ENDPOINT"
                total_breaks += 1
                report_lines.append(f"[Break?] chain {ch} {r_i}({nmO3 or 'no O3'}) "
                                    f"→ {r_ip1}({nmP or 'no P'}) : missing atom(s)")
            else:
                # compute distance
                dx = xyzO3[0]-xyzP[0]; dy = xyzO3[1]-xyzP[1]; dz = xyzO3[2]-xyzP[2]
                dist = (dx*dx+dy*dy+dz*dz)**0.5
                if dist > float(cutoff):
                    flag = f"DIST={dist:.2f}Å"
                    total_breaks += 1
                    report_lines.append(f"[Break]  chain {ch} {r_i}(O3') → {r_ip1}(P)  "
                                        f"dist={dist:.2f} Å  (> {cutoff} Å)")
                # draw distance if requested and either flagged or user wants all draws
                if draw and flag:
                    # build selections to draw distance using the exact names found
                    sel1 = f"({obj} and chain {ch} and resi {r_i} and name {nmO3})"
                    sel2 = f"({obj} and chain {ch} and resi {r_ip1} and name {nmP})"
                    dname = f"{obj}_backbone_break_{ch}_{r_i}_{r_ip1}"
                    try:
                        cmd.distance(dname, sel1, sel2)
                        cmd.hide("labels", dname)  # keep view clean
                    except:
                        pass

    # Console report
    print("="*60)
    print(f"[{obj}] RNA backbone check (chain={chain or 'ALL'}, cutoff={cutoff} Å)")
    print(f"Break-like links (O3'(i)–P(i+1)) : {total_breaks}")
    print(f"Missing key atoms (per-residue audit) : {total_missing}")
    if miss_lines:
        print("- Missing atoms per residue:")
        for line in miss_lines:
            print("  " + line)
    if report_lines:
        print("- Suspect/broken links:")
        for line in report_lines:
            print("  " + line)
    else:
        print("- No suspect links found. Backbone looks continuous under this criterion.")
    print("="*60)

# make command available in PyMOL
cmd.extend("check_rna_backbone", check_rna_backbone)
