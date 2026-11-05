"""
Shared utilities that treat the ChefBoost C4.5 rules file (outputs/rules/rules.py)
as the single source of truth for:

  * routing an instance to the exact leaf it reaches,
  * per-leaf class counts (support) on any dataset,
  * leaf-frequency probability scores for ROC / PR analysis,
  * named-pathway audits (Table 4.10), and
  * the Figure 4.5 tree diagram.

Nothing in this module hard-codes tree structure or counts: everything is parsed
from the rules file and counted from the data.

Leaf probability score
----------------------
ChefBoost only returns class labels. A continuous score is obtained the standard
way for probability-estimation trees: the Laplace-corrected positive frequency of
the training records that reach the leaf, p = (k + 1) / (n + 2), where n is the
number of training records in the leaf and k the number of positives
(Provost & Domingos, 2003). ChefBoost also writes "else" fallback leaves for
category values never seen at that node during training; such leaves hold
n = 0 training records and, following C4.5's handling of empty leaves, they
inherit the estimate of their nearest ancestor with n > 0.

Value labels below were transcribed from the DHS recode dictionaries shipped with
the data (NGKR81FL.MAP and NGPR81FL.MAP), not from memory.
"""

import ast
import os
import shutil
import tempfile

import numpy as np
import pandas as pd

FEATURES = ['b19', 'b4', 'v106', 'v190', 'v119', 'v158', 'v127', 'v128',
            'v129', 'v113', 'v116', 'hml20']
NOMINAL_FEATURES = FEATURES[1:]

FEATURE_LABELS = {
    'b19': "Child's age (months)",
    'b4': "Sex of child",
    'v106': "Maternal education",
    'v190': "Wealth quintile",
    'v119': "Household electricity",
    'v158': "Radio listening frequency",
    'v127': "Floor material",
    'v128': "Wall material",
    'v129': "Roof material",
    'v113': "Drinking-water source",
    'v116': "Toilet facility",
    'hml20': "Slept under LLIN",
}

# Source: NGKR81FL.MAP (v-variables, b4) and NGPR81FL.MAP (hml20).
VALUE_LABELS = {
    'b4': {'1': 'Male', '2': 'Female'},
    'v106': {'0': 'No education', '1': 'Primary', '2': 'Secondary', '3': 'Higher'},
    'v190': {'1': 'Poorest', '2': 'Poorer', '3': 'Middle', '4': 'Richer', '5': 'Richest'},
    'v119': {'0': 'No', '1': 'Yes'},
    'v158': {'0': 'Not at all', '1': 'Less than once a week',
             '2': 'At least once a week', '3': 'Almost every day'},
    'v127': {'11': 'Earth/sand', '12': 'Dung', '21': 'Wood planks', '22': 'Palm/bamboo',
             '31': 'Parquet/polished wood', '32': 'Vinyl/asphalt strips',
             '33': 'Ceramic tiles', '34': 'Cement', '35': 'Carpet', '96': 'Other'},
    'v128': {'11': 'No walls', '12': 'Cane/palm/trunks', '13': 'Dirt',
             '21': 'Bamboo with mud', '22': 'Stone with mud', '23': 'Uncovered adobe',
             '24': 'Plywood', '25': 'Cardboard', '26': 'Reused wood', '31': 'Cement',
             '32': 'Stone with lime/cement', '33': 'Bricks', '34': 'Cement blocks',
             '35': 'Covered adobe', '36': 'Wood planks/shingles', '96': 'Other'},
    'v129': {'11': 'No roof', '12': 'Thatch/palm leaf', '13': 'Grass', '21': 'Rustic mat',
             '22': 'Palm/bamboo', '23': 'Wood planks', '24': 'Cardboard',
             '31': 'Metal/zinc', '32': 'Wood', '33': 'Calamine/cement fibre',
             '34': 'Ceramic tiles', '35': 'Cement', '36': 'Roofing shingles',
             '37': 'Asbestos', '96': 'Other'},
    'v113': {'11': 'Piped into dwelling', '12': 'Piped to yard/plot',
             '13': 'Piped to neighbour', '14': 'Public tap/standpipe',
             '21': 'Tube well or borehole', '31': 'Protected well',
             '32': 'Unprotected well', '41': 'Protected spring',
             '42': 'Unprotected spring',
             '43': 'Surface water (river/dam/lake/pond/stream)',
             '51': 'Rainwater', '61': 'Tanker truck', '62': 'Cart with small tank',
             '71': 'Bottled water', '72': 'Sachet water', '96': 'Other'},
    'v116': {'11': 'Flush to piped sewer', '12': 'Flush to septic tank',
             '13': 'Flush to pit latrine', '14': 'Flush to somewhere else',
             '15': "Flush, don't know where", '21': 'VIP latrine',
             '22': 'Pit latrine with slab', '23': 'Pit latrine without slab/open pit',
             '31': 'No facility/bush/field', '41': 'Composting toilet',
             '42': 'Bucket toilet', '43': 'Hanging toilet/latrine', '96': 'Other'},
    'hml20': {'0': 'No', '1': 'Yes'},
}

# Named pathways reported in the thesis (Table 4.10). Each is defined by the exact
# sequence of conditions on its path in rules.py; resolve_pathways() fails loudly
# if a definition does not match a leaf of the current tree.
PATHWAY_DEFINITIONS = {
    'C1': [[('b19', '<=', '6')]],
    'C2': [[('b19', '>', '6'), ('v158', '==', '0'), ('hml20', '==', '0'),
            ('v113', '==', '31'), ('v129', '==', '31')]],
    'C3': [[('b19', '>', '6'), ('v158', '==', '0'), ('hml20', '==', '0'),
            ('v113', '==', '43'), ('v106', '==', '0')]],
    'C4': [[('b19', '>', '6'), ('v158', '==', '0'), ('hml20', '==', '0'),
            ('v113', '==', '21'), ('v128', '==', '21')],
           [('b19', '>', '6'), ('v158', '==', '0'), ('hml20', '==', '0'),
            ('v113', '==', '32'), ('v128', '==', '21')]],
    'C5': [[('b19', '>', '6'), ('v158', '==', '0'), ('hml20', '==', '1'),
            ('v119', '==', '1'), ('v190', '==', '2')]],
}

PATHWAY_NAMES = {
    'C1': 'Age exactly 6 months (b19 <= 6)',
    'C2': 'No radio, no net, protected well, metal/zinc roof',
    'C3': 'No radio, no net, surface-water source, no maternal education',
    'C4': 'No radio, no net, borehole or unprotected well, bamboo-with-mud walls',
    'C5': 'No radio, net use, electricity, wealth quintile 2 (Poorer)',
}


# ------------------------------------------------------------------------------
# Data formatting
# ------------------------------------------------------------------------------
def format_for_c45(X):
    """Cast a feature frame exactly as fourth_step.py does before ChefBoost:
    b19 -> int, nominal codes -> str of int."""
    X = X[FEATURES].copy()
    X['b19'] = pd.to_numeric(X['b19']).astype(int)
    for col in NOMINAL_FEATURES:
        X[col] = pd.to_numeric(X[col]).astype(int).astype(str).astype(object)
    return X


def rows_as_obj(X):
    """Return a list of ChefBoost 'obj' lists (feature order of rules.py)."""
    Xf = format_for_c45(X)
    return [list(r) for r in Xf.itertuples(index=False, name=None)]


# ------------------------------------------------------------------------------
# Parsing rules.py into an explicit tree
# ------------------------------------------------------------------------------
class Node:
    """Internal split node. branches: list of (condition, child). A condition is
    (feature, op, value) with op in {'==', '>', '<=', 'else'}; for 'else' the value
    is the tuple of values tested by the preceding branches."""

    def __init__(self, feature):
        self.feature = feature
        self.branches = []
        self.parent = None


class Leaf:
    def __init__(self, label, leaf_id, path):
        self.label = label
        self.id = leaf_id
        self.path = path          # list of (feature, op, value)
        self.parent = None


_OPS = {ast.Eq: '==', ast.Gt: '>', ast.LtE: '<=', ast.Lt: '<', ast.GtE: '>='}


def _parse_test(test):
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1
            and isinstance(test.left, ast.Subscript)):
        raise ValueError(f"Unsupported condition in rules file: {ast.unparse(test)}")
    idx = test.left.slice.value
    op = _OPS[type(test.ops[0])]
    val = test.comparators[0].value
    return FEATURES[idx], op, str(val)


def _parse_block(stmts, path, leaves):
    if len(stmts) != 1:
        raise ValueError("Expected exactly one statement per block in rules file")
    st = stmts[0]
    if isinstance(st, ast.Return):
        leaf = Leaf(st.value.value, len(leaves), list(path))
        leaves.append(leaf)
        return leaf
    if not isinstance(st, ast.If):
        raise ValueError(f"Unexpected statement in rules file: {ast.unparse(st)}")

    feature = _parse_test(st.test)[0]
    node = Node(feature)
    tested = []
    cur = st
    while True:
        cond = _parse_test(cur.test)
        if cond[0] != feature:
            raise ValueError("Mixed features within one if/elif chain")
        tested.append(cond[2])
        child = _parse_block(cur.body, path + [cond], leaves)
        child.parent = node
        node.branches.append((cond, child))
        if len(cur.orelse) == 1 and isinstance(cur.orelse[0], ast.If):
            cur = cur.orelse[0]
            continue
        if cur.orelse:
            cond = (feature, 'else', tuple(tested))
            child = _parse_block(cur.orelse, path + [cond], leaves)
            child.parent = node
            node.branches.append((cond, child))
        break
    return node


class C45Tree:
    def __init__(self, source):
        self.source = source
        module = ast.parse(source)
        fn = [n for n in module.body if isinstance(n, ast.FunctionDef)
              and n.name == 'findDecision'][0]
        body = [s for s in fn.body if not isinstance(s, ast.Expr)]
        self.leaves = []
        self.root = _parse_block(body, [], self.leaves)
        # Reference implementation: execute the original source unchanged
        ns = {}
        exec(compile(source, '<rules>', 'exec'), ns)
        self._find_decision = ns['findDecision']

    @classmethod
    def from_file(cls, path=os.path.join('outputs', 'rules', 'rules.py')):
        with open(path, 'r', encoding='utf-8') as f:
            return cls(f.read())

    # -- routing ---------------------------------------------------------------
    @staticmethod
    def _match(cond, value):
        feat, op, v = cond
        if op == '==':
            return value == v
        if op == '>':
            return value > float(v)
        if op == '<=':
            return value <= float(v)
        if op == '<':
            return value < float(v)
        if op == '>=':
            return value >= float(v)
        if op == 'else':
            return True
        raise ValueError(op)

    def route(self, obj):
        node = self.root
        while isinstance(node, Node):
            idx = FEATURES.index(node.feature)
            for cond, child in node.branches:
                if self._match(cond, obj[idx]):
                    node = child
                    break
            else:
                raise RuntimeError("Instance fell through a node without an else branch")
        return node

    def leaf_ids(self, X):
        objs = rows_as_obj(X)
        ids = np.array([self.route(o).id for o in objs])
        # Guarantee equivalence with the untouched ChefBoost rules function
        ref = [self._find_decision(list(o)) for o in objs]
        mine = [self.leaves[i].label for i in ids]
        if ref != mine:
            raise AssertionError("Parsed tree disagrees with rules.py findDecision()")
        return ids

    def predict(self, X):
        ids = self.leaf_ids(X)
        return np.array([1 if self.leaves[i].label == 'Positive' else 0 for i in ids])

    # -- counts and scores -----------------------------------------------------
    def leaf_counts(self, X, y):
        """Return DataFrame indexed by leaf id with n, pos, neg."""
        ids = self.leaf_ids(X)
        y = np.asarray(y).astype(int)
        n = np.bincount(ids, minlength=len(self.leaves))
        pos = np.bincount(ids, weights=y, minlength=len(self.leaves)).astype(int)
        return pd.DataFrame({'n': n, 'pos': pos, 'neg': n - pos})

    def node_counts(self, X, y):
        """Counts for every node (internal and leaf), keyed by id(node)."""
        lc = self.leaf_counts(X, y)
        out = {}

        def rec(node):
            if isinstance(node, Leaf):
                c = (int(lc.loc[node.id, 'n']), int(lc.loc[node.id, 'pos']))
            else:
                parts = [rec(ch) for _, ch in node.branches]
                c = (sum(p[0] for p in parts), sum(p[1] for p in parts))
            out[id(node)] = c
            return c

        rec(self.root)
        return out

    def fit_leaf_scores(self, X_train, y_train):
        """Laplace-corrected leaf positive frequency; empty leaves inherit from the
        nearest ancestor holding training records."""
        counts = self.node_counts(X_train, y_train)
        scores = {}
        for leaf in self.leaves:
            node = leaf
            while counts[id(node)][0] == 0 and node.parent is not None:
                node = node.parent
            n, k = counts[id(node)]
            scores[leaf.id] = (k + 1.0) / (n + 2.0)
        self.leaf_scores_ = scores
        return scores

    def predict_score(self, X):
        ids = self.leaf_ids(X)
        return np.array([self.leaf_scores_[i] for i in ids])

    # -- descriptions ----------------------------------------------------------
    @staticmethod
    def describe_condition(cond, with_labels=True):
        feat, op, v = cond
        if op == 'else':
            if feat not in NOMINAL_FEATURES:
                return (f"{feat} fails every preceding threshold test "
                        f"(unreachable for non-missing numeric values)")
            vals = ', '.join(v)
            return f"{feat} not in {{{vals}}} (value unseen at this node in training)"
        if op == '==' and with_labels and feat in VALUE_LABELS and v in VALUE_LABELS[feat]:
            return f"{feat} = {v} ({VALUE_LABELS[feat][v]})"
        return f"{feat} {op} {v}"

    def describe_path(self, leaf, with_labels=True):
        return ' AND '.join(self.describe_condition(c, with_labels) for c in leaf.path)

    def resolve_pathways(self, definitions=PATHWAY_DEFINITIONS):
        """Map each named pathway to its leaf ids; raise if any path is missing."""
        by_path = {tuple(l.path): l for l in self.leaves}
        resolved = {}
        for name, paths in definitions.items():
            ids = []
            for p in paths:
                key = tuple(tuple(c) for c in p)
                if key not in by_path:
                    raise KeyError(f"Pathway {name} path {p} is not a leaf of the current tree")
                ids.append(by_path[key].id)
            resolved[name] = ids
        return resolved


# ------------------------------------------------------------------------------
# Fitting ChefBoost without overwriting the project's outputs/rules/rules.py
# ------------------------------------------------------------------------------
def fit_chefboost_isolated(X_train, y_train):
    """Fit ChefBoost C4.5 inside a temporary working directory and return a
    C45Tree parsed from the rules file it wrote. Used by the multi-seed and
    grouped-CV experiments so that the main rules.py is never touched."""
    from chefboost import Chefboost as chef

    df = format_for_c45(X_train)
    df['Decision'] = pd.Series(np.asarray(y_train).astype(int), index=df.index).map(
        {0: 'Negative', 1: 'Positive'}).astype(object)

    old_cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix='chefboost_')
    try:
        os.chdir(tmp)
        chef.fit(df, config={'algorithm': 'C4.5'}, target_label='Decision', silent=True)
        with open(os.path.join('outputs', 'rules', 'rules.py'), 'r', encoding='utf-8') as f:
            source = f.read()
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(tmp, ignore_errors=True)
    return C45Tree(source)


# ------------------------------------------------------------------------------
# Figure 4.5: draw the tree exactly as parsed, with counts from the data
# ------------------------------------------------------------------------------
# Short forms of the DHS value labels above, used only to fit the printed figure.
SHORT_VALUE_LABELS = {
    'v158': {'0': 'Never', '1': '< once a week', '2': '>= once a week', '3': 'Almost daily'},
    'hml20': {'0': 'No net', '1': 'Net used'},
    'v113': {'21': 'Borehole', '31': 'Protected well', '32': 'Unprot. well',
             '43': 'Surface water', '51': 'Rainwater'},
    'v129': {'12': 'Thatch/palm', '13': 'Grass', '22': 'Palm/bamboo', '31': 'Metal/zinc'},
    'v106': {'0': 'No education', '1': 'Primary', '2': 'Secondary', '3': 'Higher'},
    'v128': {'11': 'No walls', '13': 'Dirt', '21': 'Bamboo + mud', '22': 'Stone + mud',
             '31': 'Cement'},
    'v119': {'0': 'No electricity', '1': 'Electricity'},
    'v190': {'1': 'Poorest', '2': 'Poorer', '3': 'Middle', '4': 'Richer', '5': 'Richest'},
    'v116': {'12': 'Flush septic', '22': 'Pit w/ slab'},
    'b4': {'1': 'Male', '2': 'Female'},
}
FEATURE_SHORT = {'b19': 'age', 'b4': 'sex', 'v106': 'educ.', 'v190': 'wealth',
                 'v119': 'electric.', 'v158': 'radio', 'v127': 'floor', 'v128': 'walls',
                 'v129': 'roof', 'v113': 'water', 'v116': 'toilet', 'hml20': 'net use'}


def _short_condition(cond):
    feat, op, v = cond
    if op == 'else':
        return f"{feat}: other value"
    if op == '==':
        lab = SHORT_VALUE_LABELS.get(feat, {}).get(v)
        return f"{feat}={v}" + (f" {lab}" if lab else "")
    return f"{feat} {op.replace('<=', '≤')} {v}"


def plot_tree(tree, X_train, y_train, out_paths, n_original=None, pathway_map=None,
              collapse_offpath=False, title=None, width_in=6.3, font=5.2, row_h=0.32):
    """Draw the parsed C4.5 tree left-to-right at print size (width_in inches).

    Each box shows: the condition leading to it; the node (split feature, leaf class,
    or a collapsed subtree); training records reaching it as 'all | real'; and the
    percentage positive as 'all | real'. 'all' is the SMOTE-NC-resampled training
    set; 'real' is its first n_original rows (the original, non-synthetic records).
    Leaves that no training record reaches (unseen-value fallbacks) are not drawn.
    With collapse_offpath=True, internal nodes not on the path to a named pathway
    are drawn as one collapsed box; their counts are still exact.
    Returns the list of undrawn empty leaves.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    y_arr = np.asarray(y_train)
    counts = tree.node_counts(X_train, y_arr)
    counts_real = tree.node_counts(X_train.iloc[:n_original], y_arr[:n_original]) \
        if n_original is not None else None

    leaf_to_pw = {i: name for name, ids in (pathway_map or {}).items() for i in ids}
    on_path = set()
    for leaf in tree.leaves:
        if leaf.id in leaf_to_pw:
            node = leaf
            while node is not None:
                on_path.add(id(node))
                node = node.parent

    def visible(node):
        return counts[id(node)][0] > 0

    def is_collapsed(node):
        return collapse_offpath and isinstance(node, Node) and id(node) not in on_path

    def n_leaves_reached(node):
        if isinstance(node, Leaf):
            return 1 if visible(node) else 0
        return sum(n_leaves_reached(ch) for _, ch in node.branches)

    pos, incoming, omitted = {}, {}, []
    next_row = [0]

    def layout(node, depth, cond):
        incoming[id(node)] = cond
        if isinstance(node, Leaf) or is_collapsed(node):
            pos[id(node)] = (depth, next_row[0])
            next_row[0] += 1
            return
        kids = []
        for c, ch in node.branches:
            if visible(ch):
                kids.append((c, ch))
            elif isinstance(ch, Leaf):
                omitted.append(ch)
        for c, ch in kids:
            layout(ch, depth + 1, c)
        ys = [pos[id(ch)][1] for _, ch in kids]
        pos[id(node)] = (depth, (min(ys) + max(ys)) / 2.0)

    layout(tree.root, 0, None)
    n_rows = next_row[0]
    n_cols = max(p[0] for p in pos.values()) + 1

    col_w = width_in / n_cols
    box_w = col_w - 0.13
    box_h = row_h - 0.06
    top_pad, bottom_pad = 0.42, 0.55
    fig_h = n_rows * row_h + top_pad + bottom_pad
    fig = plt.figure(figsize=(width_in, fig_h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_in)
    ax.set_ylim(fig_h, 0)
    ax.axis('off')

    def xy(node):
        c, r = pos[id(node)]
        return c * col_w + col_w / 2.0, top_pad + r * row_h + row_h / 2.0

    def counts_text(node):
        n, k = counts[id(node)]
        pct = f"{100.0 * k / n:.1f}%" if n else "-"
        if counts_real is None:
            return f"{n} · {pct}"
        rn, rk = counts_real[id(node)]
        rpct = f"{100.0 * rk / rn:.1f}%" if rn else "-"
        return f"{n}|{rn} · {pct}|{rpct}"

    def head_text(node):
        if isinstance(node, Leaf):
            tag = f"{leaf_to_pw[node.id]}: " if node.id in leaf_to_pw else ""
            return f"{tag}Leaf {node.id} → {node.label}"
        if is_collapsed(node):
            return f"{node.feature} subtree, {n_leaves_reached(node)} leaves"
        return f"split: {node.feature} {FEATURE_SHORT[node.feature]}"

    def draw(node):
        x, y = xy(node)
        if isinstance(node, Leaf):
            fc, ec = ('#f8d7da', '#b71c1c') if node.label == 'Positive' else ('#dcefe0', '#2e7d32')
        elif is_collapsed(node):
            fc, ec = '#eeeeee', '#757575'
        else:
            fc, ec = '#e3f2fd', '#1565c0'
        lw = 1.6 if (isinstance(node, Leaf) and node.id in leaf_to_pw) else 0.6
        ls = '--' if is_collapsed(node) else '-'
        ax.add_patch(FancyBboxPatch((x - box_w / 2, y - box_h / 2), box_w, box_h,
                                    boxstyle='round,pad=0.0,rounding_size=0.03', fc=fc, ec=ec,
                                    lw=lw, ls=ls, zorder=3))
        cond = incoming[id(node)]
        first = _short_condition(cond) if cond is not None else "All training records"
        ax.text(x, y, f"{first}\n{head_text(node)}\n{counts_text(node)}", ha='center',
                va='center', fontsize=font, zorder=4, linespacing=1.12,
                fontweight='normal')
        if isinstance(node, Node) and not is_collapsed(node):
            for c, ch in node.branches:
                if not visible(ch):
                    continue
                cx, cy = xy(ch)
                ax.plot([x + box_w / 2, x + box_w / 2 + 0.035, cx - box_w / 2 - 0.035, cx - box_w / 2],
                        [y, y, cy, cy], color='#666666', lw=0.5, zorder=1)
                draw(ch)

    draw(tree.root)

    ax.text(0.04, 0.17, title or "C4.5 (ChefBoost) tree drawn from outputs/rules/rules.py",
            fontsize=font + 2.2, fontweight='bold', va='center', ha='left')
    legend = ("Each box: condition from parent; node; 'n all|real · % positive all|real'. "
              "'all' = SMOTE-NC-resampled training set; 'real' = "
              + (f"its {n_original} original rows. " if n_original else "n/a. ")
              + f"{len(omitted)} fallback leaves for values unseen in training (n = 0) not drawn.")
    if collapse_offpath:
        legend += " Grey dashed boxes: subtrees not on a named pathway, collapsed (counts exact)."
    legend += " Value labels: DHS recode dictionary."
    import textwrap
    ax.text(0.04, fig_h - bottom_pad + 0.1, "\n".join(textwrap.wrap(legend, 165)),
            fontsize=font - 0.3, style='italic', color='#333333', va='top', ha='left')
    for p in out_paths:
        fig.savefig(p, dpi=300)
    plt.close(fig)
    return omitted
