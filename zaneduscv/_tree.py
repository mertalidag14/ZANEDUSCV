"""ZANEDUSCV agaci.

Bir agac, gradyan (g) ve hessian (h) ile Newton adimina gore kuruluyor.
Uc tur bolme var:
  0 = sayisal (histogram kutusu esigi, eksikler icin yon ogreniliyor)
  1 = kategorik (kategoriler G/H oranina gore siralanip en iyi gruplama bulunuyor)
  2 = EGIK bolme: iki sayisal kolonun agirlikli toplami  w1*z_i + w2*z_j <= esik.
      Agirliklar, o dugumdeki gradyani en iyi aciklayan yon olarak (agirlikli en kucuk
      kareler ile) hesaplaniyor. Boylece capraz sinirlar tek bolmeyle yakalanabiliyor.

Yapraklar:
  - lineer: b + w . z_yol   (yol uzerinde kullanilan sayisal kolonlarla kucuk bir ridge modeli)
  - guven katsayisi: yaprak cikisi n / (n + kappa) ile carpiliyor; az ornekli yapraklar
    modeli daha az etkiliyor.
"""

import numpy as np

NUMERIC, CATEGORICAL, OBLIQUE = 0, 1, 2


class Tree:
    def __init__(self, max_depth=6, min_samples_leaf=5, min_child_weight=1e-3, reg_lambda=1.0,
                 gamma=0.0, oblique=True, oblique_top=3, linear_leaves=True, linear_lambda=1.0,
                 min_samples_linear=30, confidence_kappa=10.0):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.oblique = oblique
        self.oblique_top = oblique_top
        self.linear_leaves = linear_leaves
        self.linear_lambda = linear_lambda
        self.min_samples_linear = min_samples_linear
        self.confidence_kappa = confidence_kappa

    # ------------------------------------------------------------------ yardimci
    def _score(self, G, H):
        return G * G / (H + self.reg_lambda)

    def _new_node(self):
        for arr in (self.kind, self.feat, self.feat2, self.w1, self.w2, self.thr,
                    self.miss_left, self.left, self.right, self.cat_mask, self.leaf_id):
            arr.append(None)
        i = len(self.kind) - 1
        self.left[i] = self.right[i] = -1
        self.leaf_id[i] = -1
        return i

    # ---------------------------------------------------------------------- fit
    def fit(self, B, Z, g, h, is_cat, n_bins, features, n_total_bins):
        self.kind, self.feat, self.feat2, self.w1, self.w2, self.thr = [], [], [], [], [], []
        self.miss_left, self.left, self.right, self.cat_mask, self.leaf_id = [], [], [], [], []
        self.leaf_const, self.leaf_feats, self.leaf_coef = [], [], []
        self.gain_by_feature = np.zeros(B.shape[1])
        self._B, self._Z, self._g, self._h = B, Z, g, h
        self._is_cat, self._n_bins, self._nb = is_cat, n_bins, n_total_bins
        self._features = np.asarray(features)
        self._num_features = self._features[~is_cat[self._features]]
        self._num_pos = np.where(~is_cat[self._features])[0]
        self._cat_pos = np.where(is_cat[self._features])[0]
        self._grow(np.arange(len(g)), 0, [], None)
        del self._B, self._Z, self._g, self._h
        self.kind = np.array([k if k is not None else -1 for k in self.kind])
        return self

    def _histograms(self, idx):
        Bn = self._B[idx][:, self._features].astype(np.int64)
        k = len(self._features)
        flat = (Bn + np.arange(k) * self._nb).ravel()
        size = k * self._nb
        gw = np.repeat(self._g[idx], k)
        hw = np.repeat(self._h[idx], k)
        G = np.bincount(flat, weights=gw, minlength=size).reshape(k, self._nb)
        H = np.bincount(flat, weights=hw, minlength=size).reshape(k, self._nb)
        C = np.bincount(flat, minlength=size).reshape(k, self._nb)
        return G, H, C

    def _valid(self, GL, HL, CL, GT, HT, CT):
        m, w = self.min_samples_leaf, self.min_child_weight
        return (CL >= m) & (CT - CL >= m) & (HL >= w) & (HT - HL >= w)

    def _best_split(self, idx, GT, HT, CT, hist):
        parent = self._score(GT, HT)
        best = {"gain": self.gamma, "kind": None}
        G, H, C = hist
        lam = self.reg_lambda

        # --- sayisal kolonlar: hepsi tek seferde (vektorel)
        npos = self._num_pos
        if len(npos) and self._nb > 2:
            gl = np.cumsum(G[npos, 1:], axis=1)[:, :-1]
            hl = np.cumsum(H[npos, 1:], axis=1)[:, :-1]
            cl = np.cumsum(C[npos, 1:], axis=1)[:, :-1]
            g0, h0, c0 = G[npos, :1], H[npos, :1], C[npos, :1]
            for miss_left in (True, False):
                GL, HL, CL = (gl + g0, hl + h0, cl + c0) if miss_left else (gl, hl, cl)
                gains = GL * GL / (HL + lam) + (GT - GL) ** 2 / (HT - HL + lam) - parent
                gains[~self._valid(GL, HL, CL, GT, HT, CT)] = -np.inf
                r, t = np.unravel_index(np.argmax(gains), gains.shape)
                if gains[r, t] > best["gain"]:
                    best = {"gain": gains[r, t], "kind": NUMERIC, "feat": int(self._features[npos[r]]),
                            "miss_left": miss_left, "thr": int(t + 1)}   # sol: kutu <= t+1

        # --- kategorik kolonlar
        for pos in self._cat_pos:
            j = self._features[pos]
            nb = self._n_bins[j]
            bins = np.where(C[pos, 1:nb] > 0)[0] + 1
            if len(bins) < 2:
                continue
            ratio = G[pos, bins] / (H[pos, bins] + lam)
            order = bins[np.argsort(ratio)]
            gl = np.cumsum(G[pos, order])[:-1]
            hl = np.cumsum(H[pos, order])[:-1]
            cl = np.cumsum(C[pos, order])[:-1]
            for miss_left in (True, False):
                GL = gl + (G[pos, 0] if miss_left else 0)
                HL = hl + (H[pos, 0] if miss_left else 0)
                CL = cl + (C[pos, 0] if miss_left else 0)
                gains = self._score(GL, HL) + self._score(GT - GL, HT - HL) - parent
                gains = np.where(self._valid(GL, HL, CL, GT, HT, CT), gains, -np.inf)
                t = int(np.argmax(gains))
                if gains[t] > best["gain"]:
                    mask = np.zeros(self._nb, dtype=bool)
                    mask[order[: t + 1]] = True
                    best = {"gain": gains[t], "kind": CATEGORICAL, "feat": int(j),
                            "miss_left": miss_left, "cat_mask": mask}

        if self.oblique and len(self._num_features) >= 2 and len(idx) >= 4 * self.min_samples_leaf:
            ob = self._best_oblique(idx, GT, HT, CT, parent)
            if ob is not None and ob["gain"] > best["gain"]:
                best = ob
        return best

    def _best_oblique(self, idx, GT, HT, CT, parent):
        Zn = self._Z[idx][:, self._num_features]
        g, h = self._g[idx], self._h[idx]
        lam = self.reg_lambda
        target = -g / np.maximum(h, 1e-12)
        # gradyanla en ilgili sayisal kolonlari sec (h agirlikli korelasyon)
        hs = h / HT
        zc = Zn - hs @ Zn
        tc = target - hs @ target
        hz = h[:, None] * zc
        cov = np.abs(tc @ hz)
        var = np.einsum("ij,ij->j", hz, zc)
        top = np.argsort(-(cov / np.sqrt(var + 1e-12)))[: self.oblique_top]
        best = None
        cl = np.arange(1, len(idx))
        for a in range(len(top)):
            for b in range(a + 1, len(top)):
                i, j = top[a], top[b]
                # 2x2 agirlikli en kucuk kareler (kapali form)
                s11, s22 = var[i] + 1e-6, var[j] + 1e-6
                s12 = hz[:, i] @ zc[:, j]
                r1, r2 = tc @ hz[:, i], tc @ hz[:, j]
                det = s11 * s22 - s12 * s12
                if det <= 1e-12:
                    continue
                w = np.array([s22 * r1 - s12 * r2, s11 * r2 - s12 * r1]) / det
                norm = np.abs(w).sum()
                if norm < 1e-12:
                    continue
                w = w / norm
                if min(abs(w[0]), abs(w[1])) < 0.1:      # neredeyse tek kolon: normal bolme yeterli
                    continue
                proj = Zn[:, i] * w[0] + Zn[:, j] * w[1]
                order = np.argsort(proj)
                ps = proj[order]
                GL = np.cumsum(g[order])[:-1]
                HL = np.cumsum(h[order])[:-1]
                gains = GL * GL / (HL + lam) + (GT - GL) ** 2 / (HT - HL + lam) - parent
                ok = self._valid(GL, HL, cl, GT, HT, CT) & (ps[:-1] < ps[1:])
                gains[~ok] = -np.inf
                t = int(np.argmax(gains))
                if np.isfinite(gains[t]) and (best is None or gains[t] > best["gain"]):
                    best = {"gain": gains[t], "kind": OBLIQUE,
                            "feat": int(self._num_features[i]),
                            "feat2": int(self._num_features[j]),
                            "w1": float(w[0]), "w2": float(w[1]),
                            "thr": float(0.5 * (ps[t] + ps[t + 1])), "miss_left": True}
        return best

    def _route(self, node, B, Z, rows):
        k = self.kind[node]
        if k == OBLIQUE:
            z = self.w1[node] * Z[rows, self.feat[node]] + self.w2[node] * Z[rows, self.feat2[node]]
            return z <= self.thr[node]
        b = B[rows, self.feat[node]]
        if k == CATEGORICAL:
            go = self.cat_mask[node][b]
        else:
            go = b <= self.thr[node]
        return np.where(b == 0, self.miss_left[node], go)

    def _grow(self, idx, depth, path_feats, hist):
        node = self._new_node()
        g, h = self._g[idx], self._h[idx]
        GT, HT, CT = g.sum(), h.sum(), len(idx)
        if depth < self.max_depth and CT >= 2 * self.min_samples_leaf:
            if hist is None:
                hist = self._histograms(idx)
            split = self._best_split(idx, GT, HT, CT, hist)
            if split["kind"] is not None:
                for key in ("kind", "feat", "miss_left"):
                    getattr(self, key)[node] = split[key]
                self.thr[node] = split.get("thr")
                self.cat_mask[node] = split.get("cat_mask")
                self.feat2[node] = split.get("feat2")
                self.w1[node], self.w2[node] = split.get("w1"), split.get("w2")
                if split["kind"] == OBLIQUE:
                    self.gain_by_feature[split["feat"]] += split["gain"] * abs(split["w1"])
                    self.gain_by_feature[split["feat2"]] += split["gain"] * abs(split["w2"])
                    new_feats = [split["feat"], split["feat2"]]
                else:
                    self.gain_by_feature[split["feat"]] += split["gain"]
                    new_feats = [] if split["kind"] == CATEGORICAL else [split["feat"]]
                self.kind[node] = split["kind"]
                go_left = self._route(node, self._B, self._Z, idx)
                feats = path_feats + [f for f in new_feats if f not in path_feats]
                li, ri = idx[go_left], idx[~go_left]
                # histogram cikarma hilesi: kucuk cocugu hesapla, buyugu = ebeveyn - kucuk
                if len(li) <= len(ri):
                    hl = self._histograms(li)
                    hr = tuple(p - c for p, c in zip(hist, hl))
                else:
                    hr = self._histograms(ri)
                    hl = tuple(p - c for p, c in zip(hist, hr))
                self.left[node] = self._grow(li, depth + 1, feats, hl)
                self.right[node] = self._grow(ri, depth + 1, feats, hr)
                return node
        self._make_leaf(node, idx, GT, HT, CT, path_feats)
        return node

    def _make_leaf(self, node, idx, GT, HT, CT, path_feats):
        conf = CT / (CT + self.confidence_kappa)
        const = -GT / (HT + self.reg_lambda)
        feats, coef = [], None
        if self.linear_leaves and path_feats and CT >= self.min_samples_linear:
            feats = list(path_feats)
            X = np.column_stack([np.ones(CT), self._Z[idx][:, feats]])
            h, g = self._h[idx], self._g[idx]
            A = X.T @ (h[:, None] * X)
            reg = np.full(X.shape[1], self.linear_lambda)
            reg[0] = self.reg_lambda
            A[np.diag_indices_from(A)] += reg
            try:
                sol = np.linalg.solve(A, -(X.T @ g))
                const, coef = sol[0], sol[1:]
            except np.linalg.LinAlgError:
                feats = []
        self.leaf_id[node] = len(self.leaf_const)
        self.leaf_const.append(conf * const)
        self.leaf_feats.append(feats)
        self.leaf_coef.append(None if coef is None else conf * coef)

    # ------------------------------------------------------------------ predict
    def apply(self, B, Z):
        nodes = np.zeros(len(B), dtype=int)
        active = self.kind[nodes] >= 0
        while active.any():
            a = np.where(active)[0]
            for node in np.unique(nodes[a]):
                rows = a[nodes[a] == node]
                go = self._route(node, B, Z, rows)
                nodes[rows] = np.where(go, self.left[node], self.right[node])
            active = self.kind[nodes] >= 0
        return nodes

    def predict(self, B, Z):
        nodes = self.apply(B, Z)
        out = np.zeros(len(B))
        for node in np.unique(nodes):
            rows = np.where(nodes == node)[0]
            lid = self.leaf_id[node]
            out[rows] = self.leaf_const[lid]
            if self.leaf_coef[lid] is not None:
                out[rows] += Z[rows][:, self.leaf_feats[lid]] @ self.leaf_coef[lid]
        return out
