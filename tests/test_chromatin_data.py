"""ATAC peaks to the configuration layer (monod.chromatin_data)."""
import numpy as np
import pandas as pd
import pytest

from monod import chromatin_data as cd


def _peaks():
    # chr1: a promoter cluster around 10 000 (three peaks within 1 kb), one far upstream, one in the body
    names = ["chr1:9000-9400", "chr1:9800-10200", "chr1:10500-10900", "chr1:2000-2300", "chr1:15000-15400",
             "chr2:5000-5300", "chr2:30000-30400"]
    return pd.DataFrame(index=names)


def _genes():
    return pd.DataFrame({"gene": ["A", "B", "C"], "chrom": ["chr1", "chr2", "chr1"],
                         "start": [10000, 4800, 15200], "end": [20000, 6000, 15300], "strand": ["+", "-", "+"]})


class TestInputs:
    def test_peak_names_parse_both_separators(self):
        var = pd.DataFrame(index=["chr1:100-200", "chrX-300-450"])
        p = cd.peaks_from_var(var)
        assert list(p["chrom"]) == ["chr1", "chrX"] and list(p["start"]) == [100, 300] and list(p["end"]) == [200, 450]
        with pytest.raises(ValueError):
            cd.peaks_from_var(pd.DataFrame(index=["peak1"]))
        cols = cd.peaks_from_var(pd.DataFrame({"Chromosome": ["chr3"], "Start": [1], "End": [9]}, index=["p"]))
        assert cols.loc["p", "chrom"] == "chr3"

    def test_gtf_reader(self, tmp_path):
        gtf = tmp_path / "g.gtf"
        gtf.write_text("#c\nchr1\tsrc\tgene\t10001\t20000\t.\t+\t.\tgene_id \"G1\"; gene_name \"A\";\n"
                       "chr1\tsrc\texon\t10001\t10100\t.\t+\t.\tgene_id \"G1\";\n"
                       "chr2\tsrc\tgene\t4801\t6000\t.\t-\t.\tgene_id \"G2\";\n")
        g = cd.read_gtf_genes(str(gtf))
        assert list(g["gene"]) == ["A", "G2"] and list(g["start"]) == [10000, 4800] and list(g["strand"]) == ["+", "-"]
        t = cd.genes_table(g)
        assert list(t["tss"]) == [10000, 6000]


class TestAssignment:
    def test_tss_window_keeps_nearest_and_orders_along_transcription(self):
        a = cd.assign_peaks(_peaks(), _genes(), rule="tss_window", window=1500, n_sites=2)
        A = a[a["gene"] == "A"]
        # three peaks within 1.5 kb of 10 000; the two nearest are the promoter peak (distance 0) and 10500-10900
        assert list(A["peak"]) == ["chr1:9800-10200", "chr1:10500-10900"]
        assert list(A["site"]) == [0, 1] and list(A["distance"]) == [0, 500]
        # gene B on the minus strand (tss 6000): only one peak within reach, so it is dropped with n_sites=2
        assert "B" not in set(a["gene"])
        one = cd.assign_peaks(_peaks(), _genes(), rule="tss_window", window=1500, n_sites=1)
        assert list(one[one["gene"] == "B"]["peak"]) == ["chr2:5000-5300"]

    def test_minus_strand_site_order_is_upstream_first(self):
        peaks = pd.DataFrame(index=["chr2:5900-6100", "chr2:6400-6600", "chr2:5300-5500"])
        genes = pd.DataFrame({"gene": ["B"], "chrom": ["chr2"], "start": [4800], "end": [6000], "strand": ["-"]})
        a = cd.assign_peaks(peaks, genes, rule="tss_window", window=1000, n_sites=3)
        # transcription runs leftwards from 6000: upstream is the larger coordinate
        assert list(a["peak"]) == ["chr2:6400-6600", "chr2:5900-6100", "chr2:5300-5500"]

    def test_gene_body_and_loci_rules(self):
        body = cd.assign_peaks(_peaks(), _genes(), rule="gene_body", upstream=500, max_sites=6)
        A = body[body["gene"] == "A"]
        assert set(A["peak"]) == {"chr1:9800-10200", "chr1:10500-10900", "chr1:15000-15400"}
        loci = cd.assign_peaks(_peaks(), _genes(), rule="loci_promoter", upstream=300, downstream=100, locus_gap=1000, max_sites=6)
        A = loci[loci["gene"] == "A"]
        # the promoter peak's locus is the three-peak run; the far-upstream and body peaks are separate loci
        assert set(A["peak"]) == {"chr1:9000-9400", "chr1:9800-10200", "chr1:10500-10900"}
        assert "C" in set(loci["gene"]) and list(loci[loci["gene"] == "C"]["peak"]) == ["chr1:15000-15400"]
        with pytest.raises(ValueError):
            cd.assign_peaks(_peaks(), _genes(), rule="nearest")

    def test_max_sites_caps_candidates(self):
        a = cd.assign_peaks(_peaks(), _genes(), rule="gene_body", upstream=500, max_sites=2)
        assert len(a[a["gene"] == "A"]) == 2


class TestConfiguration:
    def test_encoding_and_threshold(self):
        names = ["p0", "p1", "p2"]
        X = np.array([[0, 2, 0], [1, 0, 5], [3, 3, 0]])
        assignments = pd.DataFrame({"gene": ["G", "G", "H"], "site": [0, 1, 0], "peak": ["p1", "p0", "p2"],
                                    "chrom": "c", "start": 0, "end": 1, "distance": 0})
        genes, config = cd.configuration_layer((X, names), assignments, n_sites=2)
        assert genes == ["G"]                                          # H has one site only
        # G: site 0 = p1, site 1 = p0 -> cell0: p1 open -> 1; cell1: p0 open -> 2; cell2: both -> 3
        np.testing.assert_array_equal(config[:, 0], [1, 2, 3])
        _, strict = cd.configuration_layer((X, names), assignments, n_sites=2, threshold=3)
        np.testing.assert_array_equal(strict[:, 0], [0, 0, 3])

    def test_build_multiome_adata_end_to_end(self):
        import anndata as ad
        from monod.cme_toolbox import CMEModel
        from monod.extract_data import extract_data, grid_limits
        rng = np.random.default_rng(0)
        n_cells = 300
        cells = [f"c{i}" for i in range(n_cells)]
        peaks = _peaks()
        atac = ad.AnnData(X=rng.poisson(0.6, size=(n_cells, len(peaks))).astype(float),
                          obs=pd.DataFrame(index=cells), var=pd.DataFrame(index=peaks.index))
        rna_genes = ["A", "B", "C", "D"]
        rna = ad.AnnData(X=np.zeros((n_cells, 4)), obs=pd.DataFrame(index=cells[::-1]), var=pd.DataFrame(index=rna_genes),
                         layers={"unspliced": rng.poisson(3, size=(n_cells, 4)), "spliced": rng.poisson(8, size=(n_cells, 4))})
        out = cd.build_multiome_adata(rna, atac, _genes(), rule="tss_window", n_sites=2, window=1500)
        assert list(out.var_names) == ["A"] and out.n_obs == n_cells
        assert set(out.layers) == {"unspliced", "spliced", "configuration"}
        assert out.layers["configuration"].dtype == np.int64 and out.layers["configuration"].max() <= 3
        # the configuration agrees with a direct encoding from the ATAC counts in the RNA's cell order
        sites = out.uns["chromatin_sites"]
        p0, p1 = [sites[sites["site"] == s]["peak"].iloc[0] for s in (0, 1)]
        atac_r = atac[out.obs_names]
        manual = (atac_r[:, p0].X.ravel() >= 1).astype(int) + 2 * (atac_r[:, p1].X.ravel() >= 1).astype(int)
        np.testing.assert_array_equal(out.layers["configuration"][:, 0], manual)
        np.testing.assert_array_equal(out.layers["spliced"][:, 0], rna[out.obs_names, "A"].layers["spliced"].ravel())
        summary = cd.site_summary(out)
        assert list(summary["site"]) == [0, 1] and 0 < summary["open_fraction"].iloc[0] < 1
        # and monod's extraction accepts it with the configuration axis at four states
        model = CMEModel("Chromatin", "None", n_sites=2, rule="switch")
        m_adata = extract_data(out, model, n_genes=1, viz=False, dataset_name="chrom_data", filt_param={
            "min_means": [0, 0, 0], "max_maxes": [1000, 1000, 4], "min_maxes": [0, 0, 0]})
        assert grid_limits(m_adata)[2, 0] == 4
        with pytest.raises(ValueError):
            cd.build_multiome_adata(rna, atac, _genes(), rule="tss_window", n_sites=4, window=1500)   # A has three candidates
