"""Elasticity estimation: every method computed live from the loaded data."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st


def _ols(y, X):
    model = sm.OLS(y, sm.add_constant(X)).fit()
    return model


@st.cache_data
def run_models(monthly: pd.DataFrame):
    d = monthly.copy()
    d["ln_v"] = np.log(d["vol_Explorer"])
    d["ln_p"] = np.log(d["atp_Explorer"])
    d["ln_rel"] = d["ln_p"] - np.log(d["atp_Segment"])

    out = {}

    m1 = _ols(d["ln_v"], d[["ln_p"]])
    out["naive"] = dict(model=m1, e=m1.params["ln_p"], p=m1.pvalues["ln_p"],
                        se=m1.bse["ln_p"], r2=m1.rsquared, n=len(d),
                        label="1. Naive log-log OLS",
                        spec="ln(Volume) = a + e ln(ATP)")

    m2 = _ols(d["ln_v"], d[["ln_p", "refresh"]])
    out["refresh"] = dict(model=m2, e=m2.params["ln_p"], p=m2.pvalues["ln_p"],
                          se=m2.bse["ln_p"], r2=m2.rsquared, n=len(d),
                          label="2. Refresh-controlled OLS (preferred)",
                          spec="ln(Volume) = a + e ln(ATP) + b Refresh")

    m3 = _ols(d["ln_v"], d[["ln_rel", "refresh"]])
    out["relative"] = dict(model=m3, e=m3.params["ln_rel"], p=m3.pvalues["ln_rel"],
                           se=m3.bse["ln_rel"], r2=m3.rsquared, n=len(d),
                           label="3. Relative price vs segment",
                           spec="ln(Volume) = a + e ln(ATP/SegmentATP) + b Refresh")

    yoy = pd.DataFrame({
        "dlv": np.log(d["vol_Explorer"].iloc[12:21].values / d["vol_Explorer"].iloc[0:9].values),
        "dlp": np.log(d["atp_Explorer"].iloc[12:21].values / d["atp_Explorer"].iloc[0:9].values),
    })
    yoy["drefresh"] = [1, 1, 1, 1, 1, 1, 1, 0, 0]
    yoy["progA"] = [0, 0, 0, 1, 1, 1, 0, 0, 0]
    yoy["progB"] = [0, 0, 0, 0, 0, 0, 1, 1, 1]

    m4 = _ols(yoy["dlv"], yoy[["dlp"]])
    out["yoy"] = dict(model=m4, e=m4.params["dlp"], p=m4.pvalues["dlp"],
                      se=m4.bse["dlp"], r2=m4.rsquared, n=len(yoy),
                      label="4. Year-over-year differences",
                      spec="dln(Volume) = a + e dln(ATP)   [seasonality removed]")

    m5 = _ols(yoy["dlv"], yoy[["dlp", "drefresh", "progA", "progB"]])
    out["yoy_full"] = dict(model=m5, e=m5.params["dlp"], p=m5.pvalues["dlp"],
                           se=m5.bse["dlp"], r2=m5.rsquared, n=len(yoy),
                           label="5. YoY with refresh and program controls",
                           spec="dln(V) = a + e dln(P) + refresh + ProgA + ProgB")

    clean = d.iloc[7:15]  # Aug 2024 to Mar 2025: post refresh, pre programs
    m6 = _ols(clean["ln_v"], clean[["ln_p"]])
    out["clean"] = dict(model=m6, e=m6.params["ln_p"], p=m6.pvalues["ln_p"],
                        se=m6.bse["ln_p"], r2=m6.rsquared, n=len(clean),
                        label="6. Clean window (Aug 24 to Mar 25)",
                        spec="ln(Volume) = a + e ln(ATP) on post-refresh, pre-program months")

    return out, yoy


def arc_elasticity(p0, v0, p1, v1):
    pct_v = (v1 - v0) / ((v1 + v0) / 2)
    pct_p = (p1 - p0) / ((p1 + p0) / 2)
    return pct_v / pct_p, pct_p, pct_v


@st.cache_data
def program_metrics(quarterly: pd.DataFrame, baseline_e: float = -4.0):
    q = quarterly.set_index("quarter")
    ap0, av0 = q.loc["2025Q1", "atp_Explorer"], q.loc["2025Q1", "vol_Explorer"]
    ap1, av1 = q.loc["2025Q2", "atp_Explorer"], q.loc["2025Q2", "vol_Explorer"]
    bp1, bv1 = q.loc["2025Q3", "atp_Explorer"], q.loc["2025Q3", "vol_Explorer"]

    eA, dpA, dvA = arc_elasticity(ap0, av0, ap1, av1)
    eB, dpB, dvB = arc_elasticity(ap1, av1, bp1, bv1)

    price_lift_A = baseline_e * (ap1 / ap0 - 1) * 100
    campaign_lift_A = (av1 / av0 - 1) * 100 - price_lift_A
    price_lift_B = baseline_e * (bp1 / ap1 - 1) * 100
    residual_B = (bv1 / av1 - 1) * 100 - price_lift_B

    return dict(
        A=dict(p0=ap0, v0=av0, p1=ap1, v1=av1, arc=eA,
               dp=dpA * 100, dv=dvA * 100, dp_simple=(ap1 / ap0 - 1) * 100,
               dv_simple=(av1 / av0 - 1) * 100,
               price_lift=price_lift_A, campaign_lift=campaign_lift_A,
               yoy_vol=(av1 / q.loc["2024Q2", "vol_Explorer"] - 1) * 100,
               yoy_atp=(ap1 / q.loc["2024Q2", "atp_Explorer"] - 1) * 100),
        B=dict(p0=ap1, v0=av1, p1=bp1, v1=bv1, arc=eB,
               dp=dpB * 100, dv=dvB * 100, dp_simple=(bp1 / ap1 - 1) * 100,
               dv_simple=(bv1 / av1 - 1) * 100,
               price_lift=price_lift_B, residual=residual_B,
               yoy_vol=(bv1 / q.loc["2024Q3", "vol_Explorer"] - 1) * 100,
               yoy_atp=(bp1 / q.loc["2024Q3", "atp_Explorer"] - 1) * 100),
        baseline_e=baseline_e,
    )
