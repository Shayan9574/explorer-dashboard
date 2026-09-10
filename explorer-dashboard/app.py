"""Ford Explorer Yield Management Dashboard."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm
import streamlit as st

from data_loader import load_data, VEHICLES, COMPETITORS
from analysis import run_models, program_metrics, arc_elasticity
from ai import ai_section

st.set_page_config(page_title="Explorer Yield Management", page_icon="🚙",
                   layout="wide", initial_sidebar_state="expanded")

FORD = "#003478"
AMBER = "#D97706"
GREEN = "#15803D"
GRAY = "#6b7280"
PALETTE = {"Explorer": FORD, "Traverse": "#b45309", "Highlander": "#9d174d",
           "Pilot": "#0e7490", "Palisade": "#6d28d9", "Telluride": "#4d7c0f"}

st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
h1, h2, h3 {color: #003478;}
div[data-testid="stMetric"] {background: #f1f5fb; border: 1px solid #dbe4f3;
  border-radius: 10px; padding: 12px 14px;}
div[data-testid="stMetricLabel"] p {font-size: 0.82rem; color: #40506b;}
.badge {display:inline-block;padding:2px 10px;border-radius:12px;font-size:0.78rem;
  font-weight:600;margin-right:6px;}
.badge-a {background:#fef3c7;color:#92400e;} .badge-b {background:#dcfce7;color:#14532d;}
.rec {background:#eef4ff;border-left:5px solid #003478;border-radius:6px;
  padding:14px 18px;margin:8px 0;}
.insight {background:#f8fafc;border-left:3px solid #94a3b8;border-radius:4px;
  padding:7px 12px;margin:2px 0 14px 0;font-size:0.88rem;color:#334155;}
</style>""", unsafe_allow_html=True)


def insight(text: str):
    st.markdown(f'<div class="insight"><b>Insight.</b> {text}</div>', unsafe_allow_html=True)


def style_fig(fig, height=380, title=None):
    fig.update_layout(template="plotly_white", height=height,
                      margin=dict(l=40, r=30, t=55 if title else 25, b=40),
                      title=dict(text=title, font=dict(size=15, color=FORD)) if title else None,
                      legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
                      font=dict(family="Segoe UI, sans-serif"))
    return fig


def shade_programs(fig):
    fig.add_vrect(x0="2025-04-01", x1="2025-06-30", fillcolor=AMBER, opacity=0.12,
                  line_width=0, annotation_text="Program A", annotation_position="top left",
                  annotation_font=dict(size=11, color=AMBER))
    fig.add_vrect(x0="2025-07-01", x1="2025-09-30", fillcolor=GREEN, opacity=0.12,
                  line_width=0, annotation_text="Program B", annotation_position="top left",
                  annotation_font=dict(size=11, color=GREEN))
    fig.add_vline(x="2024-08-01", line_dash="dot", line_color=GRAY,
                  annotation_text="Mid-cycle refresh", annotation_font=dict(size=10, color=GRAY))
    return fig


# ---------------------------------------------------------------- data
monthly, quarterly, source = load_data()
models, yoy = run_models(monthly)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🚙 Yield Management")
    st.caption("Ford Explorer · Jan 2024 to Sep 2025")
    st.markdown("**Data source**")
    st.success(f"{source}", icon="✅")
    st.caption("Quarterly aggregates rebuilt from monthly data and verified against the "
               "workbook's own quarterly tab (exact match).")
    st.divider()
    st.markdown("**What-if: baseline elasticity**")
    e_base = st.slider("Assumed baseline price elasticity", -5.0, -3.0, -4.0, 0.1,
                       help="Program decompositions and expected-lift benchmarks across all "
                            "tabs recompute live from this value. The evidence supports "
                            "approximately -4.")
    st.divider()
    st.markdown("**Live AI analysis**")
    st.text_input("Claude API key (optional)", type="password", key="claude_api_key",
                  help="Enables the 'Generate live AI analysis' buttons at the end of each "
                       "tab. The key is held only in this browser session and is never stored.")
    st.divider()
    st.markdown("**Method in one line**")
    st.caption("Six elasticity estimators cross-checked; the Aug 2024 mid-cycle refresh is "
               "treated as a demand shifter throughout; programs evaluated as arc responses "
               "and decomposed at the baseline elasticity.")

prog = program_metrics(quarterly, e_base)
A, B = prog["A"], prog["B"]
qw = quarterly.set_index("quarter")

st.title("Ford Explorer Yield Management Dashboard")
st.caption("Price elasticity and 2026 sales program assessment · Monthly ATP and volume, "
           "January 2024 to September 2025")

tabs = st.tabs(["📊 Dataset", "📐 Price Elasticity", "🎯 Program A", "💳 Program B",
                "⚖️ Program Comparison", "🧭 Segmentation", "💡 Insights", "ℹ️ About"])


# ---------------------------------------------------------------- chart builders
def fig_atp_volume():
    fig = go.Figure()
    fig.add_bar(x=monthly["date"], y=monthly["vol_Explorer"], name="Volume (units)",
                marker_color="#b9cbe6", yaxis="y")
    fig.add_scatter(x=monthly["date"], y=monthly["atp_Explorer"], name="ATP ($)",
                    mode="lines+markers", line=dict(color=FORD, width=2.4), yaxis="y2")
    fig.update_layout(yaxis=dict(title="Monthly volume (units)"),
                      yaxis2=dict(title="ATP ($)", overlaying="y", side="right"))
    return shade_programs(style_fig(fig, 420, "Explorer ATP and volume with event windows"))


def fig_premium():
    prem = (monthly["atp_Explorer"] / monthly["atp_Segment"] - 1) * 100
    fig = go.Figure()
    fig.add_scatter(x=monthly["date"], y=prem, mode="lines+markers",
                    line=dict(color=FORD, width=2.4),
                    fill="tozeroy", fillcolor="rgba(0,52,120,0.08)")
    fig.add_hline(y=0, line_color="#64748b", line_width=1)
    fig.update_layout(yaxis_title="Explorer ATP premium vs segment (%)", showlegend=False)
    return shade_programs(style_fig(fig, 380, "Price positioning: Explorer ATP relative to the segment"))


def fig_yoy_overlay():
    m24 = monthly.iloc[0:9]; m25 = monthly.iloc[12:21]
    months = [d.strftime("%b") for d in m24["date"]]
    fig = go.Figure()
    fig.add_bar(x=months, y=m24["vol_Explorer"], name="2024", marker_color="#94a3b8")
    fig.add_bar(x=months, y=m25["vol_Explorer"], name="2025", marker_color=FORD)
    fig.update_layout(barmode="group", yaxis_title="Units")
    return style_fig(fig, 380, "Same-month volume, 2024 vs 2025 (seasonality held constant)")


def fig_revenue():
    rev = monthly["atp_Explorer"] * monthly["vol_Explorer"] / 1e6
    fig = go.Figure(go.Scatter(x=monthly["date"], y=rev, mode="lines+markers",
                               line=dict(color=GREEN, width=2.4),
                               fill="tozeroy", fillcolor="rgba(21,128,61,0.08)"))
    fig.update_layout(yaxis_title="Monthly revenue proxy ($M)", showlegend=False)
    return shade_programs(style_fig(fig, 380, "Revenue proxy (ATP × volume)"))


def fig_quarterly():
    fig = go.Figure()
    fig.add_bar(x=quarterly["quarter"], y=quarterly["vol_Explorer"],
                marker_color=[AMBER if q == "2025Q2" else GREEN if q == "2025Q3"
                              else FORD for q in quarterly["quarter"]],
                text=quarterly["vol_Explorer"].astype(int), textposition="outside")
    fig.update_layout(yaxis_title="Quarterly volume (units)")
    return style_fig(fig, 360, "Quarterly Explorer volume (amber = Program A, green = Program B)")


def fig_scatter():
    period = np.where(monthly["progA"] == 1, "Program A (Q2 25)",
             np.where(monthly["progB"] == 1, "Program B (Q3 25)",
             np.where(monthly["refresh"] == 1, "Post-refresh baseline", "Pre-refresh baseline")))
    fig = px.scatter(x=monthly["atp_Explorer"], y=monthly["vol_Explorer"], color=period,
                     color_discrete_map={"Program A (Q2 25)": AMBER, "Program B (Q3 25)": GREEN,
                                         "Post-refresh baseline": FORD,
                                         "Pre-refresh baseline": "#94a3b8"},
                     labels={"x": "ATP ($)", "y": "Monthly volume (units)"})
    fig.update_traces(marker=dict(size=11, line=dict(width=1, color="white")))
    return style_fig(fig, 400, "Price vs volume by regime: the refresh shifted the demand curve")


def fig_simpson():
    """Pooled (naive) fit vs within-regime fit: the omitted variable bias, visualized."""
    d = monthly.copy()
    base = d[(d["progA"] == 0) & (d["progB"] == 0)]
    pre = base[base["refresh"] == 0]; post = base[base["refresh"] == 1]
    fig = go.Figure()
    fig.add_scatter(x=pre["atp_Explorer"], y=pre["vol_Explorer"], mode="markers",
                    name="Pre-refresh months", marker=dict(size=11, color="#94a3b8",
                    line=dict(width=1, color="white")))
    fig.add_scatter(x=post["atp_Explorer"], y=post["vol_Explorer"], mode="markers",
                    name="Post-refresh months", marker=dict(size=11, color=FORD,
                    line=dict(width=1, color="white")))
    prm = d[d["progA"] == 1]; prb = d[d["progB"] == 1]
    fig.add_scatter(x=prm["atp_Explorer"], y=prm["vol_Explorer"], mode="markers",
                    name="Program A months", marker=dict(size=11, color=AMBER, symbol="diamond"))
    fig.add_scatter(x=prb["atp_Explorer"], y=prb["vol_Explorer"], mode="markers",
                    name="Program B months", marker=dict(size=11, color=GREEN, symbol="diamond"))
    grid = np.linspace(d["atp_Explorer"].min(), d["atp_Explorer"].max(), 60)
    naive = models["naive"]["model"].params
    fig.add_scatter(x=grid, y=np.exp(naive["const"] + naive["ln_p"] * np.log(grid)),
                    mode="lines", name=f"Pooled naive fit (e = {models['naive']['e']:.1f})",
                    line=dict(color="#dc2626", dash="dash", width=2))
    clean = models["clean"]["model"].params
    g2 = np.linspace(post["atp_Explorer"].min(), post["atp_Explorer"].max(), 60)
    fig.add_scatter(x=g2, y=np.exp(clean["const"] + clean["ln_p"] * np.log(g2)),
                    mode="lines", name=f"Within post-refresh fit (e = {models['clean']['e']:.1f})",
                    line=dict(color=FORD, width=2.5))
    fig.update_layout(xaxis_title="ATP ($)", yaxis_title="Monthly volume (units)")
    return style_fig(fig, 460, "Why the naive estimate is wrong: pooling two demand regimes flattens the slope")


def fig_forest():
    labels, est, err, col = [], [], [], []
    for k in ["naive", "refresh", "relative", "yoy", "yoy_full", "clean"]:
        m = models[k]
        labels.append(m["label"]); est.append(m["e"]); err.append(1.96 * m["se"])
        col.append("#94a3b8" if k == "naive" else FORD)
    labels += ["Arc: Program A window", "Arc: Program B window"]
    est += [A["arc"], B["arc"]]; err += [0, 0]; col += [AMBER, GREEN]
    fig = go.Figure(go.Bar(y=labels[::-1], x=est[::-1], orientation="h",
                           error_x=dict(type="data", array=err[::-1], color="#334155"),
                           marker_color=col[::-1]))
    fig.add_vline(x=e_base, line_dash="dash", line_color="#dc2626",
                  annotation_text=f"Baseline: {e_base:.1f}", annotation_font=dict(color="#dc2626"))
    fig.update_layout(xaxis_title="Estimated price elasticity (95% CI where applicable)")
    return style_fig(fig, 430, "Elasticity estimates across methods")


def fig_fit():
    m = models["refresh"]["model"]
    fitted = np.exp(m.fittedvalues)
    fig = go.Figure()
    fig.add_scatter(x=monthly["date"], y=monthly["vol_Explorer"], name="Actual volume",
                    mode="lines+markers", line=dict(color="#94a3b8", width=1.8))
    fig.add_scatter(x=monthly["date"], y=fitted, name="Model fit (refresh-controlled)",
                    mode="lines", line=dict(color=FORD, width=2.4))
    fig.update_layout(yaxis_title="Units")
    return shade_programs(style_fig(fig, 400, "Preferred model: fitted vs actual volume"))


def fig_share_area():
    share = pd.DataFrame({"date": monthly["date"]})
    tot = sum(monthly[f"vol_{v}"] for v in VEHICLES)
    for v in VEHICLES:
        share[v] = monthly[f"vol_{v}"] / tot * 100
    fig = go.Figure()
    for v in VEHICLES:
        fig.add_scatter(x=share["date"], y=share[v], stackgroup="one", name=v,
                        line=dict(width=0.6, color=PALETTE[v]))
    fig.update_layout(yaxis_title="Share of six-vehicle set (%)")
    return shade_programs(style_fig(fig, 420, "Segment share of sales volume"))


def fig_efficiency():
    effA = A["dv_simple"] / abs(A["dp_simple"])
    effB = B["dv_simple"] / abs(B["dp_simple"])
    fig = go.Figure(go.Bar(
        x=["Program A", "Program B", f"Plain price cut (e = {e_base:.1f})"],
        y=[effA, effB, abs(e_base)], marker_color=[AMBER, GREEN, FORD],
        text=[f"{effA:.1f}", f"{effB:.1f}", f"{abs(e_base):.1f}"], textposition="outside"))
    fig.update_layout(yaxis_title="Volume % gained per ATP % given up")
    return style_fig(fig, 380, "Promotional efficiency (higher is better)")


# ================================================================ TAB 1 DATASET
with tabs[0]:
    st.subheader("Dataset overview")
    c = st.columns(5)
    c[0].metric("Months of data", f"{len(monthly)}")
    c[1].metric("Explorer units sold", f"{int(monthly['vol_Explorer'].sum()):,}")
    c[2].metric("Avg Explorer ATP", f"${monthly['atp_Explorer'].mean():,.0f}")
    c[3].metric("ATP jump at refresh", "+8.9%", help="Jul 2024 to Aug 2024")
    c[4].metric("Vehicles tracked", "6 + segment ATP")

    st.plotly_chart(fig_atp_volume(), width="stretch")
    insight("The series contains three structural events, not random variation: the August 2024 "
            "refresh (ATP up 8.9 percent month over month with demand rising anyway), the Q2 2025 "
            "Program A window (volume surging while ATP barely moves), and the Q3 2025 Program B "
            "window (volume rising as ATP falls). Every estimate downstream must respect these breaks.")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_premium(), width="stretch")
        insight("Explorer sold at up to a 2 percent discount to the segment before the refresh, "
                "commanded up to an 8 percent premium after it, and was pushed back to a roughly "
                "4 percent discount by the two 2025 programs. The refresh bought real pricing "
                "power; the programs spent part of it.")
    with c2:
        st.plotly_chart(fig_yoy_overlay(), width="stretch")
        insight("Holding the month constant removes seasonality: every month of 2025 outsold its "
                "2024 counterpart, with the largest gaps in the Program A and B windows "
                "(Aug: +56 percent, Sep: +64 percent). This same-month logic is why the "
                "year-over-year elasticity model is credible.")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_quarterly(), width="stretch")
        insight("Quarterly volume steps up from a 26,920-unit trough in Q3 2024 to 43,203 in "
                "Q3 2025. The two program quarters are the two best quarters in the data, "
                "which is exactly why their effects must be separated from baseline elasticity.")
    with c2:
        st.plotly_chart(fig_scatter(), width="stretch")
        insight("The pre-refresh (gray) and post-refresh (blue) months form two distinct clouds: "
                "the market paid more and bought more after the refresh. The amber Program A "
                "points sit far above both clouds at unremarkable prices, the visual signature "
                "of a demand shift rather than a price response.")

    st.plotly_chart(fig_revenue(), width="stretch")
    insight(f"Revenue is the yield lens: Program A lifted quarterly revenue from "
            f"${A['p0']*A['v0']/1e9:.2f}B (Q1 2025) to ${A['p1']*A['v1']/1e9:.2f}B (Q2), "
            f"a {(A['p1']*A['v1']/(A['p0']*A['v0'])-1)*100:.0f} percent gain, because volume grew "
            f"while price held. Program B added a further "
            f"{(B['p1']*B['v1']/(B['p0']*B['v0'])-1)*100:.0f} percent, but by trading price for volume.")

    st.markdown("**Program windows vs immediately preceding baseline**")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_bar(x=["Q1 2025 baseline", "Q2 2025 Program A"],
                    y=[A["v0"], A["v1"]], marker_color=[FORD, AMBER],
                    text=[f"{A['v0']:,.0f}", f"{A['v1']:,.0f}"], textposition="outside")
        fig.update_layout(yaxis_title="Units")
        st.plotly_chart(style_fig(fig, 320, f"Program A: volume {A['dv_simple']:+.1f}% on ATP {A['dp_simple']:+.1f}%"),
                        width="stretch")
    with c2:
        fig = go.Figure()
        fig.add_bar(x=["Q2 2025 baseline", "Q3 2025 Program B"],
                    y=[B["v0"], B["v1"]], marker_color=[FORD, GREEN],
                    text=[f"{B['v0']:,.0f}", f"{B['v1']:,.0f}"], textposition="outside")
        fig.update_layout(yaxis_title="Units")
        st.plotly_chart(style_fig(fig, 320, f"Program B: volume {B['dv_simple']:+.1f}% on ATP {B['dp_simple']:+.1f}%"),
                        width="stretch")

    with st.expander("View raw monthly data"):
        show = monthly[["date"] + [f"atp_{v}" for v in VEHICLES] + ["atp_Segment"]
                       + [f"vol_{v}" for v in VEHICLES]].copy()
        st.dataframe(show, width="stretch", height=420)
    with st.expander("View quarterly aggregates (volume weighted ATP)"):
        st.dataframe(quarterly, width="stretch")

    ai_section(
        "dataset",
        f"""
- **Two regimes, not one.** The August 2024 mid-cycle refresh lifted ATP by 8.9 percent month over month and shifted the demand curve outward. Any elasticity estimate that ignores this break mixes two markets and is biased toward zero.
- **Program A is visually anomalous.** In Q2 2025 volume rose {A['dv_simple']:.1f} percent while ATP barely moved ({A['dp_simple']:.1f} percent). Points in that window sit far above the demand curve, indicating a demand shift rather than movement along it.
- **Program B looks like a price move.** Q3 2025 shows ATP down {B['dp_simple']:.1f} percent and volume up {B['dv_simple']:.1f} percent, a pattern consistent with ordinary price response.
- **Pricing power is the quiet story**: the refresh converted a segment discount into a premium; the 2025 programs traded part of it back for volume. 2026 strategy should decide deliberately how much premium to spend.
""",
        f"Monthly Explorer ATP and volume Jan 2024 to Sep 2025. Refresh Aug 2024 (+8.9% ATP). "
        f"Program A Q2 2025: ATP {A['dp_simple']:+.1f}%, volume {A['dv_simple']:+.1f}%. "
        f"Program B Q3 2025: ATP {B['dp_simple']:+.1f}%, volume {B['dv_simple']:+.1f}%. "
        f"Explorer moved from segment discount to premium after refresh, back to discount under programs.",
        "Analyze the dataset patterns and their implications for elasticity estimation.")


# ================================================================ TAB 2 ELASTICITY
with tabs[1]:
    st.subheader("Estimated Explorer price elasticity")
    st.markdown("Six estimators are computed live from the data. They are not interchangeable: "
                "each fixes a specific weakness, and the selection below is argued explicitly.")

    st.plotly_chart(fig_simpson(), width="stretch")
    insight(f"This is the central identification problem. Pooling all months (red dashed line) "
            f"yields {models['naive']['e']:.1f} because the refresh moved price and demand up "
            f"together. Within the stable post-refresh regime (blue line) the slope is "
            f"{models['clean']['e']:.1f}. The naive number is not a mild underestimate; it is a "
            f"different, wrong answer produced by omitted variable bias.")

    rows = []
    notes = {
        "naive": "Rejected: refresh confound biases it toward zero (omitted variable bias).",
        "refresh": "PRIMARY: controls the demand shift, uses all 21 months, significant at 5 percent.",
        "relative": "Supporting: absorbs market-wide price moves via the segment benchmark.",
        "yoy": "ROBUSTNESS: removes seasonality by construction; independent of monthly dummies.",
        "yoy_full": "Supporting: adds refresh and program controls; small n limits precision.",
        "clean": "Supporting: cleanest window, but only eight observations.",
    }
    for k in ["naive", "refresh", "relative", "yoy", "yoy_full", "clean"]:
        m = models[k]
        rows.append({"Method": m["label"], "Elasticity": round(m["e"], 2),
                     "p-value": round(m["p"], 3), "R²": round(m["r2"], 2),
                     "n": m["n"], "Verdict": notes[k]})
    rows.append({"Method": "7. Arc elasticity, Program B window", "Elasticity": round(B["arc"], 2),
                 "p-value": None, "R²": None, "n": 2,
                 "Verdict": "Model-free sanity check on a price-driven quarter."})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.markdown("**Method selection scorecard**")
    score = pd.DataFrame({
        "Criterion": ["Controls the refresh confound", "Handles seasonality",
                      "Uses the full sample", "Statistically significant (5%)",
                      "Assumption-light"],
        "Naive": ["✗", "✗", "✓", "✗", "✓"],
        "Refresh-controlled": ["✓", "partial", "✓", "✓", "✓"],
        "Relative price": ["✓", "partial", "✓", "✗", "✓"],
        "YoY": ["partial", "✓", "✗ (9 diffs)", "✓", "✓"],
        "YoY full": ["✓", "✓", "✗ (9 diffs)", "✗ (10%)", "✗"],
        "Clean window": ["✓ (by design)", "✗", "✗ (8 obs)", "✗", "✓"],
    })
    st.dataframe(score, width="stretch", hide_index=True)
    insight(f"No single method dominates every criterion, which is why the recommendation rests on "
            f"convergence. The refresh-controlled model is selected as primary: it is the only "
            f"estimator that both controls the confound and retains the full sample with "
            f"statistical significance ({models['refresh']['e']:.1f}, p = "
            f"{models['refresh']['p']:.2f}). The year-over-year model is the designated robustness "
            f"check ({models['yoy']['e']:.1f}, p = {models['yoy']['p']:.2f}) because it attacks "
            f"seasonality, the one weakness the primary model only partially covers. Their "
            f"agreement is the result.")

    st.plotly_chart(fig_forest(), width="stretch")
    insight("Every corrected estimator lands between about -3 and -5 and their confidence "
            "intervals overlap; only the knowingly biased naive model falls outside. The arc "
            "value for Program A is plotted deliberately: it is not an elasticity estimate but "
            "evidence of a demand shift, and treating it as an elasticity would be an error.")

    st.plotly_chart(fig_fit(), width="stretch")
    insight("A two-variable model (price and refresh) tracks the level shift and the broad "
            "volume pattern. What it visibly cannot explain are the program windows, where "
            "actuals run above fit, which is precisely the campaign effect quantified in the "
            "program tabs.")

    st.markdown(f"""
<div class="rec"><b>Recommendation: plan 2026 around an elasticity of approximately -4
(defensible range -3 to -5).</b><br>
Primary estimate {models['refresh']['e']:.1f} (refresh-controlled, p = {models['refresh']['p']:.2f});
robustness {models['yoy']['e']:.1f} (year over year, p = {models['yoy']['p']:.2f}); supporting
estimates {models['relative']['e']:.1f}, {models['yoy_full']['e']:.1f} and
{models['clean']['e']:.1f}. A value near -4 also matches published estimates for mainstream
midsize SUVs, a heavily cross-shopped segment. Practical reading: each 1 percent of transaction
price buys roughly 4 percent of volume, absent other stimulus. Use the sidebar slider to test
how conclusions move across the -3 to -5 range; the program verdicts do not change.</div>""",
                unsafe_allow_html=True)

    st.markdown("**Detailed calculations** (click to expand)")
    for k in ["naive", "refresh", "relative", "yoy", "yoy_full", "clean"]:
        m = models[k]
        with st.expander(f"{m['label']}  |  specification: {m['spec']}"):
            st.text(m["model"].summary().as_text())
    with st.expander("7. Arc elasticity worked example (Program B window)"):
        st.latex(r"\varepsilon_{arc}=\frac{(V_1-V_0)/\bar V}{(P_1-P_0)/\bar P}")
        st.markdown(f"""
| Quantity | Value |
|---|---|
| P0 (Q2 2025 ATP) | ${B['p0']:,.0f} |
| P1 (Q3 2025 ATP) | ${B['p1']:,.0f} |
| V0 (Q2 2025 volume) | {B['v0']:,.0f} |
| V1 (Q3 2025 volume) | {B['v1']:,.0f} |
| Midpoint % change in price | {B['dp']:.2f}% |
| Midpoint % change in volume | {B['dv']:.2f}% |
| **Arc elasticity** | **{B['arc']:.2f}** |
""")

    e_grid = [-5.0, e_base, -3.0]
    p_grid = np.linspace(-6, 0, 50)
    fig = go.Figure()
    for e in e_grid:
        fig.add_scatter(x=p_grid, y=e * p_grid, mode="lines",
                        name=f"elasticity {e:.1f}",
                        line=dict(width=2.5 if e == e_base else 1.2,
                                  color=FORD if e == e_base else "#9db4d6"))
    fig.add_scatter(x=[A["dp_simple"]], y=[A["dv_simple"]], mode="markers+text",
                    text=["Program A"], textposition="top center", name="Program A",
                    marker=dict(size=13, color=AMBER))
    fig.add_scatter(x=[B["dp_simple"]], y=[B["dv_simple"]], mode="markers+text",
                    text=["Program B"], textposition="top center", name="Program B",
                    marker=dict(size=13, color=GREEN))
    fig.update_layout(xaxis_title="% change in ATP", yaxis_title="Expected % change in volume")
    st.plotly_chart(style_fig(fig, 400, "What the baseline elasticity predicts vs what the programs delivered"),
                    width="stretch")
    insight("Program B sits essentially on the predicted lines: a conventional price response. "
            "Program A sits roughly 20 points above every plausible line: demand creation that "
            "no elasticity value can explain.")

    ai_section(
        "elast",
        f"""
- **Convergence, not a single model, drives the answer.** Four independent corrections for the refresh confound land between {models['clean']['e']:.1f} and {models['yoy_full']['e']:.1f}; only the knowingly biased naive model falls outside.
- **The best method for this dataset is the refresh-controlled OLS**, because it is the only estimator that simultaneously controls the confound, keeps all 21 observations, and achieves significance; the year-over-year model independently confirms it while eliminating seasonality.
- **The naive estimate is a textbook omitted variable bias**: the refresh raised price about 9 percent and demand simultaneously, dragging the naive slope to {models['naive']['e']:.1f}.
- **Uncertainty is real**: the preferred estimate carries a standard error near {models['refresh']['se']:.1f}. Carry the range -3 to -5 into planning, and note that the program verdicts are invariant across it.
""",
        f"Elasticity estimates: naive {models['naive']['e']:.2f}, refresh {models['refresh']['e']:.2f}, "
        f"relative {models['relative']['e']:.2f}, YoY {models['yoy']['e']:.2f}, "
        f"YoY full {models['yoy_full']['e']:.2f}, clean {models['clean']['e']:.2f}, arc B {B['arc']:.2f}. "
        f"Selected: refresh-controlled primary, YoY robustness.",
        "Compare the elasticity estimation methods, justify which is best for this dataset, and defend the recommended planning value.")


# ================================================================ TAB 3 PROGRAM A
with tabs[2]:
    st.subheader('Program A: "From America For America" employee pricing (Q2 2025)')
    c = st.columns(4)
    c[0].metric("Volume vs Q1 2025", f"{A['dv_simple']:+.1f}%")
    c[1].metric("ATP vs Q1 2025", f"{A['dp_simple']:+.1f}%")
    c[2].metric("Arc elasticity", f"{A['arc']:.1f}", help=f"vs baseline of {e_base:.1f}")
    c[3].metric("Volume vs Q2 2024", f"{A['yoy_vol']:+.1f}%", help=f"with ATP {A['yoy_atp']:+.1f}% YoY")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Waterfall(
            x=["Q1 2025 volume", f"Price effect (e = {e_base:.1f})", "Campaign / brand effect", "Q2 2025 volume"],
            measure=["absolute", "relative", "relative", "total"],
            y=[A["v0"], A["v0"] * A["price_lift"] / 100,
               A["v0"] * A["campaign_lift"] / 100, None],
            connector=dict(line=dict(color="#cbd5e1")),
            increasing=dict(marker_color=AMBER), totals=dict(marker_color=FORD),
            decreasing=dict(marker_color="#94a3b8")))
        fig.update_layout(yaxis_title="Units")
        st.plotly_chart(style_fig(fig, 400, "Decomposition of the Q2 2025 lift"), width="stretch")
        insight(f"At a baseline elasticity of {e_base:.1f}, the {abs(A['dp_simple']):.1f} percent "
                f"ATP decline explains only {A['price_lift']:.1f} points of the "
                f"{A['dv_simple']:.1f} percent lift. The remaining {A['campaign_lift']:.1f} "
                f"points, about {A['campaign_lift']/A['dv_simple']*100:.0f} percent of the total, "
                f"is campaign-created demand. Move the sidebar slider: this conclusion barely moves.")
    with c2:
        m24 = monthly.iloc[3:6]; m25 = monthly.iloc[15:18]
        months = [d.strftime("%b") for d in m24["date"]]
        fig = go.Figure()
        fig.add_bar(x=months, y=m24["vol_Explorer"], name="2024 (no program)", marker_color="#94a3b8")
        fig.add_bar(x=months, y=m25["vol_Explorer"], name="2025 (Program A)", marker_color=AMBER)
        fig.update_layout(barmode="group", yaxis_title="Units")
        st.plotly_chart(style_fig(fig, 400, "Q2 same-month comparison: 2024 vs 2025"), width="stretch")
        insight(f"Against the same months of 2024, which removes seasonality entirely, Program A "
                f"delivered {A['yoy_vol']:+.1f} percent volume while ATP was {A['yoy_atp']:+.1f} "
                f"percent HIGHER year over year. Selling more at higher prices is the strongest "
                f"possible evidence of demand creation rather than discounting.")

    c1, c2 = st.columns(2)
    with c1:
        dfA = monthly[monthly["date"].between("2025-01-01", "2025-06-30")]
        fig = go.Figure()
        fig.add_bar(x=dfA["date"], y=dfA["vol_Explorer"],
                    marker_color=[AMBER if p else FORD for p in dfA["progA"]])
        fig.add_scatter(x=dfA["date"], y=dfA["atp_Explorer"], yaxis="y2", name="ATP",
                        line=dict(color="#334155", width=2))
        fig.update_layout(yaxis_title="Units",
                          yaxis2=dict(title="ATP ($)", overlaying="y", side="right"),
                          showlegend=False)
        st.plotly_chart(style_fig(fig, 380, "Monthly view: baseline (blue) into Program A (amber)"),
                        width="stretch")
        insight("The lift built through the quarter (April to June) rather than spiking and "
                "fading, and ATP stayed essentially flat throughout: the campaign pulled buyers "
                "in without deepening the discount over time.")
    with c2:
        revs = [A["p0"] * A["v0"] / 1e9, A["p1"] * A["v1"] / 1e9]
        fig = go.Figure(go.Bar(x=["Q1 2025", "Q2 2025 (Program A)"], y=revs,
                               marker_color=[FORD, AMBER],
                               text=[f"${r:.2f}B" for r in revs], textposition="outside"))
        fig.update_layout(yaxis_title="Revenue proxy ($B)")
        st.plotly_chart(style_fig(fig, 380, f"Revenue impact: {(revs[1]/revs[0]-1)*100:+.1f}%"),
                        width="stretch")
        insight(f"Because volume grew {A['dv_simple']:.0f} percent while price gave up only "
                f"{abs(A['dp_simple']):.1f} percent, quarterly revenue rose "
                f"{(revs[1]/revs[0]-1)*100:.1f} percent, about "
                f"${(revs[1]-revs[0])*1000:.0f}M. A plain price cut cannot do this: at e = "
                f"{e_base:.1f}, revenue rises only about {(abs(e_base)-1)*abs(A['dp_simple']):.0f} "
                f"percent per {abs(A['dp_simple']):.1f} percent of price.")

    with st.expander("Detailed calculation: arc elasticity and decomposition"):
        st.markdown(f"""
Arc elasticity = ({A['dv']:.2f}% volume) / ({A['dp']:.2f}% price) = **{A['arc']:.1f}**

Decomposition at baseline elasticity {e_base:.1f}: the {A['dp_simple']:.2f} percent ATP decline
explains about **{A['price_lift']:.1f} points** of lift; the remaining
**{A['campaign_lift']:.1f} points** of the {A['dv_simple']:.1f} percent total is attributable to
the campaign itself (messaging, earned media, perceived-deal framing). An arc value of
{A['arc']:.1f} is not a price elasticity at all: it is the signature of a demand curve shifting
outward. The year-over-year check (volume {A['yoy_vol']:+.1f} percent on ATP {A['yoy_atp']:+.1f}
percent) rules out seasonality as the explanation.
""")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<span class="badge badge-a">BENEFITS</span>', unsafe_allow_html=True)
        st.markdown("""
- Exceptional yield: about 26 percent more volume for about 1 percent of ATP.
- Genuine demand creation: volume grew even against higher year-over-year prices.
- Revenue accretive by roughly 24 percent quarter over quarter.
- Simple, transparent message with strong earned media and a patriotic halo.
- Lift persisted into Q3, suggesting limited immediate payback.
""")
    with c2:
        st.markdown('<span class="badge badge-a">DOWNSIDES</span>', unsafe_allow_html=True)
        st.markdown("""
- Novelty driven: a rerun is unlikely to replicate the same magnitude.
- Repetition trains buyers to wait for deals and erodes long-run pricing power.
- Per-unit margin compression can exceed what ATP shows if trim mix shifted rich.
- Some Q2 volume is likely pull-ahead demand from later quarters.
- "Employee pricing" anchors the public reference price downward.
""")

    ai_section(
        "progA",
        f"""
- Program A delivered a **{A['dv_simple']:.1f} percent volume lift at only {abs(A['dp_simple']):.1f} percent ATP cost**, and did so while year-over-year ATP was {A['yoy_atp']:+.1f} percent: demand creation, not discounting.
- At the current baseline setting of {e_base:.1f}, roughly **{A['campaign_lift']/A['dv_simple']*100:.0f} percent of the lift is campaign-driven**, the scarcest and most valuable kind of lift, and this share is robust across the whole -3 to -5 range.
- Revenue rose about {(A['p1']*A['v1']/(A['p0']*A['v0'])-1)*100:.0f} percent quarter over quarter, which a pure price action of this size could not achieve.
- The strategic risk is not Q2 2025; it is 2027: repeated employee-pricing events convert a brand moment into a discount expectation.
""",
        f"Program A Q2 2025: volume {A['dv_simple']:+.1f}%, ATP {A['dp_simple']:+.1f}%, arc {A['arc']:.1f}, "
        f"YoY vol {A['yoy_vol']:+.1f}% at YoY ATP {A['yoy_atp']:+.1f}%, "
        f"price-explained {A['price_lift']:.1f} pts, campaign {A['campaign_lift']:.1f} pts, baseline {e_base:.1f}.",
        "Assess Program A's performance, decomposition, and strategic risks.")


# ================================================================ TAB 4 PROGRAM B
with tabs[3]:
    st.subheader("Program B: low APR and lease offers (Q3 2025)")
    c = st.columns(4)
    c[0].metric("Volume vs Q2 2025", f"{B['dv_simple']:+.1f}%")
    c[1].metric("ATP vs Q2 2025", f"{B['dp_simple']:+.1f}%")
    c[2].metric("Arc elasticity", f"{B['arc']:.1f}", help=f"vs baseline of {e_base:.1f}")
    c[3].metric("Volume vs Q3 2024", f"{B['yoy_vol']:+.1f}%",
                help=f"with ATP {B['yoy_atp']:+.1f}% YoY; Q3 2024 was a launch-transition low")

    c1, c2 = st.columns(2)
    with c1:
        exp_lift = e_base * B["dp_simple"]
        fig = go.Figure(go.Bar(
            x=[f"Expected from price alone (e = {e_base:.1f})", "Actual lift"],
            y=[exp_lift, B["dv_simple"]], marker_color=["#94a3b8", GREEN],
            text=[f"{exp_lift:+.1f}%", f"{B['dv_simple']:+.1f}%"], textposition="outside"))
        fig.update_layout(yaxis_title="% volume change vs Q2 2025")
        st.plotly_chart(style_fig(fig, 400, "Program B vs the baseline price response"),
                        width="stretch")
        insight(f"At e = {e_base:.1f}, the {abs(B['dp_simple']):.1f} percent ATP decline should "
                f"have produced about {exp_lift:.1f} percent volume; the observed "
                f"{B['dv_simple']:.1f} percent falls short by roughly "
                f"{exp_lift - B['dv_simple']:.0f} points. Two fair caveats temper this: the Q2 "
                f"comparison base was inflated by Program A carryover, and the shortfall shrinks "
                f"at the -3 end of the slider.")
    with c2:
        m24 = monthly.iloc[6:9]; m25 = monthly.iloc[18:21]
        months = [d.strftime("%b") for d in m24["date"]]
        fig = go.Figure()
        fig.add_bar(x=months, y=m24["vol_Explorer"], name="2024 (launch transition)", marker_color="#94a3b8")
        fig.add_bar(x=months, y=m25["vol_Explorer"], name="2025 (Program B)", marker_color=GREEN)
        fig.update_layout(barmode="group", yaxis_title="Units")
        st.plotly_chart(style_fig(fig, 400, "Q3 same-month comparison: 2024 vs 2025"), width="stretch")
        insight(f"Volume is {B['yoy_vol']:+.1f} percent year over year, but the honest reading "
                f"discounts this: Q3 2024 was artificially low during the launch transition "
                f"(model changeover), so the same-month comparison overstates Program B. The "
                f"quarter-over-quarter view is the fairer test, and it shows an ordinary price response.")

    c1, c2 = st.columns(2)
    with c1:
        dfB = monthly[monthly["date"].between("2025-04-01", "2025-09-30")]
        fig = go.Figure()
        fig.add_bar(x=dfB["date"], y=dfB["vol_Explorer"],
                    marker_color=[GREEN if p else AMBER for p in dfB["progB"]])
        fig.add_scatter(x=dfB["date"], y=dfB["atp_Explorer"], yaxis="y2", name="ATP",
                        line=dict(color="#334155", width=2))
        fig.update_layout(yaxis_title="Units",
                          yaxis2=dict(title="ATP ($)", overlaying="y", side="right"),
                          showlegend=False)
        st.plotly_chart(style_fig(fig, 380, "Monthly view: Program A quarter (amber) into Program B (green)"),
                        width="stretch")
        insight("ATP steps down visibly through the Program B quarter while volume peaks in "
                "August and softens in September, a fading response despite deepening price "
                "support: the classic pattern of buying volume rather than creating demand.")
    with c2:
        revs = [B["p0"] * B["v0"] / 1e9, B["p1"] * B["v1"] / 1e9]
        fig = go.Figure(go.Bar(x=["Q2 2025", "Q3 2025 (Program B)"], y=revs,
                               marker_color=[AMBER, GREEN],
                               text=[f"${r:.2f}B" for r in revs], textposition="outside"))
        fig.update_layout(yaxis_title="Revenue proxy ($B)")
        st.plotly_chart(style_fig(fig, 380, f"Revenue impact: {(revs[1]/revs[0]-1)*100:+.1f}%"),
                        width="stretch")
        insight(f"Revenue grew only {(revs[1]/revs[0]-1)*100:.1f} percent because the "
                f"{B['dv_simple']:.1f} percent volume gain was largely offset by the "
                f"{abs(B['dp_simple']):.1f} percent price give-back, and this proxy still "
                f"excludes the APR subvention cost, which sits outside ATP. True incremental "
                f"profit is lower than this chart suggests.")

    with st.expander("Detailed calculation: arc elasticity and expectation gap"):
        st.markdown(f"""
Arc elasticity = ({B['dv']:.2f}% volume) / ({B['dp']:.2f}% price) = **{B['arc']:.1f}**

At the baseline elasticity of {e_base:.1f}, the {abs(B['dp_simple']):.2f} percent ATP decline
should have produced about **{e_base * B['dp_simple']:.1f} percent** volume; the observed
**{B['dv_simple']:.1f} percent** falls short by roughly
{e_base * B['dp_simple'] - B['dv_simple']:.1f} points. Two honest caveats: the Q2 comparison base
was inflated by Program A carryover, and APR subvention is a real cost that sits **outside**
ATP, so the true effective discount, and thus the shortfall in efficiency, is larger than the
ATP series shows.
""")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<span class="badge badge-b">BENEFITS</span>', unsafe_allow_html=True)
        st.markdown("""
- Proven, low-execution-risk lever with predictable response.
- Targets payment-sensitive buyers without cutting sticker prices.
- Protects residual values better than cash rebates.
- Leases build a return and certified pre-owned pipeline that feeds future loyalty.
- Precise targeting: can be dialed by region, trim, or credit tier.
""")
    with c2:
        st.markdown('<span class="badge badge-b">DOWNSIDES</span>', unsafe_allow_html=True)
        st.markdown("""
- Lift (arc about -2) at or below an ordinary price response: no multiplier.
- True cost understated: APR subvention sits outside ATP, yet ATP still fell 4.8 percent.
- Benefits accrue mainly to prime-credit customers.
- Higher lease penetration adds residual-value risk at Ford Credit.
- Industry-standard playbook: easily matched by competitors, no brand distinction.
""")

    ai_section(
        "progB",
        f"""
- Program B behaved like a conventional price action: **{B['dv_simple']:.1f} percent volume for {abs(B['dp_simple']):.1f} percent ATP**, an arc of {B['arc']:.1f} against the {e_base:.1f} baseline.
- The year-over-year gain of {B['yoy_vol']:+.1f} percent flatters it: Q3 2024 was a launch-transition trough, so the quarter-over-quarter view is the fair test.
- Revenue rose only {(B['p1']*B['v1']/(B['p0']*B['v0'])-1)*100:.1f} percent, before counting the APR subvention cost that ATP does not capture.
- Its real value is structural, not promotional: residual-value protection and the lease-return pipeline. That argues for tactical, targeted use rather than a headline campaign.
""",
        f"Program B Q3 2025: volume {B['dv_simple']:+.1f}%, ATP {B['dp_simple']:+.1f}%, arc {B['arc']:.1f}, "
        f"YoY vol {B['yoy_vol']:+.1f}% (inflated base), baseline {e_base:.1f}.",
        "Assess Program B's performance relative to baseline elasticity and its strategic role.")


# ================================================================ TAB 5 COMPARISON
with tabs[4]:
    st.subheader("Program A vs Program B vs a plain price cut")
    comp = pd.DataFrame({
        "Metric": ["Quarter", "Mechanism", "ATP change", "Volume change", "Arc elasticity",
                   "Lift beyond price response", "Revenue impact (proxy)", "Hidden costs",
                   "Brand effect", "Repeatability", "Residual value impact", "Execution risk"],
        "Program A": ["Q2 2025", "Employee pricing + campaign", f"{A['dp_simple']:+.1f}%",
                      f"{A['dv_simple']:+.1f}%", f"{A['arc']:.1f}",
                      f"About {A['campaign_lift']:.0f} points (demand shift)",
                      f"{(A['p1']*A['v1']/(A['p0']*A['v0'])-1)*100:+.1f}%",
                      "Margin compression masked by mix", "Strong positive halo",
                      "Low: novelty dependent", "Negative if repeated (reference price anchor)",
                      "Moderate (dealer coordination)"],
        "Program B": ["Q3 2025", "Low APR + lease offers", f"{B['dp_simple']:+.1f}%",
                      f"{B['dv_simple']:+.1f}%", f"{B['arc']:.1f}",
                      "None observed",
                      f"{(B['p1']*B['v1']/(B['p0']*B['v0'])-1)*100:+.1f}%",
                      "APR subvention outside ATP", "Neutral, industry standard",
                      "High: standard playbook", "Protective vs cash discounts",
                      "Low"],
        f"Plain price cut (e = {e_base:.1f})": ["Reference", "Sticker/cash discount",
                      "-3.0% (illustration)", f"{e_base*-3.0:+.1f}% (predicted)", f"{e_base:.1f}",
                      "None by definition", f"about {(-3.0 + e_base*-3.0):+.1f}%",
                      "Full discount visible in ATP", "Negative (deal conditioning)",
                      "High but corrosive", "Negative (transaction price anchor)", "Low"],
    })
    st.dataframe(comp, width="stretch", hide_index=True)
    insight("Adding the plain price cut as the reference column makes the contrast exact: "
            "Program A beats the reference on every observable row, while Program B matches it "
            "on lift and beats it only on the structural rows (residuals, targeting, sticker "
            "protection). That is why A is a campaign and B is a tool.")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_bar(name="ATP given up (%)", x=["Program A", "Program B"],
                    y=[abs(A["dp_simple"]), abs(B["dp_simple"])], marker_color="#94a3b8")
        fig.add_bar(name="Volume gained (%)", x=["Program A", "Program B"],
                    y=[A["dv_simple"], B["dv_simple"]], marker_color=[AMBER, GREEN])
        fig.update_layout(barmode="group", yaxis_title="%")
        st.plotly_chart(style_fig(fig, 380, "Cost vs lift"), width="stretch")
        insight("Program A's bars are wildly asymmetric (large gain, tiny cost); Program B's are "
                "roughly proportional, which is exactly what an on-curve price response looks like.")
    with c2:
        st.plotly_chart(fig_efficiency(), width="stretch")
        insight(f"Volume points gained per ATP point spent: Program A "
                f"{A['dv_simple']/abs(A['dp_simple']):.0f}, Program B "
                f"{B['dv_simple']/abs(B['dp_simple']):.1f}, plain price cut {abs(e_base):.1f} by "
                f"definition. B buys volume slightly below the market exchange rate; A "
                f"transcends it.")

    cats = ["Volume lift", "Yield efficiency", "Margin protection",
            "Repeatability", "Brand building", "Execution simplicity"]
    scoreA = [5, 5, 4, 2, 5, 3]
    scoreB = [3, 2, 4, 5, 2, 5]
    scoreC = [3, 3, 2, 4, 1, 5]
    fig = go.Figure()
    for name, sc, col in [("Program A", scoreA, AMBER), ("Program B", scoreB, GREEN),
                          ("Plain price cut", scoreC, "#94a3b8")]:
        fig.add_trace(go.Scatterpolar(r=sc + sc[:1], theta=cats + cats[:1], fill="toself",
                                      name=name, line=dict(color=col)))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), height=460,
                      title=dict(text="Strategic profile (0 to 5, scored from the analysis above)",
                                 font=dict(size=15, color=FORD)),
                      legend=dict(orientation="h", y=-0.1))
    st.plotly_chart(fig, width="stretch")
    insight("The radar makes the complementarity visible: the two programs are near mirror "
            "images. A dominates the demand-creation half (lift, efficiency, brand); B dominates "
            "the operational half (repeatability, simplicity) while tying on margin protection. "
            "Scores are qualitative syntheses of the quantitative results, stated for "
            "transparency rather than measured.")

    st.markdown("**Which program in which 2026 scenario**")
    st.dataframe(pd.DataFrame({
        "2026 scenario": ["Share offense / conquest push", "Inventory overhang to clear",
                          "Margin protection priority", "Rising interest rates",
                          "Building future loyalty pipeline", "Competitor launches heavy incentives",
                          "Brand moment available (event, launch)"],
        "Better fit": ["Program A", "Program A", "Program B", "Program B (subvention more valuable)",
                       "Program B", "Program B (fast, targeted)", "Program A"],
        "Why": ["Demand creation beats price matching",
                "Maximum lift per point of ATP",
                "Sticker intact; discount lives in financing",
                "APR buy-down worth more to buyers when rates are high",
                "Lease returns feed CPO and repurchase",
                "Standard lever, deployable in weeks, dialable",
                "Campaign halo needs a story to attach to"],
    }), width="stretch", hide_index=True)

    ai_section(
        "comp",
        f"""
- **On observed 2025 evidence, A dominates on efficiency**: {A['dv_simple'] / abs(A['dp_simple']):.0f} points of volume per point of ATP versus {B['dv_simple'] / abs(B['dp_simple']):.1f} for B and {abs(e_base):.1f} for a plain price cut at the current baseline.
- **The programs are not substitutes but mirror images**: A wins the demand-creation dimensions, B wins the operational ones. A portfolio uses both.
- **Recommended 2026 posture**: one refreshed Program A event as the headline (new creative, protected novelty), Program B as the always-available tactical tool dialed by region and credit tier, and the plain price cut avoided entirely, since B strictly dominates it.
""",
        f"A: {A['dv_simple']:+.1f}% vol / {A['dp_simple']:+.1f}% ATP, arc {A['arc']:.1f}. "
        f"B: {B['dv_simple']:+.1f}% vol / {B['dp_simple']:+.1f}% ATP, arc {B['arc']:.1f}. Baseline {e_base:.1f}.",
        "Compare the two programs against the plain-price-cut reference and recommend a 2026 deployment strategy by scenario.")


# ================================================================ TAB 6 SEGMENTATION
with tabs[5]:
    st.subheader("Competitive segment analysis")

    st.plotly_chart(fig_share_area(), width="stretch")
    insight("Explorer's share band widens visibly after the refresh and again in each program "
            "window, while Highlander's band narrows almost continuously: the two dominant and "
            "opposite trajectories in the set.")

    c1, c2 = st.columns(2)
    q0, q1 = quarterly.iloc[0], quarterly.iloc[-1]
    with c1:
        fig = go.Figure(go.Pie(labels=VEHICLES, values=[q0[f"vol_{v}"] for v in VEHICLES],
                               hole=0.5, marker=dict(colors=[PALETTE[v] for v in VEHICLES]),
                               textinfo="label+percent"))
        fig.update_layout(height=380, title=dict(text="Share of segment volume, Q1 2024",
                                                 font=dict(size=15, color=FORD)), showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = go.Figure(go.Pie(labels=VEHICLES, values=[q1[f"vol_{v}"] for v in VEHICLES],
                               hole=0.5, marker=dict(colors=[PALETTE[v] for v in VEHICLES]),
                               textinfo="label+percent"))
        fig.update_layout(height=380, title=dict(text="Share of segment volume, Q3 2025",
                                                 font=dict(size=15, color=FORD)), showlegend=False)
        st.plotly_chart(fig, width="stretch")
    shr0 = q0["vol_Explorer"] / sum(q0[f"vol_{v}"] for v in VEHICLES) * 100
    shr1 = q1["vol_Explorer"] / sum(q1[f"vol_{v}"] for v in VEHICLES) * 100
    hshr0 = q0["vol_Highlander"] / sum(q0[f"vol_{v}"] for v in VEHICLES) * 100
    hshr1 = q1["vol_Highlander"] / sum(q1[f"vol_{v}"] for v in VEHICLES) * 100
    insight(f"Explorer's share rose from {shr0:.1f} to {shr1:.1f} percent of the six-vehicle set "
            f"while Highlander's fell from {hshr0:.1f} to {hshr1:.1f} percent. The redistribution "
            f"is nearly one-for-one, and it is still in progress.")

    tidy = []
    for _, r in quarterly.iterrows():
        for v in VEHICLES:
            tidy.append({"quarter": r["quarter"], "vehicle": v,
                         "ATP": r[f"atp_{v}"], "volume": r[f"vol_{v}"]})
    tidy = pd.DataFrame(tidy)
    fig = px.scatter(tidy, x="ATP", y="volume", color="vehicle", size="volume",
                     animation_frame="quarter", size_max=42,
                     color_discrete_map=PALETTE, text="vehicle",
                     range_x=[tidy["ATP"].min() * 0.96, tidy["ATP"].max() * 1.04],
                     range_y=[0, tidy["volume"].max() * 1.12],
                     labels={"ATP": "Quarterly ATP ($)", "volume": "Quarterly volume (units)"})
    fig.update_traces(textposition="top center", textfont=dict(size=10))
    fig.update_layout(height=520, template="plotly_white",
                      title=dict(text="Press play: price-volume positioning, quarter by quarter",
                                 font=dict(size=15, color=FORD)))
    st.plotly_chart(fig, width="stretch")
    insight("The animation shows three stories at once: Highlander drifting up-price and "
            "down-volume into irrelevance, Traverse repositioning sharply upmarket after its "
            "redesign while holding volume, and Explorer moving up-and-right after the refresh, "
            "then trading price for a decisive volume lead through the 2025 programs.")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for v in VEHICLES:
            fig.add_scatter(x=monthly["date"], y=monthly[f"atp_{v}"], name=v,
                            line=dict(color=PALETTE[v], width=2.6 if v == "Explorer" else 1.3))
        fig.update_layout(yaxis_title="ATP ($)")
        st.plotly_chart(style_fig(fig, 420, "ATP by nameplate"), width="stretch")
        insight("Three pricing strategies coexist: premium climbers (Highlander, then Traverse "
                "post-redesign), the stable mid-pack (Pilot, Palisade, Telluride), and Explorer, "
                "the only nameplate that moved up with product news and back down with programs. "
                "Explorer is using price as an active lever; most rivals are not.")
    with c2:
        chg = [(q1[f"vol_{v}"] / q0[f"vol_{v}"] - 1) * 100 for v in VEHICLES]
        fig = go.Figure(go.Bar(x=VEHICLES, y=chg,
                               marker_color=[PALETTE[v] for v in VEHICLES],
                               text=[f"{c:+.0f}%" for c in chg], textposition="outside"))
        fig.update_layout(yaxis_title="% change in quarterly volume")
        st.plotly_chart(style_fig(fig, 420, "Volume change, Q1 2024 to Q3 2025"), width="stretch")
        insight("Growth is not segment-wide lift: Explorer and Palisade are gaining materially, "
                "Highlander is collapsing, and the rest are roughly flat. This is share "
                "reallocation, which rewards offense in 2026.")

    hl = monthly[["date", "atp_Highlander", "vol_Highlander"]]
    fig = go.Figure()
    fig.add_bar(x=hl["date"], y=hl["vol_Highlander"], name="Highlander volume",
                marker_color="#e5a3bd")
    fig.add_scatter(x=hl["date"], y=hl["atp_Highlander"], name="Highlander ATP", yaxis="y2",
                    line=dict(color="#9d174d", width=2.4))
    fig.update_layout(yaxis=dict(title="Units"),
                      yaxis2=dict(title="ATP ($)", overlaying="y", side="right"))
    st.plotly_chart(style_fig(fig, 400, "Case study: the Highlander price-volume spiral"),
                    width="stretch")
    hl0 = int(qw.loc["2024Q1", "vol_Highlander"]); hl1 = int(qw.loc["2025Q3", "vol_Highlander"])
    insight(f"Highlander raised ATP from about $47,000 toward $52,000 while volume fell from "
            f"{hl0:,} to {hl1:,} units per quarter ({(hl1/hl0-1)*100:.0f} percent), an implied "
            f"response far more elastic than Explorer's. It is the segment's cautionary tale: "
            f"pricing up without product news, the mirror image of Explorer's refresh-backed "
            f"increase, and the source of the share now available for conquest.")

    ai_section(
        "seg",
        f"""
- **The Highlander collapse is the segment's dominant story**: {hl0:,} units in Q1 2024 to {hl1:,} in Q3 2025 ({(hl1 / hl0 - 1) * 100:.0f} percent) as its ATP climbed above 51,000 dollars without a product refresh to justify it. Explorer is the natural claimant of that volume.
- **Traverse proves the counterfactual**: it repositioned upmarket by roughly 18 percent in ATP after a redesign and held volume, showing the segment tolerates premium pricing when backed by fresh product, exactly the effect Explorer's own refresh produced.
- **Explorer exits Q3 2025 as the volume leader** at the segment's mid-price point, with Palisade the most aggressive challenger.
- Strategic read for 2026: contested share plus a proven demand-creation lever (Program A) argues for offense, with Program B reserved as the rapid response if Palisade or others escalate incentives.
""",
        f"Highlander {hl0} to {hl1} units; Explorer share {shr0:.1f}% to {shr1:.1f}%; Traverse ATP +18% post-redesign.",
        "Analyze competitive dynamics and their implications for Explorer 2026 strategy.")


# ================================================================ TAB 7 INSIGHTS
with tabs[6]:
    st.subheader("Executive insights for decision makers")
    c = st.columns(4)
    c[0].metric("Planning elasticity", "≈ -4", "range -3 to -5")
    c[1].metric("Program A lift", f"{A['dv_simple']:+.1f}%", f"ATP {A['dp_simple']:+.1f}%")
    c[2].metric("Program B lift", f"{B['dv_simple']:+.1f}%", f"ATP {B['dp_simple']:+.1f}%")
    c[3].metric("2026 recommendation", "Redeploy A", "B as tactical lever")

    st.markdown(f"""
<div class="rec">
<b>1. Use an elasticity of approximately -4 for 2026 go-to-market decisions.</b>
Naive estimation gives {models['naive']['e']:.1f} but is confounded by the August 2024 refresh;
four corrected methods converge between about -3 and -5, led by the refresh-controlled estimate
of {models['refresh']['e']:.1f} (p = {models['refresh']['p']:.2f}). Each 1 percent of price buys
roughly 4 percent of volume. Every program verdict below is robust across the whole range
(test it with the sidebar slider).
</div>
<div class="rec">
<b>2. Program A created demand; Program B bought it at market rate.</b>
A: {A['dv_simple']:+.1f} percent volume for {A['dp_simple']:+.1f} percent ATP (arc {A['arc']:.0f}),
with about {A['campaign_lift']/A['dv_simple']*100:.0f} percent of the lift campaign-driven, and
volume up {A['yoy_vol']:+.1f} percent year over year at HIGHER prices. Revenue rose about
{(A['p1']*A['v1']/(A['p0']*A['v0'])-1)*100:.0f} percent.
B: {B['dv_simple']:+.1f} percent volume for {B['dp_simple']:+.1f} percent ATP (arc {B['arc']:.1f}),
revenue up only {(B['p1']*B['v1']/(B['p0']*B['v0'])-1)*100:.1f} percent before subvention costs
that ATP does not capture.
</div>
<div class="rec">
<b>3. The segment is redistributing share, and offense is the correct posture.</b>
Highlander has lost roughly two thirds of its quarterly volume while pricing up without product
news; Explorer's share of the six-vehicle set rose from {shr0:.1f} to {shr1:.1f} percent and the
reallocation is still in progress. A refreshed Program A is the headline play; Program B is the
standing rapid-response tool, dialed by region, trim, and credit tier; a plain price cut is
strictly dominated and should not be used.
</div>
<div class="rec">
<b>4. Protect the premium the refresh earned.</b>
The refresh converted a roughly 2 percent segment discount into up to an 8 percent premium; the 2025
programs spent it back to a roughly 4 percent discount. 2026 pricing should treat that premium as
an asset with an explicit budget, not an incidental casualty of promotions.
</div>
<div class="rec">
<b>5. Carry the uncertainty honestly, and instrument the next campaign.</b>
Twenty-one observations, ATP endogenous to trim mix, program effects identified from single
quarters. Re-estimate quarterly as 2026 actuals accrue, and run the next Program A with regional
holdouts so its ROI is measured by design rather than inferred afterward.
</div>""", unsafe_allow_html=True)

    st.plotly_chart(fig_atp_volume(), width="stretch", key="ins_atp")
    insight("The single chart to remember: one refresh, two programs, and a volume trajectory "
            "that ends 2025 at record levels with ATP back near early-2024 levels. 2026's job is "
            "to keep the volume without giving back more price.")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_forest(), width="stretch", key="ins_forest")
        insight("All corrected estimates agree; only the naive one dissents, and it is wrong for "
                "a known reason. The -4 planning value is evidence, not convention.")
    with c2:
        st.plotly_chart(fig_efficiency(), width="stretch", key="ins_eff")
        insight("The one-number comparison for the program decision: A returned about 21 volume "
                "points per ATP point, five times what any price-based mechanism can deliver.")

    st.markdown("**2026 monitoring plan**")
    st.markdown("""
- Re-estimate elasticity each quarter with the refresh and program controls, and retire the -4 value if the interval shifts.
- Track the ATP premium versus segment monthly as the pricing-power KPI, with an explicit floor.
- Watch Palisade incentive activity as the leading indicator for deploying Program B.
- Design the next Program A with holdout regions and pre-registered success metrics.
""")


# ================================================================ TAB 8 ABOUT
with tabs[7]:
    st.subheader("About this project")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("""
#### Author
**Sanaz Vasheghani**

Manufacturing Data Scientist at General Motors (Warren, MI). Previously Data Scientist in
Sales and Marketing at Naseh Holding, a Toyota subsidiary, from 2016 to 2023, applying
analytics to automotive sales, pricing, and marketing decisions.

M.S. in Data Science and Business Analytics, Wayne State University (GPA 3.93). MBA in
Business Management and Marketing and B.A. in Business Management and Marketing, Azad
University.

Author of peer-reviewed research on adaptive and dynamic ensemble learning for robust image
classification, including work published in Knowledge-Based Systems (Q1).

[Google Scholar](https://scholar.google.com/citations?user=RxuIyU4AAAAJ&hl=en) ·
[LinkedIn](https://www.linkedin.com/in/sanaz-vasheghani/)
""")
    with c2:
        st.markdown("""
#### Goal
Interactive companion to the Yield Management interview business case: estimate the Ford
Explorer's price elasticity from January 2024 to September 2025 actuals, evaluate the two 2025
sales programs against that baseline, and recommend a 2026 go-to-market strategy, with every
figure computed live from the source workbook.

#### Methodology summary
1. **Data**: monthly ATP and volume for six midsize SUVs plus segment ATP, read from Google
   Drive (repo fallback); quarterly aggregates rebuilt as volume-weighted means and verified
   against the workbook's own quarterly tab.
2. **Elasticity**: six approaches (naive log-log OLS, refresh-controlled OLS, relative-price,
   year-over-year differences, fully controlled YoY, clean-window OLS) plus model-free arc
   elasticities. The August 2024 mid-cycle refresh is treated as a demand shifter throughout,
   and the refresh-controlled model is selected as primary with year-over-year as the
   robustness check.
3. **Program evaluation**: quarter-over-quarter and year-over-year changes, arc elasticities,
   revenue proxies, and a decomposition of lift into price-explained and campaign-driven
   components at an adjustable baseline elasticity (sidebar).
4. **Limitations**: 21 monthly observations, ATP endogenous to trim mix, single-quarter program
   identification, and financing subvention costs not visible in ATP.

#### Stack
Python, Streamlit, Plotly, statsmodels, pandas · Data on Google Drive · Deployed from GitHub
via Streamlit Community Cloud · Optional live AI analysis via the Claude API (key entered in
the sidebar).
""")
