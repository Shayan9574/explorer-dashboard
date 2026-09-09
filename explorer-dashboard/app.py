"""Ford Explorer Yield Management Dashboard."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from data_loader import load_data, VEHICLES, COMPETITORS
from analysis import run_models, program_metrics, arc_elasticity
from ai import ai_section

# ---------------------------------------------------------------- page setup
st.set_page_config(page_title="Explorer Yield Management", page_icon="🚙",
                   layout="wide")

FORD = "#003478"
LIGHT = "#4a7bd0"
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
</style>""", unsafe_allow_html=True)


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
prog = program_metrics(quarterly)
A, B = prog["A"], prog["B"]

st.title("Ford Explorer Yield Management Dashboard")
st.caption(f"Price elasticity and 2026 sales program assessment  |  "
           f"Monthly ATP and volume, Jan 2024 to Sep 2025  |  Data source: {source}")

tabs = st.tabs(["📊 Dataset", "📐 Price Elasticity", "🇺🇸 Program A", "💳 Program B",
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
    fig.add_vline(x=-4, line_dash="dash", line_color="#dc2626",
                  annotation_text="Recommended: -4", annotation_font=dict(color="#dc2626"))
    fig.update_layout(xaxis_title="Estimated price elasticity (95% CI where applicable)")
    return style_fig(fig, 430, "Elasticity estimates across methods")


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

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_quarterly(), width="stretch")
    with c2:
        st.plotly_chart(fig_scatter(), width="stretch")

    st.markdown("**Program windows vs surrounding baseline**")
    c1, c2 = st.columns(2)
    qw = quarterly.set_index("quarter")
    with c1:
        fig = go.Figure()
        fig.add_bar(x=["Q1 2025 baseline", "Q2 2025 Program A"],
                    y=[qw.loc["2025Q1", "vol_Explorer"], qw.loc["2025Q2", "vol_Explorer"]],
                    marker_color=[FORD, AMBER], text=[f"{A['v0']:,.0f}", f"{A['v1']:,.0f}"],
                    textposition="outside")
        fig.update_layout(yaxis_title="Units")
        st.plotly_chart(style_fig(fig, 320, f"Program A: volume {A['dv_simple']:+.1f}% on ATP {A['dp_simple']:+.1f}%"),
                        width="stretch")
    with c2:
        fig = go.Figure()
        fig.add_bar(x=["Q2 2025 baseline", "Q3 2025 Program B"],
                    y=[qw.loc["2025Q2", "vol_Explorer"], qw.loc["2025Q3", "vol_Explorer"]],
                    marker_color=[FORD, GREEN], text=[f"{B['v0']:,.0f}", f"{B['v1']:,.0f}"],
                    textposition="outside")
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
- **Seasonality is visible** (summer softness in 2024), which motivates the year-over-year specification used in the elasticity tab.
""",
        f"Monthly Explorer ATP and volume Jan 2024 to Sep 2025. Refresh Aug 2024 (+8.9% ATP). "
        f"Program A Q2 2025: ATP {A['dp_simple']:+.1f}%, volume {A['dv_simple']:+.1f}%. "
        f"Program B Q3 2025: ATP {B['dp_simple']:+.1f}%, volume {B['dv_simple']:+.1f}%.",
        "Analyze the dataset patterns and their implications for elasticity estimation.")


# ================================================================ TAB 2 ELASTICITY
with tabs[1]:
    st.subheader("Estimated Explorer price elasticity")
    st.markdown("""
Six complementary approaches are computed live from the data. Each answers a weakness of the others,
and their convergence is the basis of the recommendation.
""")

    rows = []
    notes = {
        "naive": "Simple, but biased toward zero: the refresh raised price and demand together (omitted variable).",
        "refresh": "Controls the refresh as a demand shifter. Preferred monthly specification.",
        "relative": "Prices Explorer against the segment, absorbing market-wide price movements.",
        "yoy": "Twelve-month differences remove seasonality without estimating monthly dummies.",
        "yoy_full": "Adds refresh and program controls so promotional quarters do not masquerade as price response.",
        "clean": "Only post-refresh, pre-program months: no confounders, but just eight observations.",
    }
    for k in ["naive", "refresh", "relative", "yoy", "yoy_full", "clean"]:
        m = models[k]
        rows.append({"Method": m["label"], "Elasticity": round(m["e"], 2),
                     "p-value": round(m["p"], 3), "R²": round(m["r2"], 2),
                     "n": m["n"], "Comment": notes[k]})
    rows.append({"Method": "7. Arc elasticity, Program B window", "Elasticity": round(B["arc"], 2),
                 "p-value": None, "R²": None, "n": 2,
                 "Comment": "Model-free two-point check on a quarter driven mainly by price."})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.plotly_chart(fig_forest(), width="stretch")

    st.markdown(f"""
<div class="rec"><b>Recommendation: plan 2026 around an elasticity of approximately -4
(defensible range -3 to -5).</b><br>
The refresh-controlled model gives {models['refresh']['e']:.1f} (p = {models['refresh']['p']:.2f}),
the year-over-year model {models['yoy']['e']:.1f} (p = {models['yoy']['p']:.2f}), the fully
controlled YoY model {models['yoy_full']['e']:.1f}, and the clean window {models['clean']['e']:.1f}.
The naive {models['naive']['e']:.1f} should be rejected as confounded. A value near -4 is also
consistent with published estimates for mainstream midsize SUVs, a heavily cross-shopped segment.
Practical reading: each 1 percent of transaction price given up should buy roughly 4 percent
of volume, absent other stimulus.</div>""", unsafe_allow_html=True)

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

    e_grid = np.linspace(-5, -3, 3)
    p_grid = np.linspace(-6, 0, 50)
    fig = go.Figure()
    for e in e_grid:
        fig.add_scatter(x=p_grid, y=e * p_grid, mode="lines",
                        name=f"elasticity {e:.0f}",
                        line=dict(width=2.5 if e == -4 else 1.2,
                                  color=FORD if e == -4 else "#9db4d6"))
    fig.add_scatter(x=[A["dp_simple"]], y=[A["dv_simple"]], mode="markers+text",
                    text=["Program A"], textposition="top center", name="Program A",
                    marker=dict(size=13, color=AMBER))
    fig.add_scatter(x=[B["dp_simple"]], y=[B["dv_simple"]], mode="markers+text",
                    text=["Program B"], textposition="top center", name="Program B",
                    marker=dict(size=13, color=GREEN))
    fig.update_layout(xaxis_title="% change in ATP", yaxis_title="Expected % change in volume")
    st.plotly_chart(style_fig(fig, 400, "What the baseline elasticity predicts vs what the programs delivered"),
                    width="stretch")

    ai_section(
        "elast",
        f"""
- **Convergence, not a single model, drives the answer.** Four independent corrections for the refresh confound land between {models['clean']['e']:.1f} and {models['yoy']['e']:.1f}; only the knowingly biased naive model falls outside.
- **The naive estimate is a teaching case of omitted variable bias**: the refresh raised price about 9 percent and demand simultaneously, dragging the naive slope toward zero ({models['naive']['e']:.1f}).
- **Program A sits about 20 points above every plotted demand curve**, confirming its lift was predominantly a campaign-driven demand shift. Program B sits close to the curves, behaving like a conventional price action.
- **Uncertainty is real**: with 21 observations the preferred estimate carries a standard error near {models['refresh']['se']:.1f}. The range -3 to -5 should be carried into 2026 planning, not a false point estimate.
""",
        f"Elasticity estimates: naive {models['naive']['e']:.2f}, refresh {models['refresh']['e']:.2f}, "
        f"relative {models['relative']['e']:.2f}, YoY {models['yoy']['e']:.2f}, "
        f"YoY full {models['yoy_full']['e']:.2f}, clean {models['clean']['e']:.2f}, arc B {B['arc']:.2f}.",
        "Compare the elasticity estimation methods and justify the recommended planning value.")


# ================================================================ TAB 3 PROGRAM A
with tabs[2]:
    st.subheader("Program A: \"From America For America\" employee pricing (Q2 2025)")
    c = st.columns(4)
    c[0].metric("Volume vs Q1 2025", f"{A['dv_simple']:+.1f}%")
    c[1].metric("ATP vs Q1 2025", f"{A['dp_simple']:+.1f}%")
    c[2].metric("Arc elasticity", f"{A['arc']:.1f}", help="vs baseline of about -4")
    c[3].metric("Volume vs Q2 2024", f"{A['yoy_vol']:+.1f}%", help=f"with ATP {A['yoy_atp']:+.1f}% YoY")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Waterfall(
            x=["Q1 2025 volume", "Price effect (e = -4)", "Campaign / brand effect", "Q2 2025 volume"],
            measure=["absolute", "relative", "relative", "total"],
            y=[A["v0"], A["v0"] * A["price_lift"] / 100,
               A["v0"] * A["campaign_lift"] / 100, None],
            connector=dict(line=dict(color="#cbd5e1")),
            increasing=dict(marker_color=AMBER), totals=dict(marker_color=FORD),
            decreasing=dict(marker_color="#94a3b8")))
        fig.update_layout(yaxis_title="Units")
        st.plotly_chart(style_fig(fig, 400, "Decomposition of the Q2 2025 lift"), width="stretch")
    with c2:
        dfA = monthly[monthly["date"].between("2025-01-01", "2025-06-30")]
        fig = go.Figure()
        fig.add_bar(x=dfA["date"], y=dfA["vol_Explorer"],
                    marker_color=[AMBER if p else FORD for p in dfA["progA"]])
        fig.add_scatter(x=dfA["date"], y=dfA["atp_Explorer"], yaxis="y2", name="ATP",
                        line=dict(color="#334155", width=2))
        fig.update_layout(yaxis_title="Units",
                          yaxis2=dict(title="ATP ($)", overlaying="y", side="right"),
                          showlegend=False)
        st.plotly_chart(style_fig(fig, 400, "Monthly view: baseline (blue) vs Program A (amber)"),
                        width="stretch")

    with st.expander("Detailed calculation: arc elasticity and decomposition"):
        st.markdown(f"""
Arc elasticity = ({A['dv']:.2f}% volume) / ({A['dp']:.2f}% price) = **{A['arc']:.1f}**

Decomposition at baseline elasticity -4: the {A['dp_simple']:.2f} percent ATP decline explains about
**{A['price_lift']:.1f} points** of lift; the remaining **{A['campaign_lift']:.1f} points**
of the {A['dv_simple']:.1f} percent total is attributable to the campaign itself
(messaging, earned media, perceived-deal framing). An arc value of {A['arc']:.1f} is not a price
elasticity at all: it is the signature of a demand curve shifting outward.
""")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<span class="badge badge-a">BENEFITS</span>', unsafe_allow_html=True)
        st.markdown("""
- Exceptional yield: about 26 percent more volume for about 1 percent of ATP.
- Genuine demand creation, not just movement along the curve.
- Simple, transparent message ("employee pricing for all") with strong earned media and a patriotic halo.
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
- Program A delivered a **{A['dv_simple']:.1f} percent volume lift at only {abs(A['dp_simple']):.1f} percent ATP cost**, an efficiency no pure price action can match at any plausible elasticity.
- The implied arc value of {A['arc']:.1f} versus a baseline near -4 means roughly **four fifths of the lift was campaign-driven demand creation**, the scarcest and most valuable kind of lift.
- The strategic risk is not Q2 2025; it is 2027. Repeated employee-pricing events convert a brand moment into a discount expectation.
""",
        f"Program A Q2 2025: volume {A['dv_simple']:+.1f}%, ATP {A['dp_simple']:+.1f}%, arc {A['arc']:.1f}, "
        f"price-explained {A['price_lift']:.1f} pts, campaign {A['campaign_lift']:.1f} pts.",
        "Assess Program A's performance, decomposition, and strategic risks.")


# ================================================================ TAB 4 PROGRAM B
with tabs[3]:
    st.subheader("Program B: low APR and lease offers (Q3 2025)")
    c = st.columns(4)
    c[0].metric("Volume vs Q2 2025", f"{B['dv_simple']:+.1f}%")
    c[1].metric("ATP vs Q2 2025", f"{B['dp_simple']:+.1f}%")
    c[2].metric("Arc elasticity", f"{B['arc']:.1f}", help="vs baseline of about -4")
    c[3].metric("Volume vs Q3 2024", f"{B['yoy_vol']:+.1f}%", help=f"with ATP {B['yoy_atp']:+.1f}% YoY; Q3 2024 was a launch-transition low")

    c1, c2 = st.columns(2)
    with c1:
        exp_lift = prog["baseline_e"] * B["dp_simple"]
        fig = go.Figure(go.Bar(
            x=["Expected lift from price alone (e = -4)", "Actual lift"],
            y=[exp_lift, B["dv_simple"]], marker_color=["#94a3b8", GREEN],
            text=[f"{exp_lift:+.1f}%", f"{B['dv_simple']:+.1f}%"], textposition="outside"))
        fig.update_layout(yaxis_title="% volume change vs Q2 2025")
        st.plotly_chart(style_fig(fig, 400, "Program B under-delivered vs the baseline price response"),
                        width="stretch")
    with c2:
        dfB = monthly[monthly["date"].between("2025-04-01", "2025-09-30")]
        fig = go.Figure()
        fig.add_bar(x=dfB["date"], y=dfB["vol_Explorer"],
                    marker_color=[GREEN if p else AMBER for p in dfB["progB"]])
        fig.add_scatter(x=dfB["date"], y=dfB["atp_Explorer"], yaxis="y2", name="ATP",
                        line=dict(color="#334155", width=2))
        fig.update_layout(yaxis_title="Units",
                          yaxis2=dict(title="ATP ($)", overlaying="y", side="right"),
                          showlegend=False)
        st.plotly_chart(style_fig(fig, 400, "Monthly view: Program A quarter (amber) into Program B (green)"),
                        width="stretch")

    with st.expander("Detailed calculation: arc elasticity and expectation gap"):
        st.markdown(f"""
Arc elasticity = ({B['dv']:.2f}% volume) / ({B['dp']:.2f}% price) = **{B['arc']:.1f}**

At the baseline elasticity of -4, the {abs(B['dp_simple']):.2f} percent ATP decline should have
produced about **{prog['baseline_e'] * B['dp_simple']:.1f} percent** volume; the observed
**{B['dv_simple']:.1f} percent** falls short by roughly
{prog['baseline_e'] * B['dp_simple'] - B['dv_simple']:.1f} points. Two honest caveats: the Q2
comparison base was inflated by Program A carryover, and APR subvention is a real cost that sits
**outside** ATP, so the true effective discount, and thus the shortfall in efficiency, is larger
than the ATP series shows.
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
- Program B behaved like a conventional price action: **{B['dv_simple']:.1f} percent volume for {abs(B['dp_simple']):.1f} percent ATP**, an arc of {B['arc']:.1f} against a baseline near -4.
- Judged on ATP alone it bought volume **below the market rate**, and the unrecorded APR subvention makes true efficiency worse than the chart shows.
- Its real value is structural, not promotional: residual-value protection and the lease-return pipeline. That argues for tactical, targeted use rather than a headline campaign.
""",
        f"Program B Q3 2025: volume {B['dv_simple']:+.1f}%, ATP {B['dp_simple']:+.1f}%, arc {B['arc']:.1f}.",
        "Assess Program B's performance relative to baseline elasticity and its strategic role.")


# ================================================================ TAB 5 COMPARISON
with tabs[4]:
    st.subheader("Program A vs Program B")
    comp = pd.DataFrame({
        "Metric": ["Quarter", "Mechanism", "ATP change", "Volume change", "Arc elasticity",
                   "Lift beyond price response", "Hidden costs", "Brand effect",
                   "Repeatability", "Residual value impact", "Execution risk"],
        "Program A": ["Q2 2025", "Employee pricing + campaign", f"{A['dp_simple']:+.1f}%",
                      f"{A['dv_simple']:+.1f}%", f"{A['arc']:.1f}",
                      f"About {A['campaign_lift']:.0f} points (demand shift)",
                      "Margin compression masked by mix", "Strong positive halo",
                      "Low: novelty dependent", "Negative if repeated (reference price anchor)",
                      "Moderate (dealer coordination)"],
        "Program B": ["Q3 2025", "Low APR + lease offers", f"{B['dp_simple']:+.1f}%",
                      f"{B['dv_simple']:+.1f}%", f"{B['arc']:.1f}",
                      "None observed",
                      "APR subvention outside ATP", "Neutral, industry standard",
                      "High: standard playbook", "Protective vs cash discounts",
                      "Low"],
    })
    st.dataframe(comp, width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_bar(name="ATP given up (%)", x=["Program A", "Program B"],
                    y=[abs(A["dp_simple"]), abs(B["dp_simple"])], marker_color="#94a3b8")
        fig.add_bar(name="Volume gained (%)", x=["Program A", "Program B"],
                    y=[A["dv_simple"], B["dv_simple"]], marker_color=[AMBER, GREEN])
        fig.update_layout(barmode="group", yaxis_title="%")
        st.plotly_chart(style_fig(fig, 380, "Cost vs lift"), width="stretch")
    with c2:
        effA = A["dv_simple"] / abs(A["dp_simple"])
        effB = B["dv_simple"] / abs(B["dp_simple"])
        fig = go.Figure(go.Bar(x=["Program A", "Program B", "Baseline elasticity"],
                               y=[effA, effB, 4.0], marker_color=[AMBER, GREEN, FORD],
                               text=[f"{effA:.1f}", f"{effB:.1f}", "4.0"], textposition="outside"))
        fig.update_layout(yaxis_title="Volume % gained per ATP % given up")
        st.plotly_chart(style_fig(fig, 380, "Promotional efficiency (higher is better)"),
                        width="stretch")

    st.markdown("**Which program in which scenario**")
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
- **On observed 2025 evidence, A dominates on efficiency**: {A['dv_simple'] / abs(A['dp_simple']):.0f} points of volume per point of ATP versus {B['dv_simple'] / abs(B['dp_simple']):.1f} for B and 4 for a plain price cut.
- **The programs are not substitutes.** A is a demand-creation event whose power decays with repetition; B is a durable, targetable operating lever with structural side benefits.
- **Recommended 2026 posture**: one refreshed Program A event as the headline (new creative, protected novelty), Program B held as the always-available tactical tool, dialed by region and credit tier, and as the default response to competitor incentive escalation.
""",
        f"A: {A['dv_simple']:+.1f}% vol / {A['dp_simple']:+.1f}% ATP, arc {A['arc']:.1f}. "
        f"B: {B['dv_simple']:+.1f}% vol / {B['dp_simple']:+.1f}% ATP, arc {B['arc']:.1f}. Baseline -4.",
        "Compare the two programs and recommend a 2026 deployment strategy by scenario.")


# ================================================================ TAB 6 SEGMENTATION
with tabs[5]:
    st.subheader("Competitive segment analysis")
    st.plotly_chart(fig_share_area(), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for v in VEHICLES:
            fig.add_scatter(x=monthly["date"], y=monthly[f"atp_{v}"], name=v,
                            line=dict(color=PALETTE[v], width=2.6 if v == "Explorer" else 1.3))
        fig.update_layout(yaxis_title="ATP ($)")
        st.plotly_chart(style_fig(fig, 420, "ATP by nameplate"), width="stretch")
    with c2:
        q0, q1 = quarterly.iloc[0], quarterly.iloc[-1]
        chg = [(q1[f"vol_{v}"] / q0[f"vol_{v}"] - 1) * 100 for v in VEHICLES]
        fig = go.Figure(go.Bar(x=VEHICLES, y=chg,
                               marker_color=[PALETTE[v] for v in VEHICLES],
                               text=[f"{c:+.0f}%" for c in chg], textposition="outside"))
        fig.update_layout(yaxis_title="% change in quarterly volume")
        st.plotly_chart(style_fig(fig, 420, "Volume change, Q1 2024 to Q3 2025"),
                        width="stretch")

    qv = quarterly.set_index("quarter")
    fig = go.Figure()
    for v in VEHICLES:
        fig.add_scatter(x=[qv.loc["2025Q3", f"atp_{v}"]], y=[qv.loc["2025Q3", f"vol_{v}"]],
                        mode="markers+text", text=[v], textposition="top center", name=v,
                        marker=dict(size=16, color=PALETTE[v]))
    fig.update_layout(xaxis_title="Q3 2025 ATP ($)", yaxis_title="Q3 2025 volume (units)")
    st.plotly_chart(style_fig(fig, 420, "Price-volume positioning, Q3 2025"), width="stretch")

    hl0 = int(qv.loc["2024Q1", "vol_Highlander"]); hl1 = int(qv.loc["2025Q3", "vol_Highlander"])
    ai_section(
        "seg",
        f"""
- **The Highlander collapse is the segment's dominant story**: {hl0:,} units in Q1 2024 to {hl1:,} in Q3 2025 ({(hl1 / hl0 - 1) * 100:.0f} percent) as its ATP climbed above 51,000 dollars. That volume is being redistributed, and Explorer is a natural claimant.
- **Traverse repositioned upward** (ATP from about 41,000 to about 48,500 dollars after its redesign) and grew volume anyway, showing the segment tolerates premium pricing when backed by fresh product, the same effect Explorer's refresh produced.
- **Explorer exits Q3 2025 as the volume leader** at the segment's mid-price point, with Palisade the most aggressive challenger.
- Strategic read for 2026: the share opportunity vacated by Highlander favors an offensive posture, which strengthens the case for a Program A style demand-creation event.
""",
        f"Highlander volume {hl0} to {hl1}; Traverse ATP repositioned up ~18%; Explorer Q3 2025 leader.",
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
Naive estimation gives -1.6 but is confounded by the August 2024 refresh; four corrected methods
converge between about -3 and -5. Each 1 percent of price buys roughly 4 percent of volume.
</div>
<div class="rec">
<b>2. Program A created demand; Program B bought it at market rate.</b>
A: {A['dv_simple']:+.1f} percent volume for {A['dp_simple']:+.1f} percent ATP (arc {A['arc']:.0f});
about {A['campaign_lift']:.0f} points of that lift came from the campaign, not the price.
B: {B['dv_simple']:+.1f} percent volume for {B['dp_simple']:+.1f} percent ATP (arc {B['arc']:.1f}),
with APR subvention cost sitting outside ATP.
</div>
<div class="rec">
<b>3. Redeploy a refreshed Program A as the 2026 headline; hold B as the precision tool.</b>
Protect A's novelty with new creative and deliberate timing. Use B dialed by region, trim, and
credit tier, and as the standing response to competitor incentive escalation. The Highlander
collapse has opened share for the taking.
</div>
<div class="rec">
<b>4. Carry the uncertainty honestly.</b>
Twenty-one observations, ATP endogenous to trim mix, program effects identified from single
quarters. Re-estimate quarterly as 2026 actuals accrue, and instrument the next campaign
(regional holdouts) so program ROI is measured by design rather than inferred afterward.
</div>""", unsafe_allow_html=True)

    st.plotly_chart(fig_atp_volume(), width="stretch", key="ins_atp")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_forest(), width="stretch", key="ins_forest")
    with c2:
        st.plotly_chart(fig_share_area(), width="stretch", key="ins_share")


# ================================================================ TAB 8 ABOUT
with tabs[7]:
    st.subheader("About this project")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("""
#### Author
**Shayan**

Senior Data Science Engineer and PhD candidate in Industrial and Systems Engineering
(Wayne State University). Research applies optimization, machine learning, and AI to
decision problems; peer reviewer for Applied Soft Computing and Information
Processing and Management.
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
   elasticities. The August 2024 mid-cycle refresh is treated as a demand shifter throughout.
3. **Program evaluation**: quarter-over-quarter and year-over-year changes, arc elasticities,
   and a decomposition of lift into price-explained and campaign-driven components at the
   recommended baseline elasticity.
4. **Limitations**: 21 monthly observations, ATP endogenous to trim mix, single-quarter program
   identification, and financing subvention costs not visible in ATP.

#### Stack
Python, Streamlit, Plotly, statsmodels, pandas · Data on Google Drive · Deployed from GitHub
via Streamlit Community Cloud · Optional live analysis via the Anthropic API.
""")
