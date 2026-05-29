# HydroTune Paper — AI Drawing Prompts for All Figures

> Each prompt below is designed to be directly usable by an AI image/diagram generation tool.  
> It describes the visual layout, elements, labels, colors, and spatial relationships in natural language.

---

## Figure 1 — HydroAgent Workflow Diagram

**Type:** Horizontal flowchart with 6 connected stages.

```
Prompt:

Create a horizontal flowchart diagram with 6 rounded-rectangle nodes arranged left-to-right, connected by thick arrows. Use a clean professional color scheme: light blue for data-related nodes, green for computation, orange for LLM/AI, and dark blue for output.

Node 1 (leftmost, light blue): "Data Upload" — icon of a file/folder with upward arrow. Sub-label: "Single-file or multi-file CSV input".

Node 2 (green): "LLM Pre-Analysis" — icon of a magnifying glass over a chart. Sub-label: "Data quality check, time-scale inference, flood event detection".

Node 3 (blue): "Semantic Encoding" — icon of a tag/label with attributes. Sub-label: "15-field FloodSemanticCode: magnitude, peak shape, rise/recession, rainfall-driven".

Node 4 (green): "Multi-Algorithm Calibration" — icon of a tuning slider or gear. Sub-label: "Tank (3p) / HBV (10p) / XAJ (21p) with 5 optimizers".

Node 5 (orange): "LLM Analysis & Ranking" — icon of a brain or neural network. Sub-label: "Model ranking, overfitting diagnosis, strategy recommendations".

Node 6 (rightmost, dark blue): "Intelligent Report" — icon of a document with checkmark. Sub-label: "7-section diagnostic report with embedded charts".

At the bottom of the diagram, add a feedback loop arrow curving from node 6 back to node 4, labeled "Iterative refinement (user-in-the-loop)".

Above the flowchart, add the title: "HydroAgent: LLM-Guided Automated Calibration Workflow".

Use: professional academic style, no 3D effects, 2px black borders on nodes, white background, legend at bottom-right showing color meaning.
```

---

## Figure 2 — Semantic Encoding Pipeline + P-III Frequency Curve

**Type:** Two-panel figure. (a) Flow diagram, (b) Statistical plot.

### Panel (a) — Encoding Pipeline

```
Prompt:

Create a vertical pipeline diagram with 4 stages, top to bottom, with connecting downward arrows.

Stage 1 (top, label "Raw Hydrograph"): Draw a small time-series line chart showing a flood hydrograph — a blue line starting at a low baseflow, rising sharply to a high peak, then gradually receding. The x-axis is labeled "Time steps (t₁ ... tₚ ... tₑ)", y-axis "Flow Q(t)". Highlight the peak point with a red dot.

Stage 2 (label "Feature Extraction"): Show 6 extracted feature boxes arranged in a grid (3×2):
- "Peak Flow = 850 m³/s"
- "Flood Volume = 1.2×10⁷ m³"
- "Duration = 72 steps"
- "Rise Rate = 6.3 m³/s per step"
- "Recession Rate = 2.1 m³/s per step"
- "Runoff Coeff φ = 0.42"

Draw thin dotted lines from specific points on the hydrograph to the corresponding feature boxes (e.g., from peak to "Peak Flow").

Stage 3 (label "Semantic Classification"): Show classification categories as colored badges/tags:
- "Magnitude: LARGE" (red badge)
- "Rise Type: RAPID" (orange badge)
- "Recession: MODERATE" (yellow badge)
- "Peak Shape: SHARP" (purple badge)
- "Rainfall-Driven: HIGH" (green badge)
- "Representative Score: 82%"

Stage 4 (bottom, label "Structured Encoding"): Show a curly-brace representation or JSON-like key-value box:
```
{
  event_id: "F001",
  magnitude: "large",
  peak_flow: 850,
  flood_volume: 1.2e7,
  rise_type: "rapid",
  recession_type: "moderate",
  peak_shape: "sharp",
  rainfall_driven: "high",
  rep_score: 82
}
```

Add title: "(a) Flood Semantic Encoding Pipeline".
Use professional academic style with clean box borders.
```

### Panel (b) — P-III Frequency Curve

```
Prompt:

Create a P-III (Pearson Type III) frequency analysis plot. This is a statistical probability plot.

Plot area: Use Hazen probability grid paper style — the x-axis is "Exceedance Probability P (%)" with values 0.01, 0.1, 1, 5, 10, 20, 50, 80, 90, 95, 99, 99.9. The y-axis is "Peak Flow (m³/s)" on a normal logarithmic or arithmetic scale from 0 to 2000.

Draw the empirical points as small solid black circles connected by a thin dashed line, representing observed annual maximum peak flows plotted at their Hazen plotting positions.

Draw the P-III theoretical frequency curve as a thick solid blue line that fits through the empirical points. The curve should be concave-up on the left side (low exceedance = high flows) and flatten on the right.

Annotate at the top-right corner:
"P-III Parameters:
X̄ (mean) = 520 m³/s
Cv = 0.45
Cs = 1.35"

Draw a horizontal dashed line at y = 1000 with label "2-year return period" and another at y = 1500 with "100-year".

Add title: "(b) P-III Frequency Curve of Annual Maximum Peak Flows".

Use academic style, grid lines in very light gray, clean font.
```

---

## Figure 3 — Algorithm Convergence Curves + Model Structures

**Type:** Two-panel figure. (a) Line chart, (b) Three schematic diagrams.

### Panel (a) — Algorithm Convergence

```
Prompt:

Create a single line chart showing optimization convergence.

X-axis: "Iterations" (logarithmic scale, 1 to 1000).
Y-axis: "Objective Function Value (1 - NSE)" (decreasing, from 0.8 to 0.1).

Plot 5 lines, each representing one algorithm:
- "Two-Stage" — solid blue line, thickest, converges fastest and lowest (reaches ~0.12 by iteration ~50)
- "PSO" — dashed green line with circle markers every 5 iterations, converges to ~0.18 by iteration ~200
- "SCE-UA" — dotted red line with triangle markers, staircase-like drops, converges to ~0.15 by iteration ~300
- "DE" — dash-dot orange line, gradual descent, converges to ~0.20 by iteration ~500
- "GA" — dotted purple line, high variance with small fluctuations, converges to ~0.25 by iteration ~800

Add a subtle horizontal shaded band from y=0.12 to y=0.15 labeled "Target zone".

Add legend at bottom-right with all 5 algorithms and their line styles.

Add title: "(a) Algorithm Convergence: Objective vs. Iterations".

Academic style, white background, thin gray grid lines, 10pt font on axes.
```

### Panel (b) — Model Structure Schematics

```
Prompt:

Create three schematic diagrams of conceptual hydrological models, arranged horizontally side by side. Each occupies roughly 1/3 of the panel width. Use consistent visual language: tanks for storage, curved arrows for fluxes, straight arrows for inputs.

**Left sub-panel — Tank Model (3 parameters):**
Draw 3 stacked rectangular tanks connected vertically. Top tank labeled "Surface (k1)" with input arrow "Precipitation P" entering from top. Side outlet with arrow → "Fast Runoff Q1". Middle tank "Intermediate (c)" receiving overflow from top tank. Side outlet → "Interflow Q2". Bottom tank "Baseflow (k2)" receiving overflow from middle tank. Side outlet → "Slow Baseflow Q3". Sum all 3 outputs → "Total Streamflow Q".

Annotate: "3 parameters: k1, k2, c".

**Center sub-panel — HBV Model (10 parameters):**
Draw a more complex structure. Start with "Precipitation P" and "Temperature T" inputs. Top box: "Snow Routine" → outputs "Rainfall + Snowmelt". Middle box: "Soil Moisture Zone (fc, beta, lp)" with actual evapotranspiration arrow "ETa" going out. Bottom box: "Response Function" with upper zone "UZ (k0, k1, perc)" and lower zone "LZ (k2)". Upper zone has quick runoff arrow "Q0", lower zone has slow baseflow arrow "Q1". Sum → "Total Runoff Q".

Annotate: "10 parameters: fc, beta, k0, k1, lp, perc, uzl, k2, …".

**Right sub-panel — Xinanjiang Model (21 parameters):**
Draw a watershed schematic. Top: "Precipitation P" and "ETp" entering. Layer 1: "Evapotranspiration (WUM, WLM, C)" — 3-layer soil moisture accounting. Layer 2: "Runoff Generation (B, WM, IM)" — split into surface, interflow, and groundwater components via curves. Layer 3: "Runoff Separation (SM, EX, K, KI, KG)" — free water reservoir splitting into 3 components. Layer 4: "Flow Routing (CG, CI, CS, L, X)" — linear reservoirs for each component + Muskingum channel routing. Final output: "Streamflow Q".

Annotate: "21 parameters: B, C, WM, WUM, WLM, IM, SM, EX, K, KI, KG, CG, CI, CS, L, X, …".

Add title: "(b) Schematic of Model Structures with Parameter Counts".

Use clean academic style, soft blue fills for water bodies, no 3D.
```

---

## Figure 4 — Hybrid Physics-ML Architecture + Correction Hydrograph

**Type:** Two-panel figure. (a) Architecture diagram, (b) Line chart.

### Panel (a) — Hybrid Architecture

```
Prompt:

Create a horizontal flow diagram with 2 parallel paths that converge.

**Top path (Physical Model, blue-themed):**
Left: Box "Input Data" containing "P(t) precip" and "ET(t) evap" as small labeled entries.
Arrow → Box "Conceptual Hydrological Model (Tank/HBV/XAJ)" — a gear/tank icon.
Arrow → Box "Simulated Flow Q_sim(t)" — a blue line icon.
Arrow → Branch to "Error e(t) = Q_obs(t) - Q_sim(t)" — a red dashed box.

**Bottom path (ML Corrector, green-themed):**
Box "Feature Construction" receives input from the error branch above, showing:
"e(t-1), e(t-2), e(t-3), e(t-4), e(t-5)" (5 error lags)
"P(t-1), P(t-2), P(t-3)" (3 precipitation lags)
Arrow → Box "XGBoost Regressor" — a tree ensemble icon.
Arrow → Box "Predicted Error ê(t)".

**Convergence (orange box at right):**
Arrow from Q_sim(t) and Arrow from ê(t) merge into:
Box "Corrected Flow Q_cor(t) = Q_sim(t) + ê(t)" — a green checkmark icon.

**Training/Validation split label at top:**
"Calibration events (75%) → Train XGBoost | Validation events (25%) → Evaluate".

Add title: "(a) Hybrid Physics-ML Error Correction Architecture".

Use professional academic style. The two paths should be visually parallel (top blue, bottom green), converging on the right into the orange corrected output. Connect error branch to feature construction with a dashed red line. Add a small legend box: "Blue = Physical Model, Green = ML Corrector, Orange = Hybrid Output".
```

### Panel (b) — Pre/Post Correction Hydrograph

```
Prompt:

Create a hydrograph comparison chart showing flow over time for a single validation flood event.

X-axis: "Time (hours)" from 0 to 96.
Y-axis: "Flow (m³/s)" from 0 to 1000.

Plot 3 lines against the same axes:
1. "Observed Q_obs" — solid blue line, moderate thickness. It rises from baseflow (~50) sharply to a peak of ~850 at t=36, then recedes with a slight secondary bump at t=60.
2. "Simulated Q_sim" — dashed green line. It follows the observed shape but peaks lower at ~700 (underestimates peak) and is slightly delayed by ~4 hours.
3. "Corrected Q_cor" — solid orange line. It closely tracks the observed line: correctly reaches ~850 at the peak, corrects the timing, and follows the recession more accurately than the green line.

Add a subtle vertical gray band from t=30 to t=42 labeled "Peak region" with annotation arrow pointing to the orange/blue convergence.

Add a small inset box at upper-left showing:
"Metrics:
Q_sim: NSE = 0.78, RMSE = 95
Q_cor: NSE = 0.91, RMSE = 52"

Add legend at bottom-right with all 3 lines.

Add title: "(b) Validation Event: Observed vs. Simulated vs. XGB-Corrected Hydrograph".

Academic style, clean line weights, no background fill.
```

---

## Figure 5 — Platform Interface Screenshot

**Type:** UI mockup / annotated screenshot.

```
Prompt:

Create a clean annotated mockup of a web application interface (Streamlit-style). The layout has two columns: a narrow left sidebar (~25% width) and a wide main panel (~75%).

**Left Sidebar (dark gray/blue background, white text):**
Header: "HydroTune" with a small water-drop logo.
Section "1. Upload Data":
- Radio buttons: ○ Single File ○ Multiple Files
- File upload button: "Browse files..." with dashed border
- Dropdown: "Time-step: Auto-inferred (Hourly)"

Section "2. Configure":
- Multi-select checkboxes: ☑ Tank ☑ HBV ☑ Xinanjiang
- Dropdown: "Algorithm: Two-Stage ▼"
- Slider: "XGBoost max_depth: 6 ■■■■□□"
- Slider: "XGBoost learning_rate: 0.1 ■■□□□□"
- Number input: "Warm-up period: 24 hours"
- Number input: "Catchment area: 850 km²"
- Button: "⚡ Run Calibration" (bright green, prominent)

Section "3. Status":
- Progress bar at 65% filled with blue: "Calibrating Xinanjiang..."
- Spinner icon next to "Estimated time: 2 min remaining"

**Right Main Panel (white background):**
Tab bar: "Pre-Analysis | Calibration | Report | Compare"

Currently showing "Pre-Analysis" tab:
- Panel 1 (top-left): Radar chart with 4 axes labeled "Completeness", "Continuity", "Correlation", "Quality" — values 85%, 92%, 78%, 88% forming a near-square polygon.
- Panel 2 (top-right): Small hydrograph with blue line and 3 flood events highlighted in colored bands.
- Panel 3 (bottom-left): P-III frequency plot with empirical points and fitted curve.
- Panel 4 (bottom-right): Scatter plot of precipitation vs. runoff with regression line.

Add annotation callouts (numbered circles with leader lines):
① → Sidebar file upload: "Dual-mode data ingestion"
② → Configuration checkboxes: "Human-in-the-loop parameter selection"
③ → Progress bar: "Real-time calibration feedback"
④ → Radar chart: "Automated data quality assessment"
⑤ → Hydrograph: "Flood event auto-detection"

Title: "Figure 5. HydroTune Platform Interface — Data Upload and Configuration".

Use a clean modern dark-themed sidebar, professional scientific plotting style for the charts.
```

---

## Figure 6 — Pre-Analysis Multi-Panel

**Type:** Four-panel composite figure (2×2 grid).

```
Prompt:

Create a 2×2 grid of four coordinated sub-plots sharing a common theme of automated data pre-analysis. Use consistent fonts and axis styles.

**Panel (a) — Top-Left: Data Quality Radar Chart**
Draw a radar/spider chart with 4 axes radiating from center:
- Axis 1 (top): "Completeness" — value 85%
- Axis 2 (right): "Continuity" — value 92%
- Axis 3 (bottom): "Corr(P,Q)" — value 78%
- Axis 4 (left): "Quality Level" — value 88%

Fill the polygon with semi-transparent blue (alpha 0.3). Mark each axis value with a dot. Add percentage labels at each vertex. Center text: "Overall Score: 86/100". No title inside the sub-panel.

**Panel (b) — Top-Right: P-III Frequency Curve**
Same as Figure 2(b) but smaller. Hazen grid, empirical points (black circles), theoretical P-III fit (blue line). Annotate parameters in top-right. Use compact layout. Y-axis: "Peak Flow (m³/s)", X-axis: "Exceedance Probability P(%)".

**Panel (c) — Bottom-Left: Precipitation-Runoff Scatter**
X-axis: "Event Precipitation (mm)", Y-axis: "Event Runoff (mm)".
Scatter plot with 50+ points (small blue circles), each representing one flood event. Draw a diagonal dashed line (1:1 line). Draw a linear regression line in red with gray confidence band. Annotate "R² = 0.72" and "φ_mean = 0.38". Points cluster in the lower-left with a few high outliers in the upper-right.

**Panel (d) — Bottom-Right: Hydrograph with Event Annotations**
X-axis: "Date", Y-axis: "Flow (m³/s)".
Full time series (1-2 years of data) showing baseflow with several flood peaks rising above. Each detected flood event is highlighted with a semi-transparent colored band (alternating light blue and light orange bands). Peak of each event marked with a small triangle ▲. Baseflow shown as a thin brown line along the bottom of the hydrograph.

Bottom title: "Figure 6. Automated Pre-Analysis: Data Quality Assessment, Frequency Analysis, Rainfall-Runoff Relationship, and Flood Event Detection."

Academic journal style, white background, thin borders around each sub-panel.
```

---

## Figure 7 — Layered Software Architecture

**Type:** Vertical layered architecture diagram.

```
Prompt:

Create a clean layered (tiered) architecture diagram with 5 horizontal layers stacked top-to-bottom, connected by vertical arrows between layers.

**Layer 1 (top, lightest color - gray): "Presentation Layer"**
Contains 2 boxes side by side:
- Left box: "Streamlit Web UI" with sub-items: "File Upload", "Parameter Sidebar", "Progress Callback"
- Right box: "Report Viewer" with sub-items: "Hydrograph Charts", "Comparison Tables", "Diagnostic Reports"

**Layer 2 (light blue): "Agent Orchestration Layer"**
Contains: "Agent Workflow Engine" with sub-items: "Data Quality Assessment", "Flood Detection & Encoding", "Algorithm Dispatch", "Report Triggers", "Language Adapter (EN/ZH)"

Vertical arrow from Layer 2 to Layer 3 labeled "Calibration Commands".

**Layer 3 (green): "Calibration & Correction Core"**
Contains 3 boxes:
- Left: "Unified Calibration Dispatcher" with sub-items: "Two-Stage", "PSO", "SCE-UA", "DE", "GA"
- Center: "Model Registry" (singleton) with sub-items: "Tank Adapter", "HBV Adapter", "XAJ Adapter"
- Right: "XGBoost Error Corrector" with sub-items: "Feature Builder (8 lag features)", "Event-Based Split (75/25)", "Recursive / Oracle Modes"

**Layer 4 (dark blue): "Model Engine Layer"**
Contains 3 boxes:
- "Tank Engine (NumPy)" — 3 params [k1, k2, c]
- "HBV Engine (NumPy)" — 10 params [fc, beta, k0, k1, lp, perc, uzl, k2, …]
- "XAJ Engine (NumPy)" — 21 params [B, C, WM, WUM, WLM, IM, SM, EX, K, KI, KG, CG, CI, CS, L, X, …]

Vertical arrow from Layer 4 to Layer 5 labeled "Evaluation Metrics".

**Layer 5 (bottom, darkest - purple): "Evaluation & Output Layer"**
Contains: "Metrics Calculator" with sub-items: "NSE", "KGE", "RMSE", "MAE", "PBIAS". And "BMA Ensemble" with sub-item: "exp(NSE/T) Weighting".

**Between layers 3 and 4**, add a horizontal double-ended arrow label: "Separation of Concerns: Adapter Pattern".

Add labels on the left side of each layer: "5. Evaluation", "4. Model Engines", "3. Calibration Core", "2. Agent Layer", "1. Presentation".

Add title: "Figure 7. HydroTune Layered Software Architecture".

Use professional diagram style: solid borders, subtle gradient fills within each layer, clear text labels.
```

---

## Figure 8 — Report Generation Pipeline

**Type:** Three-panel figure showing the LLM-driven report system.

### Panel (a) — Hydrograph Comparison Example

```
Same as Figure 4(b) but compact. Use a single validation event with 3 lines (observed blue, simulated green dashed, corrected orange solid). Add small metrics box. 3-inch wide.
```

### Panel (b) — Example Calibration Report

```
Prompt:

Create a mockup of a professional calibration report page (Markdown-style rendered document).

Header: "HydroTune Calibration Report" in large bold font.
Sub-header: "Watershed A5 | HBV Model | Two-Stage Algorithm | 2025-06-15 14:30"

Section 1 (with divider): "1. Calibration Summary"
- 3-column table: Model | NSE_calib | NSE_valid | KGE | RMSE | PBIAS
- Tank row: 0.82 / 0.79 / 0.84 / 45.2 / +3.2%
- HBV row: 0.89 / 0.85 / 0.88 / 32.1 / -1.8%  ← highlighted row in light green
- XAJ row: 0.91 / 0.78 / 0.82 / 28.5 / -4.1%  ← "Overfitting risk!" annotation in red

Section 2: "2. Model Comparison"
- Bar chart comparing NSE across models (3 grouped bars: Tank=0.82, HBV=0.89, XAJ=0.91 for calibration; darker bars for validation)
- Annotation: "HBV recommended: best calibration-validation balance"

Section 3: "3. Overfitting Diagnosis"
- Text block: "Xinanjiang shows ΔNSE = 0.13 (13 percentage point drop from calibration to validation), indicating potential overfitting due to high parameter dimensionality (21 parameters vs. 57 calibration events)."

Section 4: "4. LLM-Generated Recommendations"
- Italic text block: "Based on the calibration results, the HBV model with Two-Stage optimization is recommended as the primary model for operational forecasting. It achieves the best validation NSE (0.85) while maintaining stable parameter behavior. Consider using XAJ only when supported by ≥30 calibration events."

Footer: "Generated by HydroTune AI Agent | Powered by LLM | Page 1/2"

Academic/professional style, clean formatting.
```

### Panel (c) — Visual-Aware Prompting Pipeline

```
Prompt:

Create a vertical pipeline diagram showing how visual data flows into LLM prompts.

Top: 3 separate boxes representing chart types:
- Box 1: Small hydrograph icon with label "Hydrograph Comparison"
- Box 2: Small radar chart icon with label "Data Quality Radar"
- Box 3: Small bar chart icon with label "NSE Comparison"

Arrow down from each → merging into a single box: "Base64 Encoding".

Arrow down → Box: "Construct Multi-Modal Prompt" showing a composite representation:
```
[System Instruction]
You are a hydrological analysis expert...
[User Prompt]
Analyze the calibration results for Watershed A5...
[Image 1: base64] — Hydrograph
[Image 2: base64] — Radar Chart
[Image 3: base64] — NSE Comparison
[Statistical Summary]
NSE: 0.89, KGE: 0.88, RMSE: 32.1...
```

Arrow down → Box: "LLM (Vision-Capable)". Icon of a brain with an eye symbol.

Arrow down → Box: "Structured Diagnostic Report" with label "7-section Markdown output: Flood Characteristics, Anomaly Detection, Spatiotemporal Analysis, Representativeness Assessment, Quality Evaluation, Calibration Strategy, Anomaly Handling".

Title: "Figure 8. Visual-Aware LLM Prompting: From Diagnostic Charts to Intelligent Reports".

Academic diagramming style, consistent colors.
```

---

## Figure 9 — Case 1: Small-Sample Calibration (3 Events)

**Type:** Three-panel figure. (a) Hydrograph grid, (b) Radar chart, (c) Boxplot.

### Panel (a) — 3-Event Hydrograph Comparison

```
Prompt:

Create a 1×3 grid of hydrograph sub-plots (3 columns, 1 row), each showing one flood event from the small-sample case study (Watershed A5).

Each sub-panel has the same layout:
- X-axis: "Time (hours)" from 0 to 72
- Y-axis: "Flow (m³/s)" from 0 to the event's maximum (~600-900 range)
- 3 lines per panel:
  - "Observed" — solid blue line
  - "Simulated (HBV)" — dashed green line
  - "Corrected (XGB)" — solid orange line

Panel 1 (left): "Event 1 — Calibration" 
- Sharp single peak at ~750 m³/s, rapid rise, moderate recession

Panel 2 (center): "Event 2 — Calibration"
- Broader peak at ~580 m³/s, slower rise, multi-peak (small secondary bump)

Panel 3 (right): "Event 3 — Validation"
- Highest peak at ~890 m³/s, very rapid rise (flash-flood style), steep recession

In all panels, the orange corrected line tracks closest to the blue observed line.

Small NSE labels in each panel:
- Panel 1: "NSE_sim=0.81, NSE_cor=0.87"
- Panel 2: "NSE_sim=0.76, NSE_cor=0.83"
- Panel 3: "NSE_sim=0.72, NSE_cor=0.79"

Title: "(a) Observed vs. Simulated vs. XGB-Corrected Hydrographs for 3 Events".
```

### Panel (b) — NSE Radar Chart

```
Prompt:

Create a radar/spider chart with 3 axes (one per flood event), showing NSE performance across models.

3 axes radiating from center at 120° angles, labeled "Event 1", "Event 2", "Event 3".
Scale: 0.5 to 1.0 (NSE range).

Plot 3 overlapping polygons/shapes:
- Tank: blue line, values Event1=0.78, Event2=0.74, Event3=0.68
- HBV: green line, values Event1=0.81, Event2=0.76, Event3=0.72
- XAJ: red line, values Event1=0.88, Event2=0.72, Event3=0.55 (very small on Event 3)

Add a dashed reference circle at NSE=0.70 labeled "Acceptable threshold".

Legend at bottom-right.

Title: "(b) Per-Event NSE Across Models".
```

### Panel (c) — Parameter Sensitivity Boxplot

```
Prompt:

Create a horizontal boxplot showing parameter value ranges for the best model (HBV) calibrated across the 3 leave-one-out cross-validation folds.

Y-axis: 10 parameter names listed top-to-bottom: "fc", "beta", "k0", "k1", "lp", "perc", "uzl", "k2", "k0_uzl", "k1_uzl".
X-axis: "Normalized Parameter Value [0-1]".

For each parameter, draw a boxplot (box = IQR, whiskers = 1.5×IQR, median = vertical line inside box).

Parameters with small boxes (low variance across folds): fc, beta, k1, k2 → mark these with a star ★ as "Identifiable".
Parameters with wide boxes (high variance): k0, lp, uzl → mark "Uncertain" in red text.

Add annotation: "3-fold cross-validation reveals parameter identifiability trade-offs; wider boxes indicate parameters that cannot be well-constrained with only 2 calibration events."

Title: "(c) Parameter Sensitivity: HBV Model Cross-Validation (Leave-One-Event-Out)".

Academic journal style throughout all panels. Figure title at bottom: "Figure 9. Case 1 — Small-Sample Calibration Results for Watershed A5 (3 Flood Events)".
```

---

## Figure 10 — Case 2: Large-Sample Calibration (57 Events)

**Type:** Five-panel composite figure. (a) Boxplot, (b) Convergence curves, (c) Scatter, (d) Heatmap, (e) Taylor diagram.

### Panel (a) — NSE Boxplot

```
Prompt:

Create a grouped boxplot comparing NSE distributions across models and between calibration/validation sets.

X-axis: 6 categories grouped in pairs:
- Group "Tank": 2 boxes side-by-side — "Calib" (blue fill) and "Valid" (hatched)
- Group "HBV": 2 boxes — "Calib" and "Valid"
- Group "XAJ": 2 boxes — "Calib" and "Valid"

Y-axis: "NSE" from 0.5 to 1.0.

Each box shows: median (thick line), IQR box (25th-75th percentile), whiskers (1.5×IQR), outliers as small dots.

Add horizontal dashed line at NSE=0.70 labeled "Satisfactory (NSE>0.7)".

Sample values:
- Tank: Calib median ~0.82, Valid median ~0.78
- HBV: Calib median ~0.88, Valid median ~0.85
- XAJ: Calib median ~0.91, Valid median ~0.76 (large drop)

Add bracket annotation between XAJ Calib and Valid boxes: "ΔNSE = 0.15 (overfitting)".

Title: "(a) NSE Distribution: Calibration vs. Validation Across 57 Events".
```

### Panel (b) — Algorithm Convergence

```
Same as Figure 3(a) but adapted for 57-event case:
- Deeper convergence (final values lower)
- Two-Stage converges fastest (by iteration ~30)
- Add annotation: "57-event calibration enables faster, more stable convergence"
- Compact 3-inch wide version.
```

### Panel (c) — Pre/Post Correction Scatter

```
Prompt:

Create a scatter plot comparing observed vs. simulated flow values, split into pre-correction and post-correction.

2 side-by-side scatter sub-plots with same axis limits.

**Left sub-plot: "Before XGB Correction"**
X-axis: "Observed Q (m³/s)" from 0 to 1200
Y-axis: "Simulated Q (m³/s)" from 0 to 1200
Diagonal dashed line: 1:1 perfect line.
Plot ~2000 points (all time steps from validation events). Points color-coded by event (distinct qualitative colors for 14 validation events).
Most points: below the 1:1 line (underestimation). Wide scatter.
Annotate: "NSE=0.85, R²=0.86, PBIAS=-12%".

**Right sub-plot: "After XGB Correction"**
Same axes and diagonal.
Same 2000 points but now clustered tightly around the 1:1 line.
Annotate: "NSE=0.93, R²=0.94, PBIAS=+2%".

Title: "(c) Observed vs. Simulated: Pre- and Post-XGB Error Correction (Validation Events)".

Points use partial transparency (alpha 0.4) to show density.
```

### Panel (d) — NSE Heatmap

```
Prompt:

Create a heatmap (color grid) showing NSE values stratified by flood magnitude (rows) and model type (columns).

Rows (4): "Small Floods (n=18)", "Medium Floods (n=15)", "Large Floods (n=14)", "Extreme Floods (n=10)".
Columns (3): "Tank", "HBV", "XAJ".

Each cell shows the mean validation NSE value as both a number and a color:
- Color scale: red (NSE=0.5) → yellow (NSE=0.7) → green (NSE=0.85) → dark green (NSE=0.95).

Sample values:
- Small: Tank=0.80, HBV=0.84, XAJ=0.78
- Medium: Tank=0.78, HBV=0.86, XAJ=0.82
- Large: Tank=0.76, HBV=0.87, XAJ=0.79
- Extreme: Tank=0.72, HBV=0.83, XAJ=0.68

Add bold border around the highest NSE cell per row.
Annotate: "HBV consistently best across all magnitude categories."

Title: "(d) NSE by Flood Magnitude × Model (Validation Set)".

Color bar legend on the right side.
```

### Panel (e) — Taylor Diagram

```
Prompt:

Create a Taylor diagram comparing all model and correction combinations.

A Taylor diagram is a polar plot showing correlation coefficient (angular axis), centered RMS difference (distance from observed point on x-axis), and standard deviation ratio (radial axis).

Place the reference point "Observed" at (1.0, 0) on the polar plot — standard deviation = 1.0, correlation = 1.0.

Plot symbols/labels for:
- "Tank raw" — blue circle, at (corr=0.90, std_ratio=0.85, centered RMS ~0.35)
- "Tank+XGB" — blue circle with "+" inside, at (corr=0.94, std_ratio=0.95, centered RMS ~0.20)
- "HBV raw" — green triangle, at (corr=0.92, std_ratio=0.88, centered RMS ~0.30)
- "HBV+XGB" — green triangle with "+", at (corr=0.96, std_ratio=0.97, centered RMS ~0.15)
- "XAJ raw" — red square, at (corr=0.88, std_ratio=0.80, centered RMS ~0.38)
- "XAJ+XGB" — red square with "+", at (corr=0.93, std_ratio=0.92, centered RMS ~0.22)
- "BMA Ensemble" — purple diamond, at (corr=0.97, std_ratio=0.98, centered RMS ~0.12)

Draw arcs for standard deviation ratios (0.25, 0.5, 0.75, 1.0, 1.25, 1.5) and concentric RMS circles.

Add a small annotation arrow from each raw model to its XGB-corrected counterpart.

Title: "(e) Taylor Diagram: Model Performance Before and After XGBoost Correction".

Bottom: "Figure 10. Case 2 — Large-Sample Calibration Results (57 Flood Events, 75%/25% Calibration/Validation Split)".

Academic journal composite figure style.
```
