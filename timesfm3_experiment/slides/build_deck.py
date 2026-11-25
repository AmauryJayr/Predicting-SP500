"""Build timesfm3_slides.pptx (python-pptx; plain text boxes, shapes and PNGs only). Run build_figs.py first."""
import pandas as pd
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree

HERE = Path(__file__).resolve().parent
FIG = HERE / "fig"
INK, MUTE, BLUE = RGBColor(0x1F, 0x1F, 0x1E), RGBColor(0x52, 0x51, 0x4E), RGBColor(0x2A, 0x78, 0xD6)
CARD = RGBColor(0xF3, 0xF4, 0xF6)
FONT = "Arial"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]

def rgb(h): return RGBColor.from_string(h)

def text(slide, x, y, w, h, paras, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         space_after=0, bullets=False, name=None):
    """paras: list of paragraphs; each is a str or a list of (text, {bold,color,size,italic}) runs."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name: tb.name = name
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, p_ in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if space_after: p.space_after = Pt(space_after)
        runs = [(p_, {})] if isinstance(p_, str) else p_
        for t, o in runs:
            r = p.add_run(); r.text = t
            f = r.font; f.name = FONT; f.size = Pt(o.get("size", size)); f.bold = o.get("bold", bold)
            f.italic = o.get("italic", False); f.color.rgb = o.get("color", color)
        if bullets:
            pPr = p._p.get_or_add_pPr(); pPr.set("marL", str(Inches(0.28))); pPr.set("indent", str(-Inches(0.28)))
            bc = etree.SubElement(pPr, qn("a:buClr")); sc = etree.SubElement(bc, qn("a:srgbClr")); sc.set("val", "2A78D6")
            bu = etree.SubElement(pPr, qn("a:buChar")); bu.set("char", "•")
    return tb

def shape(slide, kind, x, y, w, h, fill, line=None, radius=None, name=None):
    s = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    if name: s.name = name
    s.fill.solid(); s.fill.fore_color.rgb = fill
    if line is None: s.line.fill.background()
    else: s.line.color.rgb = line; s.line.width = Pt(1)
    st = s._element.find(qn('p:style'))
    if st is not None: s._element.remove(st)
    if radius is not None and kind == MSO_SHAPE.ROUNDED_RECTANGLE: s.adjustments[0] = radius
    return s

def shape_text(s, t, size, color=INK, bold=False, align=PP_ALIGN.CENTER):
    tf = s.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.08); tf.margin_top = tf.margin_bottom = Inches(0.03)
    p = tf.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = t; r.font.name = FONT; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color

def pic(slide, path, x, y, w, alt):
    p = slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w))
    p._element.nvPicPr.cNvPr.set("descr", alt)
    return p

n_slides = 0
def new_slide(title, notes, title_size=32):
    global n_slides
    n_slides += 1
    s = prs.slides.add_slide(BLANK)
    if title:
        text(s, 0.5, 0.35, 12.3, 0.7, [title], size=title_size, bold=True, name="Title")
    text(s, 12.2, 7.0, 0.65, 0.3, [str(n_slides)], size=12, color=MUTE, align=PP_ALIGN.RIGHT, name="Slide number")
    s.notes_slide.notes_text_frame.text = notes
    return s

# ---------------------------------------------------------------- 1 title
s = new_slide(None, "[~5 s] My part is TimesFM-3, a pretrained forecaster from Google. The question: can it call SPY's direction with no training on our data?")
text(s, 0.8, 2.35, 6.2, 1.0, ["TimesFM-3 on SPY"], size=44, bold=True, name="Title")
text(s, 0.8, 3.55, 5.9, 1.4, ["Can a pretrained forecaster call the market's direction?", "Zero-shot test, 2019\u20132025"], size=22, space_after=8, color=MUTE, name="Subtitle")
pic(s, FIG / "title_spy.png", 7.2, 2.2, 5.5, "SPY close 2019 to 2025 with the three big falls shaded violet")
text(s, 7.2, 5.45, 5.5, 0.3, ["SPY, 2019–2025; violet = the three big falls (2020, 2022, 2025)"], size=12, color=MUTE)

# ---------------------------------------------------------------- 2 how it was built
# Facts from the model card (~/.cache/huggingface/.../timesfm-3.0-pytorch/README.md) and report section 3.
from pptx.enum.shapes import MSO_CONNECTOR

def arrow(slide, x1, y1, x2, y2, color="8A8F98", w=1.75):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color); c.line.width = Pt(w)
    ln = c.line._get_or_add_ln(); te = etree.SubElement(ln, qn("a:tailEnd")); te.set("type", "triangle")
    return c

def card(slide, x, y, w, h, lines, fill=CARD, name=None):
    """lines: list of (text, size, bold, color) paragraphs inside a rounded card."""
    b = shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill, radius=0.12, name=name)
    tf = b.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.18); tf.margin_top = tf.margin_bottom = Inches(0.05)
    for i, (t, sz, bd, col) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        r = p.add_run(); r.text = t; f = r.font; f.name = FONT; f.size = Pt(sz); f.bold = bd; f.color.rgb = col
    return b

s = new_slide("How TimesFM-3 was built",
 "[~20 s] First, how TimesFM-3 was built. Google Research trained one 20-layer transformer, 331 million parameters, "
 "on over a trillion time points: a large public collection of time series, Wikipedia page views, Google Trends queries, and synthetic curves. "
 "It cuts the past into chunks of 32 points, reads them along time and across related series, and writes the next points as nine quantiles. "
 "It was not trained on our task: we use the released weights as they are. That is what zero-shot means.")
text(s, 0.5, 1.3, 3.6, 0.35, ["Pretraining data"], size=16, bold=True, color=MUTE, name="Data header")
data_cards = [("Public time-series collection", "GiftEvalPretrain: many public datasets"),
              ("Wikipedia page views", "up to Nov 2023"),
              ("Google Trends top queries", "up to the end of 2022"),
              ("Synthetic data", "plus augmented (transformed) series")]
targets = [2.95, 3.45, 3.95, 4.45]
for i, (t1, t2) in enumerate(data_cards):
    y = 1.75 + i * 1.08
    card(s, 0.5, y, 3.6, 0.9, [(t1, 16, True, INK), (t2, 13, False, MUTE)], name=f"Data {i+1}")
    arrow(s, 4.12, y + 0.45, 4.72, targets[i])
box = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 4.75, 2.55, 3.35, 2.3, BLUE, radius=0.1, name="TimesFM box")
tf = box.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
for i, (t, sz, bd) in enumerate([("TimesFM-3", 30, True), ("Google Research · Aug 2026", 15, False),
                                 ("20-layer transformer", 15, False), ("331M parameters", 15, False),
                                 ("1T+ time points", 15, False)]):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.alignment = PP_ALIGN.CENTER
    if i == 1: p.space_before = Pt(6)
    r = p.add_run(); r.text = t; f = r.font; f.name = FONT; f.size = Pt(sz); f.bold = bd; f.color.rgb = RGBColor(255, 255, 255)
arrow(s, 8.12, 3.7, 8.62, 3.7)
text(s, 8.65, 1.3, 4.2, 0.35, ["How it reads a series"], size=16, bold=True, color=MUTE, name="Steps header")
steps_b = [("1  Cut into chunks", "the past is split into patches of 32 points"),
           ("2  Read with attention", "20 layers look along time and across the related series"),
           ("3  Draw the future", "the next points come out as 9 quantiles (10%…90%)")]
for i, (t1, t2) in enumerate(steps_b):
    card(s, 8.65, 1.75 + i * 1.44, 4.2, 1.22, [(t1, 16, True, BLUE), (t2, 14, False, INK)], name=f"Step {i+1}")
text(s, 0.5, 6.25, 12.3, 0.4, [[("Not trained on our task: ", {"bold": True}),
                                ("we use Google's released weights as they are (", {}),
                                ("zero-shot", {"bold": True, "color": BLUE}), (").", {})]], size=17, name="Zero-shot line")
text(s, 0.5, 7.0, 9.0, 0.3, ["Source: model card google/timesfm-3.0-pytorch (data, layers, patch size)"], size=11, color=MUTE, name="Source")

# ---------------------------------------------------------------- 3 how it works
s = new_slide("How TimesFM-3 works",
 "[~30 s] Each day we give it the last 256 trading days of SPY plus 15 related markets. "
 "For each of the next 5 days it returns nine quantiles, like this table with made-up numbers. "
 "Take the day-5 row: there is a 10% chance the close is below 585, a 50% chance below 601, and a 90% chance below 616. "
 "Today's close, 600, sits between the 40% and 50% values, three quarters of the way up. "
 "So the chance of ending lower is about 47.5%, and P(up) is about 52.5%. "
 "Our results add the team's drift layer here, before the threshold: stacking for tomorrow, recalibration for 5 days, chosen on 2015 to 2018. Raw, 52.5% is below 58%, so 'not up'.")
text(s, 0.5, 1.45, 5.2, 4.2, [
    "Input: last 256 trading days of SPY + 15 related series (QQQ, IWM, 9 sector ETFs, TLT, IEF, GLD, VIX)",
    "Output: for each of the next 5 days, 9 quantiles (10%…90%) of the price"],
    size=20, space_after=20, bullets=True, name="Bullets")

# quantile table: the made-up example of the report ("How TimesFM-3 makes a forecast")
def cell_border(cell, color="D0D0D0", w=12700):
    tcPr = cell._tc.get_or_add_tcPr()  # borders must precede the fill in tcPr, so call this before cell.fill
    for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        ln = etree.SubElement(tcPr, qn(tag)); ln.set("w", str(w))
        sf = etree.SubElement(ln, qn("a:solidFill")); c_ = etree.SubElement(sf, qn("a:srgbClr")); c_.set("val", color)

text(s, 6.0, 1.4, 6.85, 0.35, [[("Example (made-up numbers): ", {"bold": True}), ("today's close = 600", {})]], size=16, name="Table caption")
q_hdr = ["day"] + [f"{q}%" for q in range(10, 100, 10)]
q_rows = [["t+1", "594", "596", "598", "599", "600.5", "601.5", "603", "604", "606"],
          ["t+2…t+4"] + ["…"] * 9,
          ["t+5", "585", "590", "594", "597", "601", "604", "607", "611", "616"]]
gf = s.shapes.add_table(4, 10, Inches(6.0), Inches(1.85), Inches(6.85), Inches(1.72))
gf.name = "Quantile table"
tbl = gf.table
tbl.first_row = True; tbl.horz_banding = False
tbl.columns[0].width = Inches(1.09)
for j in range(1, 10): tbl.columns[j].width = Inches(0.64)
HI = rgb("FBE3A6")
for i, vals in enumerate([q_hdr] + q_rows):
    tbl.rows[i].height = Inches(0.43)
    for j, v in enumerate(vals):
        c = tbl.cell(i, j)
        hi = (i == 3 and j in (4, 5))  # t+5 row, 40% and 50%: today's 600 lies between them
        cell_border(c)
        c.fill.solid(); c.fill.fore_color.rgb = HI if hi else (CARD if i == 0 or j == 0 else RGBColor(255, 255, 255))
        c.margin_left = c.margin_right = Inches(0.03); c.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = v
        f = r.font; f.name = FONT; f.size = Pt(15); f.bold = (i == 0 or j == 0 or hi); f.color.rgb = INK
text(s, 6.0, 3.75, 6.85, 2.05, [
    [("Read the t+5 row: ", {"bold": True}), ("a 10% chance the close is below 585, 50% below 601, 90% below 616.", {})],
    [("Today's 600 ", {"bold": True}), ("sits between 597 (40%) and 601 (50%), ¾ of the way up → about 47.5% chance of ending lower.", {})],
    [("P(up) ≈ 52.5%", {"bold": True, "color": BLUE}), (" < 58% threshold → “not up” (raw, before the drift layer).", {})]],
    size=16, space_after=10, name="Table reading")
steps = ["past 256 days", "TimesFM-3", "9 quantiles", "P(up)", "drift layer", "≥ threshold?", "“up” / “not up”"]
bw, gap, x0, y0 = 1.5, 0.27, 0.5, 5.85
for i, t in enumerate(steps):
    x = x0 + i * (bw + gap)
    last = i == len(steps) - 1
    b = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, y0, bw, 0.65, BLUE if i == 1 else rgb("FBE3A6") if i == 4 else CARD, radius=0.2, name=f"Flow {i+1}")
    shape_text(b, t, 14, color=RGBColor(255, 255, 255) if i == 1 else INK, bold=(i in (1, 4)))
    if not last:
        text(s, x + bw, y0, gap, 0.65, ["→"], size=16, color=MUTE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
text(s, 0.5, 6.7, 11.5, 0.55, [[("Drift layer: ", {"bold": True}), ("the team's methods on top of TimesFM's P(up): stacking for tomorrow, recalibration for 5 days, picked on 2015–2018. All results after this slide include it.", {})]],
     size=14, name="Drift layer line")

# ---------------------------------------------------------------- 3 result
s = new_slide("Result: tomorrow just above a coin flip, 5 days below",
 "[~20 s] We locked the setup on 2013 to 2018 and ran the 2019 to 2025 test once. With the drift layer, TimesFM scores AUC 0.52 for tomorrow, just above a coin flip and our regression's 0.51, but not significantly. "
 "For 5 days it scores 0.48, below a coin flip and clearly below the regression's 0.54. The weak years are the big-fall years: 2020, 2022, 2025. Year by year.")
text(s, 0.5, 1.3, 6.0, 0.35, ["AUC by horizon"], size=18, bold=True)
text(s, 6.85, 1.3, 6.0, 0.35, ["5-day AUC per test year"], size=18, bold=True)
pic(s, FIG / "auc_horizon.png", 0.5, 1.75, 6.0, "Bar chart: AUC for TimesFM-3 with the drift layer and logistic regression at h=1 and h=5")
pic(s, FIG / "auc_by_year.png", 6.85, 1.75, 6.0, "Bar chart: h=5 AUC per test year with the drift layer, 2020, 2022 and 2025 highlighted")
text(s, 0.5, 6.0, 12.3, 0.35, [[("Drift layer: ", {"bold": True, "color": BLUE}), ("stacking (h=1), recalibration (h=5), picked on 2015–2018. Test run once (1,755–1,759 days).", {})]], size=16, color=INK, name="Caption")
text(s, 0.5, 6.42, 11.4, 0.8, ["AUC 0.5 = coin flip, 1.0 = always right. Gain over static TimesFM: h=1 +0.043 [−0.003, +0.090], h=5 +0.022 [+0.001, +0.046]; h=1 vs the fixed LR +0.017; "
                               "none significant after Holm correction. Each family's settings were pre-registered; stacking for h=1 and recalibration for h=5 were picked "
                               "on 2015–2018 numbers only, but that rule was set after the test had been run."], size=11, color=MUTE, name="Footnote")

# ---------------------------------------------------------------- 5-11 years (drift-applied TimesFM: h=1 stack-4, h=5 platt-6)
det = pd.read_csv(HERE.parent / "results" / "timesfm_drift_year_detail.csv"); det = det[det.year != "all"]; det["year"] = det.year.astype(int)
def row(y, h): return det[(det.year == y) & (det.horizon == h)].iloc[0]
RED, AMBER, GREY = (rgb("FBE3E1"), rgb("B42318")), (rgb("FDEBC8"), rgb("8A5A00")), (rgb("EDEDEB"), rgb("3F3F3D"))
red_found = {(y, h) for y in range(2019, 2026) for h in (1, 5) if row(y, h).auc_ci_hi < 0.5}
amber_found = {(y, h) for y in range(2019, 2026) for h in (1, 5) if row(y, h).auc < 0.45 and (y, h) not in red_found}
assert red_found == {(2020, 5), (2022, 5)}, red_found
assert amber_found == {(2025, 5)}, amber_found

Q = "“"; Qe = "”"
RAW = "Raw TimesFM's causes: see the root-cause slide."
YEARS = [
 (2019, "A calm climb", "[~10 s] 2019 was a calm climb, plus 31 percent. TimesFM said up on 38 percent of days. The 5-day AUC is 0.48, a coin flip; tomorrow's 0.57 has an interval that includes 0.5.",
  "+31%, one small dip (−7% in May).",
  f"Said {Q}up{Qe} on 38% of days, while 68% of 5-day windows went up.",
  "Nothing real to find in a calm year; the 95% interval (0.35–0.64) includes 0.5, so this is a coin flip."),
 (2020, "Covid crash and rebound", "[~12 s] 2020, the Covid crash. TimesFM still said up every day of the crash, which was wrong, and in the rebound up on only 41 percent of days, mostly the wrong ones. AUC 0.36, the whole interval below one half. "
  "The bounce habit scores 0.54; the rest scores 0.34.",
  "−34% in Feb–Mar, back at the high in August.",
  f"Said {Q}up{Qe} every day of the crash (right on 30%). In the rebound: {Q}up{Qe} on 41% of days, right on 40%; always {Q}up{Qe} gets 71%.",
  f"Not the bounce habit: that part alone scores 0.54. The rest of the forecast scores 0.34. {RAW}"),
 (2021, "A smooth climb", "[~8 s] 2021, a smooth climb; the biggest dip was only 5 percent. With the drift layer TimesFM said up on 68 percent of days instead of 90. AUC 0.49, a coin flip.",
  "+29%, biggest dip only −5%.",
  f"Said {Q}up{Qe} on 68% of days (raw TimesFM: 90%), while 63% of windows went up.",
  "Little to rank; the interval (0.36–0.59) includes 0.5, a coin flip."),
 (2022, "Rate hikes: three falls, three bounces", "[~12 s] 2022, rate hikes: three falls with sharp bounces. In the bounces the market went up in 71 percent of the windows, but TimesFM said up on only 1 percent of those days. AUC 0.38, the whole interval below one half. Again not the bounce habit, but the rest of the forecast.",
  "−18% for the year (−24% from the high), falling in 3 steps with sharp bounces.",
  f"In the 3 bounces the market rose in 71% of windows, but TimesFM said {Q}up{Qe} on only 1% of those days; its P(up) was higher in the falls (0.50) than in the bounces (0.33).",
  f"Not the bounce habit (that part scores 0.51); the rest of the forecast (0.37) pushed it down in the bounces. {RAW}"),
 (2023, "A calm climb with a 10% dip", "[~8 s] 2023, a calm climb with a 10 percent dip. The drift layer lifts the up calls from 29 to 61 percent of days, but the ranking is a coin flip: AUC 0.47.",
  "+26%, led by AI and big tech; −10% from August to October.",
  f"Said {Q}up{Qe} on 61% of days (raw TimesFM: 29%), close to the 62% of windows that went up.",
  "Its calls now match the market's up rate, but the ranking is still a coin flip (interval 0.30–0.68)."),
 (2024, "The best year for tomorrow", "[~12 s] 2024 flips. Raw TimesFM had its worst tomorrow-year here, 0.42. With the stacking layer it is the best, 0.60, interval above one half. "
  "The stack leans on the forecast spread. We did not check why 2024; it is one year of seven, and no gain is significant.",
  "+25%, one sharp drop (−8%) in July–August.",
  f"For tomorrow (h=1) this is the best year: AUC 0.60, whole interval (0.56–0.68) above 0.5. Raw TimesFM scored 0.42 here. The 5-day AUC is a coin flip (0.54).",
  "Not explained. The h=1 stack gets its AUC from the q90−q10 spread, not the up/down score. One year of seven; no gain is significant after correction."),
 (2025, "Tariff shock and recovery", "[~13 s] 2025, the tariff shock. TimesFM said up on most fall days, then in the recovery on only 7 percent of days, while the market rose in 76 percent of the windows. "
  "For the raw score we found part of the reason: a built-in step draws a straight line through the past year and continues it. The drift layer does not fix that.",
  "−19% from February to April, back at the high in late June.",
  f"Said {Q}up{Qe} on 76% of fall days (right on 56%). In the recovery the market rose in 76% of windows, but TimesFM said {Q}up{Qe} on only 7% of days.",
  "For raw TimesFM, partly checked: a built-in step draws a straight line through the past year and continues it; after the fall it pointed down and pulled the forecast down (see the root-cause slide). The drift layer does not fix it."),
]
for yr, sub, notes, spy_t, tf_t, why_t in YEARS:
    s = new_slide(f"{yr} · {sub}", notes)
    x = 0.5
    for h in (5, 1):
        r = row(yr, h)
        fill, fg = RED if (yr, h) in red_found else AMBER if (yr, h) in amber_found else GREY
        b = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, 1.12, 2.35, 0.46, fill, radius=0.3, name=f"Badge h={h}")
        shape_text(b, f"h={h} AUC {r.auc:.2f}", 18, color=fg, bold=True)
        x += 2.55
    pic(s, FIG / f"year_{yr}.png", 0.4, 1.8, 8.2, f"{yr}: SPY close with fall and recovery phases, TimesFM + drift layer up calls and market up days")
    lab = lambda t: [(t, {"bold": True, "color": BLUE, "size": 15})]
    text(s, 8.95, 1.85, 3.95, 5.0, [lab("SPY"), spy_t, lab("TimesFM + drift"), tf_t, lab("Why"), why_t], size=15, space_after=3, name="Explanation")
    # extra gap before labels: set space_before on label paragraphs
    for p in s.shapes[-1].text_frame.paragraphs:
        if p.runs and p.runs[0].font.bold and p.runs[0].text in ("TimesFM + drift", "Why"): p.space_before = Pt(9)
    text(s, 0.5, 6.9, 11.4, 0.45, ["TimesFM = with the drift layer (stacking h=1, recalibration h=5). Red = whole 95% interval below 0.5; amber = AUC below 0.45 but the interval reaches 0.5. "
                                   "Figure and phases: 5-day windows. Year analysis is exploratory (after the test)."], size=11, color=MUTE, name="Footnote")

# ---------------------------------------------------------------- 12 root cause (TIMESFM_ROOT_CAUSE.md)
def fig_slide(title, notes, png, alt, bullets, answer, foot, sub=None, title_size=32, fig_w=7.45, tx=8.2, bsize=16, bgap=12, bw=None):
    s = new_slide(title, notes, title_size=title_size)
    if sub: text(s, 0.5, 0.93, 12.3, 0.35, [sub], size=17, color=MUTE, name="Subtitle")
    pic(s, FIG / png, 0.45, 1.45, fig_w, alt)
    text(s, tx, 1.55, bw or 4.65, 4.4, bullets, size=bsize, space_after=bgap, bullets=True, name="Bullets")
    text(s, 0.5, 6.1, 12.3, 0.45, [answer], size=18, name="Answer")
    text(s, 0.5, 6.97, 11.2, 0.3, [foot], size=11, color=MUTE, name="Footnote")
    return s

fig_slide("Is the transformer the root cause?",
 "[~18 s] Is the transformer the problem? This is the raw TimesFM forecast, under the drift layer. We split every forecast into the transformer's part and the built-in trend line. "
 "Neither part is significantly below one half. The weak years, 2020 and 2022, sit in the transformer's output, "
 "but simple models like a random walk with drift score just as low there. In 2025 the trend line is the weaker part. "
 "So we find no transformer defect: simple extrapolators fail in the same years.",
 "root_cause.png", "Dot plot: h=5 AUC with 95% intervals for TimesFM, its transformer part, its trend-line part and three simple forecasters",
 ["We split each forecast into the transformer's output and the built-in trend line: h=5 AUC 0.480 and 0.470 (official 0.459). Neither is significantly below 0.5.",
  "2020 and 2022 are low in the transformer's output, but simple non-transformer forecasters (random walk + drift, AR(5)) are just as low there.",
  "2025: the trend line is the weaker part (0.38)."],
 [("Answer: ", {"bold": True}), ("no transformer defect found; simple extrapolators fail in the same years.", {"bold": True, "color": BLUE})],
 "Exploratory (after the test); bars = 95% block-bootstrap intervals; details in TIMESFM_ROOT_CAUSE.md",
 sub="Analysis of the raw TimesFM score (static, before the drift layer)")

# ---------------------------------------------------------------- 13 TimesFM vs Chronos-2 (TIMESFM_VS_CHRONOS.md)
fig_slide("TimesFM-3 vs Chronos-2: opposite bets, same result",
 "[~20 s] How does TimesFM compare with the team's Chronos-2? On the same test days, raw scores, the AUC difference is not real: tomorrow 0.482 against 0.477, five days 0.459 against 0.479, and both intervals include zero. "
 "But the two make unrelated daily calls; they agree on about half of the days, like two coins. TimesFM leans on turns and ignores volatility. Chronos follows the trend at five days, and says up less often when volatility is high. "
 "TimesFM's drift layer does not lift Chronos. So: no real difference in skill, different bets, both near a coin flip.",
 "timesfm_vs_chronos.png", "Grouped horizontal bars: Spearman correlation of the TimesFM-3 and Chronos-2 daily scores with the trailing 1-day return, trailing 5-day return, 20-day realized volatility and the model's own forecast band width, at h=1 and h=5",
 ["Test AUC (h=1 / h=5): TimesFM 0.482 / 0.459, Chronos-2 0.477 / 0.479. Difference +0.005 [\u22120.033, +0.043] and \u22120.019 [\u22120.087, +0.044].",
  "Their daily scores are unrelated: Spearman \u22120.003 / \u22120.048; same up/down call on 47.6% / 50.1% of days.",
  "TimesFM leans on turns (5-day: \u22120.23 with the trailing 5-day return) and ignores volatility (about 0). Chronos follows the trend at h=5 (+0.60) and gives a lower P(up) when volatility is high (\u22120.61 at h=1, \u22120.36 at h=5).",
  "TimesFM's drift layer does not lift Chronos: h=1 \u22120.009, h=5 \u22120.0003."],
 [("Answer: ", {"bold": True}), ("no real difference in skill: they bet on different things and both land near a coin flip.", {"bold": True, "color": BLUE})],
 "Exploratory, computed after the test. A controlled swap (same input for both models) is running and is not on this slide yet. Details in TIMESFM_VS_CHRONOS.md",
 sub="Raw scores, no drift layer \u00b7 Chronos-2 from the team's chronos branch", title_size=30, bsize=15, bgap=9)

# ---------------------------------------------------------------- 14 drift methods (TIMESFM_DRIFT.md)
fig_slide("The drift layer: four methods compared",
 "[~10 s] The detail behind the drift layer: four layers on the frozen score, settings chosen on 2015 to 2018, test run once. "
 "Stacking at one day and recalibration at five days had the biggest 2015 to 2018 gain, so they are in this talk. No gain is significant, and at five days the logistic regression is clearly better.",
 "drift_methods.png", "Bar chart: test AUC at h=1 and h=5 for static TimesFM, four drift methods and the fixed logistic regression",
 ["Four layers on the frozen TimesFM score: recalibration, stacking, Hedge, drift detector.",
  "Settings chosen on 2015–2018 and locked; test 2019–2025 run once.",
  "Used in this talk: stacking at h=1 (0.482 → 0.524) and recalibration at h=5 (0.459 → 0.481), the largest 2015–2018 gains.",
  "Not significant after correction. h=5 stays below 0.5 and the fixed LR (0.536) stays better; at h=1 the stack is +0.017 above the LR."],
 [("Result: ", {"bold": True}), ("gains over static TimesFM are at most 0.04 and not significant.", {"bold": True, "color": BLUE})],
 "Pre-registered per family; the cross-method pick used 2015–2018 numbers but its rule was set after the test; details in TIMESFM_DRIFT.md")

# ---------------------------------------------------------------- 15 why the drift layer helped (TIMESFM_DRIFT.md section 7)
fig_slide("Why did the drift layer help?",
 "[~22 s] Why did the drift layer raise AUC? Tomorrow: the layer puts almost no weight on TimesFM's up or down call. It uses the width of TimesFM's forecast band, and that width is mostly a volatility gauge, tracking the VIX at 0.80. "
 "A wider band gives a slightly higher P(up). Five days: inside each 21-day block the ranking does not change; the whole gain comes from shifting entire periods, mostly 2022, the bear year. Without 2022 the gain is gone. "
 "So the layer did not fix the direction call.",
 "drift_why.png", "Left: bar chart of test AUC at h=1 for the TimesFM score, forecast median, forecast band width, VIX, 20-day realized vol and the stack-4 layer, against the 0.5 coin-flip line. Right: h=5 AUC gain with all years (+0.022) and without 2022 (-0.006)",
 ["Tomorrow (h=1, 0.482 \u2192 0.524): the layer puts almost no weight on TimesFM's direction call (\u22120.003) and bets on the width of its forecast band (q90\u2212q10, +0.046): a wider band \u2192 slightly higher P(up).",
  "The band width is mostly a volatility gauge: Spearman 0.80 with the VIX, 0.86 with 20-day realized vol. Stack-4 minus an LR on the VIX: +0.018 [\u22120.007, +0.042], not significant.",
  "5 days (h=5, 0.459 \u2192 0.481): inside each 21-day refit block the ranking is unchanged (0.522 for both); the gain comes from shifting whole periods. In 2022, the bear year, the offset pulled all probabilities down (mean \u22120.51); without 2022 the gain is \u22120.006.",
  "A version with no labels at all (TimesFM's score minus its own trailing 1-year mean) gets the same 0.481."],
 [("Answer: ", {"bold": True}), ("it did not fix the direction call: for tomorrow it reads TimesFM's uncertainty as a volatility gauge; for 5 days the gain is the 2022 bear year.", {"bold": True, "color": BLUE})],
 "Exploratory, computed after the test. Dashed line = 0.5 (coin flip). Details in TIMESFM_DRIFT.md, section 7",
 sub="TimesFM + drift layer: h=1 stack-4, h=5 recalibration (platt-6)", fig_w=6.8, tx=7.45, bw=5.4, bsize=14, bgap=7)

# ---------------------------------------------------------------- 16 VIX on top of the drift layer (TIMESFM_VIX.md, follow-up section)
fig_slide("Does the VIX help TimesFM?",
 "[~15 s] Does the VIX help on top of the drift layer? Same baseline as the rest of this talk. We refit the layer when the VIX regime flips, picked the model by VIX regime, "
 "added VIX terms to the layer, and removed the VIX from TimesFM's inputs. No variant is significantly better than the baseline, and removing the VIX changes nothing. "
 "This follow-up came after the earlier results, so it is exploratory.",
 "vix_drift_methods.png", "Bar chart: test AUC at h=1 and h=5 for TimesFM with the drift layer, three VIX variants of the layer, the layer without the VIX input and the fixed logistic regression",
 ["Four changes to the drift layer: refit when the VIX regime flips, pick the model by VIX regime, add VIX terms, or remove the VIX from TimesFM's inputs.",
  "Baseline AUC: 0.524 (h=1) and 0.481 (h=5).",
  "With the VIX: h=1 0.463–0.521 and h=5 0.474–0.486. None is significantly better than the baseline.",
  "VIX removed from the input: 0.530 (h=1) and 0.479 (h=5). The VIX in the input does not help."],
 [("Answer: ", {"bold": True}), ("no. No VIX version is significantly better than the drift layer alone.", {"bold": True, "color": BLUE})],
 "Exploratory: asked after the TimesFM, drift and VIX test results were known; the drift layer was picked by a rule set after its test. Details in TIMESFM_VIX.md",
 sub="Baseline: TimesFM + drift layer")

# ---------------------------------------------------------------- 17 bottom line
s = new_slide("Bottom line",
 "[~8 s] So: with our drift layer TimesFM is just above a coin flip for tomorrow and below it for five days, no better than the logistic regression. The weak years are still the fall years. We explained part of 2025; 2020 and 2022 stay open. "
 "Two exploratory follow-ups: against Chronos-2 there is no real difference, both near a coin flip with unrelated bets; and the drift layer works through volatility at one day and through the 2022 bear year at five days, not through a better direction call.")
items = ["TimesFM-3 + drift layer: AUC 0.52 (h=1) and 0.48 (h=5). Not significantly above our logistic regression at h=1 (0.51), clearly below it at h=5 (0.54).",
         "Calm years: a coin flip. The weak 5-day years are still the fall years: 2020 (0.36), 2022 (0.38), 2025 (0.40).",
         "Explained: part of raw TimesFM's 2025 (the trend line). 2020 and 2022 stay open. The h=1 stack's AUC comes from the forecast spread, not the up/down score."]
for i, t in enumerate(items):
    y = 1.6 + i * 1.7
    shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 0.5, y, 12.3, 1.4, CARD, radius=0.12, name=f"Card {i+1}")
    c = shape(s, MSO_SHAPE.OVAL, 0.85, y + 0.37, 0.66, 0.66, BLUE, name=f"Number {i+1}")
    shape_text(c, str(i + 1), 22, color=RGBColor(255, 255, 255), bold=True)
    text(s, 1.85, y + 0.1, 10.6, 1.2, [t], size=20, anchor=MSO_ANCHOR.MIDDLE, name=f"Bullet {i+1}")

text(s, 0.5, 6.62, 11.5, 0.6, [[("Also (exploratory): ", {"bold": True, "color": BLUE}),
     ("Chronos-2 is no different in skill (unrelated bets, both near a coin flip); the drift layer works through volatility (h=1) and the 2022 bear year (h=5).", {})]], size=16, name="Extra line")

prs.save(HERE / "timesfm3_slides.pptx")
print("saved", n_slides, "slides")
