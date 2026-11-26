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
s = new_slide(None, "[~5 s] My part is TimesFM-3. It is a forecasting model from Google that is already trained. The question is simple: can it tell us if the S&P 500 goes up or down, with no training on our data?")
text(s, 0.8, 2.35, 6.2, 1.0, ["TimesFM-3 on SPY"], size=44, bold=True, name="Title")
text(s, 0.8, 3.55, 5.9, 1.4, ["Can a model that is already trained call the market's direction?", "Used as is, no training on our data \u00b7 tested on 2019\u20132025"], size=22, space_after=8, color=MUTE, name="Subtitle")
pic(s, FIG / "title_spy.png", 7.2, 2.2, 5.5, "SPY close 2019 to 2025 with the three big falls shaded violet")
text(s, 7.2, 5.45, 5.5, 0.3, ["SPY (the fund that tracks the S&P 500), 2019–2025; violet = the three big falls (2020, 2022, 2025)"], size=12, color=MUTE)

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
 "[~20 s] First, how TimesFM-3 was built. Google trained one big neural network, with 20 layers, on over a trillion data points. "
 "The data are public collections of time series, Wikipedia page views, Google search trends, and made-up curves. "
 "The network cuts the past into pieces of 32 points and looks for patterns over time and between related series. "
 "Then it writes out the future as nine possible prices, from low to high. "
 "Important: it was not trained on our data. We use Google's model as it is. People call this zero-shot.")
text(s, 0.5, 1.3, 3.6, 0.35, ["Pretraining data"], size=16, bold=True, color=MUTE, name="Data header")
data_cards = [("Public time-series collection", "GiftEvalPretrain: many public datasets"),
              ("Wikipedia page views", "up to Nov 2023"),
              ("Google Trends top queries", "up to the end of 2022"),
              ("Made-up (synthetic) data", "plus altered copies of real series")]
targets = [2.95, 3.45, 3.95, 4.45]
for i, (t1, t2) in enumerate(data_cards):
    y = 1.75 + i * 1.08
    card(s, 0.5, y, 3.6, 0.9, [(t1, 16, True, INK), (t2, 13, False, MUTE)], name=f"Data {i+1}")
    arrow(s, 4.12, y + 0.45, 4.72, targets[i])
box = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 4.75, 2.55, 3.35, 2.3, BLUE, radius=0.1, name="TimesFM box")
tf = box.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
for i, (t, sz, bd) in enumerate([("TimesFM-3", 30, True), ("Google Research · Aug 2026", 15, False),
                                 ("Neural network, 20 layers", 15, False), ("331 million adjustable numbers", 15, False),
                                 ("Over 1 trillion data points", 15, False)]):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.alignment = PP_ALIGN.CENTER
    if i == 1: p.space_before = Pt(6)
    r = p.add_run(); r.text = t; f = r.font; f.name = FONT; f.size = Pt(sz); f.bold = bd; f.color.rgb = RGBColor(255, 255, 255)
arrow(s, 8.12, 3.7, 8.62, 3.7)
text(s, 8.65, 1.3, 4.2, 0.35, ["How it reads a series"], size=16, bold=True, color=MUTE, name="Steps header")
steps_b = [("1  Cut into pieces", "the past is split into pieces of 32 points"),
           ("2  Look for patterns", "20 layers look at how a series moves over time and how related series move together"),
           ("3  Write out the future", "the next points come out as 9 possible prices, from low (10%) to high (90%)")]
for i, (t1, t2) in enumerate(steps_b):
    card(s, 8.65, 1.75 + i * 1.44, 4.2, 1.22, [(t1, 16, True, BLUE), (t2, 14, False, INK)], name=f"Step {i+1}")
text(s, 0.5, 6.25, 12.3, 0.4, [[("Not trained on our data: ", {"bold": True}),
                                ("we use Google's released model as it is (technical word: ", {}),
                                ("zero-shot", {"bold": True, "color": BLUE}), (").", {})]], size=17, name="Zero-shot line")
text(s, 0.5, 7.0, 9.0, 0.3, ["Source: Google's model card, google/timesfm-3.0-pytorch (data, layers, piece size)"], size=11, color=MUTE, name="Source")

# ---------------------------------------------------------------- 3 how it works
s = new_slide("How TimesFM-3 works",
 "[~30 s] Each day we give the model the last 256 trading days of SPY, plus 15 related markets. "
 "For each of the next 5 days it gives nine possible prices, from low to high. Here is a table with made-up numbers. "
 "Look at the day-5 row. There is a 10% chance the price ends below 585, a 50% chance below 601, and a 90% chance below 616. "
 "Today's price is 600. It sits between the 40% price, 597, and the 50% price, 601, about three quarters of the way up. "
 "So the chance of ending lower is about 47.5%, and the chance of up is about 52.5%. "
 "Our rule calls 'up' only if that chance is at least 58%. Here it is lower, so the call is 'not up'. "
 "In our results there is one more step, the drift layer. It is a small correction step on top of TimesFM that adapts to recent data. We chose it using 2015 to 2018 only.")
text(s, 0.5, 1.45, 5.2, 4.2, [
    "Input: the last 256 trading days of SPY plus 15 related markets (other stock funds, 9 sector funds, bonds, gold, and the VIX, the market's “fear index”)",
    "Output: for each of the next 5 days, 9 possible prices, from low (10%) to high (90%)"],
    size=20, space_after=20, bullets=True, name="Bullets")

# quantile table: the made-up example of the report ("How TimesFM-3 makes a forecast")
def cell_border(cell, color="D0D0D0", w=12700):
    tcPr = cell._tc.get_or_add_tcPr()  # borders must precede the fill in tcPr, so call this before cell.fill
    for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        ln = etree.SubElement(tcPr, qn(tag)); ln.set("w", str(w))
        sf = etree.SubElement(ln, qn("a:solidFill")); c_ = etree.SubElement(sf, qn("a:srgbClr")); c_.set("val", color)

text(s, 6.0, 1.4, 6.85, 0.35, [[("Example (made-up numbers): ", {"bold": True}), ("today's close = 600", {})]], size=16, name="Table caption")
q_hdr = ["when"] + [f"{q}%" for q in range(10, 100, 10)]
q_rows = [["day 1", "594", "596", "598", "599", "600.5", "601.5", "603", "604", "606"],
          ["days 2–4"] + ["…"] * 9,
          ["day 5", "585", "590", "594", "597", "601", "604", "607", "611", "616"]]
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
        hi = (i == 3 and j in (4, 5))  # day-5 row, 40% and 50%: today's 600 lies between them
        cell_border(c)
        c.fill.solid(); c.fill.fore_color.rgb = HI if hi else (CARD if i == 0 or j == 0 else RGBColor(255, 255, 255))
        c.margin_left = c.margin_right = Inches(0.03); c.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = v
        f = r.font; f.name = FONT; f.size = Pt(15); f.bold = (i == 0 or j == 0 or hi); f.color.rgb = INK
text(s, 6.0, 3.75, 6.85, 2.05, [
    [("Read the day-5 row: ", {"bold": True}), ("a 10% chance the close is below 585, 50% below 601, 90% below 616.", {})],
    [("Today's 600 ", {"bold": True}), ("sits between 597 (40%) and 601 (50%), three quarters of the way up → about 47.5% chance of ending lower.", {})],
    [("Chance of up ≈ 52.5%", {"bold": True, "color": BLUE}), (", below our 58% cut-off → “not up” (before the drift layer).", {})]],
    size=16, space_after=10, name="Table reading")
steps = ["last 256 days", "TimesFM-3", "9 possible prices", "chance of up", "drift layer", "above cut-off?", "“up” / “not up”"]
bw, gap, x0, y0 = 1.5, 0.27, 0.5, 5.85
for i, t in enumerate(steps):
    x = x0 + i * (bw + gap)
    last = i == len(steps) - 1
    b = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, y0, bw, 0.65, BLUE if i == 1 else rgb("FBE3A6") if i == 4 else CARD, radius=0.2, name=f"Flow {i+1}")
    shape_text(b, t, 14, color=RGBColor(255, 255, 255) if i == 1 else INK, bold=(i in (1, 4)))
    if not last:
        text(s, x + bw, y0, gap, 0.65, ["→"], size=16, color=MUTE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
text(s, 0.5, 6.7, 11.5, 0.55, [[("Drift layer: ", {"bold": True}), ("a small correction step on top of TimesFM that adapts to recent data. For tomorrow: a small model on top that uses TimesFM's outputs. For the next 5 days: a monthly up/down shift of all its chances. Chosen on 2015–2018; all results after this slide include it.", {})]],
     size=14, name="Drift layer line")

# ---------------------------------------------------------------- 4 result
s = new_slide("Result: tomorrow just above a coin flip, next 5 days below",
 "[~25 s] The result. We fixed all settings using data up to 2018. Then we ran the 2019 to 2025 test once. "
 "One measure first, called AUC. Pick one day that went up and one day that went down. AUC is how often the model gave the up day the higher chance. 0.5 is a coin flip. 1.0 is always right. "
 "For tomorrow, TimesFM with the drift layer scores 0.52. That is just above a coin flip, and about the same as our simple model, 0.51. The gap could be luck. "
 "For the next 5 days it scores 0.48. That is below a coin flip, and worse than our simple model, 0.54. "
 "The weak years are the big-fall years: 2020, 2022 and 2025. Now, year by year.")
text(s, 0.5, 1.3, 6.0, 0.35, ["AUC: tomorrow and next 5 days"], size=18, bold=True)
text(s, 6.85, 1.3, 6.0, 0.35, ["AUC for the next 5 days, per test year"], size=18, bold=True)
pic(s, FIG / "auc_horizon.png", 0.6, 1.75, 5.6, "Bar chart: AUC for TimesFM-3 with the drift layer and our simple model, for tomorrow and the next 5 days")
pic(s, FIG / "auc_by_year.png", 7.0, 1.75, 5.6, "Bar chart: AUC for the next 5 days per test year with the drift layer, 2020, 2022 and 2025 highlighted")
text(s, 0.5, 5.72, 12.3, 0.65, [[("AUC: ", {"bold": True, "color": BLUE}), ("pick one up day and one down day; how often does the model rank the up day higher? 0.5 = coin flip, 1.0 = always right.", {})]], size=16, color=INK, name="AUC definition")
text(s, 0.5, 6.34, 12.3, 0.35, [[("Our simple model: ", {"bold": True, "color": BLUE}), ("a logistic regression trained once. ", {}), ("Drift layer: ", {"bold": True, "color": BLUE}), ("see the previous slide.", {})]], size=16, color=INK, name="Caption")
text(s, 0.5, 6.78, 11.4, 0.55, ["Gains over TimesFM alone (+0.04 tomorrow, +0.02 next 5 days) could be luck; tomorrow's 0.52 vs 0.51 is not clearly better. Test run once on 1,755–1,759 days. "
                               "Settings fixed before the test; the layer for each horizon was picked on 2015–2018 numbers only, but that rule was set after the test."], size=11, color=MUTE, name="Footnote")

# ---------------------------------------------------------------- 5-11 years (drift-applied TimesFM: h=1 stack-4, h=5 platt-6)
det = pd.read_csv(HERE.parent / "results" / "timesfm_drift_year_detail.csv"); det = det[det.year != "all"]; det["year"] = det.year.astype(int)
def row(y, h): return det[(det.year == y) & (det.horizon == h)].iloc[0]
RED, AMBER, GREY = (rgb("FBE3E1"), rgb("B42318")), (rgb("FDEBC8"), rgb("8A5A00")), (rgb("EDEDEB"), rgb("3F3F3D"))
red_found = {(y, h) for y in range(2019, 2026) for h in (1, 5) if row(y, h).auc_ci_hi < 0.5}
amber_found = {(y, h) for y in range(2019, 2026) for h in (1, 5) if row(y, h).auc < 0.45 and (y, h) not in red_found}
assert red_found == {(2020, 5), (2022, 5)}, red_found
assert amber_found == {(2025, 5)}, amber_found

Q = "“"; Qe = "”"
RAW = "Causes: see the root-cause slide."
YEARS = [
 (2019, "A calm climb", "[~10 s] 2019 was a calm climb. SPY went up 31 percent, with one small dip. TimesFM said up on only 38 percent of days, while 68 percent of the 5-day periods went up. "
  "For the next 5 days the AUC is 0.48, a coin flip. For tomorrow it is 0.57, but that could still be luck.",
  "Up 31% over the year, with one small dip (−7% in May).",
  f"Said {Q}up{Qe} on 38% of days, while 68% of 5-day periods went up.",
  "A calm year: nothing real to find. A coin flip."),
 (2020, "Covid crash and rebound", "[~12 s] 2020, the Covid crash. TimesFM said up on every day of the crash, and it was right on only 30 percent of those days. In the rebound it said up on only 41 percent of days, "
  "while just saying up every day would have been right 71 percent of the time. The AUC is 0.36, clearly worse than a coin flip. "
  "Now a habit of TimesFM: after a fall it expects a bounce, and after a rise it expects a dip. We call it the reversal bet. On its own it scores 0.54, so it is not the cause. The rest of the forecast scores 0.34.",
  "Down 34% in Feb–Mar (Covid), back at the high by August.",
  f"Said {Q}up{Qe} on every day of the crash; right only 30% of the time. In the rebound it said {Q}up{Qe} on 41% of days; saying {Q}up{Qe} every day would have been right 71%.",
  f"Clearly worse than a coin flip. Not the reversal bet (after a fall it expects a bounce): on its own it scores 0.54. The rest of the forecast scores 0.34. {RAW}"),
 (2021, "A smooth climb", "[~8 s] 2021, a smooth climb. The biggest dip was only 5 percent. With the drift layer, TimesFM said up on 68 percent of days, instead of 90 percent without it. AUC 0.49, a coin flip.",
  "Up 29%; the biggest dip was only −5%.",
  f"Said {Q}up{Qe} on 68% of days (TimesFM alone: 90%); 63% of 5-day periods went up.",
  "Little to rank in such a smooth year. A coin flip."),
 (2022, "Rate hikes: three falls, three bounces", "[~12 s] 2022, rate hikes: three falls, each with a sharp bounce. In the bounces the market went up in 71 percent of the 5-day periods, but TimesFM said up on only 1 percent of those days. "
  "It was more upbeat in the falls than in the bounces. The AUC is 0.38, clearly worse than a coin flip. Again it is not the reversal bet, which scores 0.51. It is the rest of the forecast, 0.37.",
  "Down 18% for the year (−24% from the high): three falls, each with a sharp bounce.",
  f"In the 3 bounces the market rose in 71% of 5-day periods, but TimesFM said {Q}up{Qe} on only 1% of those days. It was more upbeat in the falls than in the bounces.",
  f"Clearly worse than a coin flip. Again not the reversal bet (on its own it scores 0.51); the rest of the forecast (0.37) pushed it down in the bounces. {RAW}"),
 (2023, "A calm climb with a 10% dip", "[~8 s] 2023, a calm climb with a 10 percent dip. The drift layer lifts the up calls from 29 to 61 percent of days. That matches the market better, but the ranking is still a coin flip: AUC 0.47.",
  "Up 26%, led by AI and big tech; a 10% dip from August to October.",
  f"Said {Q}up{Qe} on 61% of days (TimesFM alone: 29%), close to the 62% of 5-day periods that went up.",
  "Its up-calls now match how often the market went up, but it still cannot tell which days are up: a coin flip."),
 (2024, "The best year for tomorrow", "[~12 s] 2024 flips. TimesFM alone had its worst tomorrow-year here, 0.42. With the drift layer it is the best, 0.60, and clearly above a coin flip. "
  "For tomorrow the layer leans on how wide TimesFM's price range is. We did not check why 2024 is special. It is one year out of seven, and the overall gains could be luck.",
  "Up 25%, with one sharp drop (−8%) in July–August.",
  f"Tomorrow: the best year, AUC 0.60, clearly above a coin flip. TimesFM alone scored 0.42 here. The next-5-days AUC is a coin flip (0.54).",
  "Not explained. For tomorrow the layer gets its score from how wide TimesFM's price range is, not from its up/down call. One year out of seven; the overall gains could be luck."),
 (2025, "Tariff shock and recovery", "[~13 s] 2025, the tariff shock. TimesFM said up on most fall days. In the recovery it said up on only 7 percent of days, while the market rose in 76 percent of the 5-day periods. "
  "For TimesFM alone we found part of the reason: a built-in step draws a straight line through the past year and continues it. After the fall, that line pointed down and pulled the forecast down. The drift layer does not fix that.",
  "Down 19% from February to April, back at the high in late June.",
  f"Said {Q}up{Qe} on 76% of fall days (right on 56%). In the recovery the market rose in 76% of 5-day periods, but TimesFM said {Q}up{Qe} on only 7% of days.",
  "Partly explained, for TimesFM alone: a built-in step draws a straight line through the past year and continues it. After the fall the line pointed down and pulled the forecast down (root-cause slide). The drift layer does not fix this."),
]
for yr, sub, notes, spy_t, tf_t, why_t in YEARS:
    s = new_slide(f"{yr} · {sub}", notes)
    x = 0.5
    for h in (5, 1):
        r = row(yr, h)
        fill, fg = RED if (yr, h) in red_found else AMBER if (yr, h) in amber_found else GREY
        b = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, 1.12, 3.05, 0.46, fill, radius=0.3, name=f"Badge h={h}")
        shape_text(b, f"{'Tomorrow' if h == 1 else 'Next 5 days'}: AUC {r.auc:.2f}", 17, color=fg, bold=True)
        x += 3.2
    pic(s, FIG / f"year_{yr}.png", 0.4, 1.8, 8.2, f"{yr}: SPY close with fall and recovery phases, TimesFM + drift layer up calls and market up days")
    lab = lambda t: [(t, {"bold": True, "color": BLUE, "size": 15})]
    text(s, 8.95, 1.85, 3.95, 5.0, [lab("SPY"), spy_t, lab("TimesFM + drift layer"), tf_t, lab("In plain words"), why_t], size=15, space_after=3, name="Explanation")
    # extra gap before labels: set space_before on label paragraphs
    for p in s.shapes[-1].text_frame.paragraphs:
        if p.runs and p.runs[0].font.bold and p.runs[0].text in ("TimesFM + drift layer", "In plain words"): p.space_before = Pt(9)
    text(s, 0.5, 6.9, 11.4, 0.45, ["TimesFM = with the drift layer. Red = clearly worse than a coin flip (the whole range that luck alone could give is below 0.5); amber = AUC below 0.45 but luck cannot be ruled out. "
                                   "F = fall, R = recovery; charts use 5-day periods. This year-by-year look was done after the test (exploratory)."], size=11, color=MUTE, name="Footnote")

# ---------------------------------------------------------------- 12 root cause (TIMESFM_ROOT_CAUSE.md)
def fig_slide(title, notes, png, alt, bullets, answer, foot, sub=None, title_size=32, fig_w=7.45, tx=8.2, bsize=16, bgap=12, bw=None):
    s = new_slide(title, notes, title_size=title_size)
    if sub: text(s, 0.5, 0.93, 12.3, 0.35, [sub], size=17, color=MUTE, name="Subtitle")
    pic(s, FIG / png, 0.45, 1.45, fig_w, alt)
    text(s, tx, 1.55, bw or 4.65, 4.4, bullets, size=bsize, space_after=bgap, bullets=True, name="Bullets")
    text(s, 0.5, 6.1, 12.3, 0.45, [answer], size=18, name="Answer")
    text(s, 0.5, 6.97, 11.2, 0.3, [foot], size=11, color=MUTE, name="Footnote")
    return s

fig_slide("Is the model itself the problem?",
 "[~18 s] Is the model itself the problem? This is TimesFM alone, before the drift layer. We split every forecast into two parts: the neural network's part, and the built-in straight trend line. "
 "Neither part is clearly worse than a coin flip. The weak years, 2020 and 2022, show up in the network's part. "
 "But simple forecasters that need no AI, like 'tomorrow equals today plus the average daily gain', score just as low in those years. "
 "In 2025 the trend line is the weaker part. So we found no defect in the network. Simple forecasters fail in the same years.",
 "root_cause.png", "Dot plot: AUC for the next 5 days with the range luck alone could give, for TimesFM as released, its neural network part, its built-in trend line part and three simple forecasters",
 ["We split each forecast into the neural network's part (a “transformer”) and the built-in trend line: AUC 0.48 and 0.47 (TimesFM as released: 0.46). Neither is clearly worse than a coin flip.",
  "2020 and 2022 are low in the network's part, but simple non-AI forecasters (e.g. “tomorrow = today + the average daily gain”) are just as low there.",
  "2025: the built-in trend line is the weaker part (0.38)."],
 [("Answer: ", {"bold": True}), ("no defect found in the neural network; simple forecasters fail in the same years.", {"bold": True, "color": BLUE})],
 "Exploratory (after the test); bars show the range that luck alone could give; details in TIMESFM_ROOT_CAUSE.md",
 sub="Analysis of TimesFM alone (before the drift layer), AUC for the next 5 days")

# ---------------------------------------------------------------- 13 TimesFM vs Chronos-2 (TIMESFM_VS_CHRONOS.md)
fig_slide("TimesFM-3 vs Chronos-2: same skill, different bets",
 "[~25 s] How does TimesFM compare with Chronos-2, the model from our teammate? Both are close to a coin flip. TimesFM scores 0.48 for tomorrow and 0.46 for the next 5 days. Chronos-2 scores 0.48 and 0.48. The difference is not real. "
 "Their daily calls agree on only about half of the days, like two separate coins. TimesFM bets on reversals: after a rise it expects a fall. Chronos-2 bets that the move continues, and it turns gloomy when the market swings a lot. "
 "Why? We ran a swap test: we gave both models the same input. Fed daily prices, both lean toward reversals. Fed daily returns, both lean toward the move continuing. And the accuracy does not change. "
 "So the difference is the input, not the model. One more thing: the drift layer does not help Chronos-2.",
 "timesfm_vs_chronos.png", "Grouped horizontal bars: how much the chance of up of TimesFM-3 and Chronos-2 moves with yesterday's move, the last 5 days' move, recent swings and the width of the model's own price range, for tomorrow and the next 5 days",
 ["Both are near a coin flip. AUC, tomorrow / next 5 days: TimesFM 0.48 / 0.46, Chronos-2 0.48 / 0.48. No real difference.",
  "Their daily calls agree on only about half of the days, like two separate coins.",
  "TimesFM bets on reversals (after a rise it expects a fall). Chronos-2 bets the move continues, and turns gloomy when the market swings a lot.",
  "Swap test: give both the same input and they bet the same way. Fed daily prices, both lean toward reversals; fed daily returns, both lean toward the move continuing. Accuracy does not change.",
  "The drift layer does not help Chronos-2."],
 [("The difference is the input, not the model. ", {"bold": True, "color": BLUE}), ("Same skill (a coin flip), different bets.", {"bold": True})],
 "Exploratory, after the test. Swap test details: TIMESFM_VS_CHRONOS.md, section 7. In the chart, + = moves together, − = moves against. Models used as is; Chronos-2 from the team's chronos branch.",
 sub="TimesFM alone, no drift layer · Chronos-2 is another ready-made forecaster", title_size=30, bsize=14, bgap=8)

# ---------------------------------------------------------------- 14 drift methods (TIMESFM_DRIFT.md)
fig_slide("The drift layer: four methods compared",
 "[~15 s] Here is the detail behind the drift layer. We tried four ways to correct TimesFM's output. "
 "A monthly up or down shift of all its chances. A small model on top that uses TimesFM's outputs. A vote that gives more weight to whichever model did well lately. And an alarm that refits when errors jump. "
 "We fixed the settings on 2015 to 2018, and ran the test once. The small model on top, for tomorrow, and the monthly shift, for the next 5 days, had the biggest gains on 2015 to 2018, so those are the two in this talk. "
 "The gains could be luck. For the next 5 days our simple model is clearly better.",
 "drift_methods.png", "Bar chart: test AUC for tomorrow and the next 5 days for TimesFM alone, four ways to correct it and our simple model",
 ["Four ways to correct TimesFM's output: a monthly up/down shift of all its chances (recalibration); a small model on top that uses TimesFM's outputs (stacking); a vote that gives more weight to whichever model did well lately (Hedge); an alarm that refits when errors jump (drift detector).",
  "Settings fixed on 2015–2018; the 2019–2025 test was run once.",
  "Used in this talk: the small model on top for tomorrow (0.48 → 0.52) and the monthly shift for the next 5 days (0.46 → 0.48), the biggest gains on 2015–2018.",
  "Gains could be luck. For the next 5 days the layer stays below a coin flip and our simple model (0.54) stays better; for tomorrow the small model on top is only 0.02 above our simple model."],
 [("Result: ", {"bold": True}), ("gains over TimesFM alone are at most 0.04 and could be luck.", {"bold": True, "color": BLUE})],
 "Settings fixed before the test for each method; the pick between methods used 2015–2018 numbers, but its rule was set after the test; details in TIMESFM_DRIFT.md",
 bsize=14, bgap=7)

# ---------------------------------------------------------------- 15 why the drift layer helped (TIMESFM_DRIFT.md section 7)
fig_slide("Why did the drift layer help?",
 "[~30 s] Why did the drift layer raise AUC? Tomorrow first. The layer ignores TimesFM's up or down call. Instead it looks at how wide TimesFM's price range is. A wide range means the model expects a jumpy market. "
 "On jumpy days there are slightly more up days. That is a small effect. And the VIX, the fear index, tells you almost the same: 0.506 against 0.520. So this is not a new skill. "
 "Now the next 5 days. Think of a teacher who shifts a whole class's marks up or down. The ranking inside the class does not change. Here the layer shifts whole months up or down. It never changes which day ranks higher inside a month. "
 "It pushed all of 2022 down, and 2022 was the one bear year, a year of falling prices. So it helped only because the year it pushed down really was the weakest. Without 2022 the gain is gone. "
 "One more check: a version that uses no answers at all, just TimesFM's score minus its own one-year average, gets the same 0.48. So the layer did not fix the direction call.",
 "drift_why.png", "Left: bar chart of test AUC for tomorrow for TimesFM's up/down call, its middle price, the width of its price range, the VIX, recent swings and the drift layer, against the 0.5 coin-flip line. Right: AUC gain for the next 5 days with all years (+0.022) and without 2022 (-0.006)",
 ["Tomorrow (0.48 → 0.52): the layer ignores TimesFM's up/down call. It looks at how wide TimesFM's price range is: a wide range = a jumpy market → slightly more up days.",
  "The VIX tells you almost the same (0.506, against 0.520 for the width alone), so this is not a new skill.",
  "Next 5 days (0.46 → 0.48): the layer shifts whole months up or down; it never changes which day ranks higher inside a month. It pushed all of 2022 down, and 2022 was the one bear year. Without 2022 the gain is gone.",
  "Like a teacher shifting a whole class's marks: the ranking inside the class stays; it only helped because the class it pushed down (2022) really was the weakest."],
 [("Answer: ", {"bold": True}), ("it did not fix the direction call. Tomorrow it reads TimesFM's price range as a gauge of how jumpy the market is; for the next 5 days the gain is the 2022 bear year.", {"bold": True, "color": BLUE})],
 "Exploratory, computed after the test. Dashed line = 0.5 (coin flip). A version with no answers used at all (TimesFM's score minus its own 1-year average) also gets 0.48. Details in TIMESFM_DRIFT.md, section 7",
 sub="TimesFM + drift layer: small model on top (tomorrow), monthly shift (next 5 days)", fig_w=6.8, tx=7.45, bw=5.4, bsize=14, bgap=6)

# ---------------------------------------------------------------- 16 VIX on top of the drift layer (TIMESFM_VIX.md, follow-up section)
fig_slide("Does the VIX help TimesFM?",
 "[~15 s] Does the VIX, the market's fear index, help on top of the drift layer? We start from the same drift layer as in the rest of this talk. We tried four changes. "
 "Refit the layer when the VIX flips between calm and jumpy. Pick the model by VIX regime. Add VIX terms. And take the VIX out of TimesFM's inputs. "
 "None is clearly better than the starting point, and taking the VIX out changes nothing. "
 "This question came after the earlier results, so it is exploratory.",
 "vix_drift_methods.png", "Bar chart: test AUC for tomorrow and the next 5 days for TimesFM with the drift layer, three VIX variants of the layer, the layer with the VIX taken out of the input and our simple model",
 ["Four changes to the drift layer: refit it when the VIX regime flips (calm or jumpy), pick the model by VIX regime, add VIX terms, or take the VIX out of TimesFM's inputs.",
  "Starting point: AUC 0.52 tomorrow, 0.48 next 5 days.",
  "With the VIX: tomorrow 0.46–0.52, next 5 days 0.47–0.49. None is clearly better than the starting point.",
  "VIX taken out of the input: 0.53 tomorrow, 0.48 next 5 days. The VIX in the input does not help."],
 [("Answer: ", {"bold": True}), ("no. No VIX version is clearly better than the drift layer alone.", {"bold": True, "color": BLUE})],
 "Exploratory: asked after the TimesFM, drift and VIX test results were known; the drift layer was picked by a rule set after its test. Details in TIMESFM_VIX.md",
 sub="Starting point: TimesFM + drift layer; VIX = the market's “fear index”")

# ---------------------------------------------------------------- 17 bottom line
s = new_slide("Bottom line",
 "[~10 s] So: with our drift layer, TimesFM is just above a coin flip for tomorrow and below it for the next 5 days. It is no better than our simple model. The weak years are still the big-fall years. We explained part of 2025. 2020 and 2022 stay open. "
 "Two extra checks, both after the test. Against Chronos-2 there is no real difference in skill: the difference is the input, not the model. "
 "And the drift layer works through how jumpy the market is, for tomorrow, and through the 2022 bear year, for the next 5 days. It does not make the up or down call better.")
items = ["TimesFM-3 + drift layer: AUC 0.52 tomorrow, 0.48 next 5 days. Our simple model: 0.51 and 0.54. Tomorrow: not clearly different. Next 5 days: worse.",
         "Calm years: a coin flip. The weak years are still the big-fall years: 2020 (0.36), 2022 (0.38), 2025 (0.40).",
         "Explained: part of TimesFM alone's 2025 (its built-in trend line). Not explained: 2020 and 2022. For tomorrow, the layer's score comes from how wide TimesFM's price range is, not from its up/down call."]
for i, t in enumerate(items):
    y = 1.6 + i * 1.7
    shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 0.5, y, 12.3, 1.4, CARD, radius=0.12, name=f"Card {i+1}")
    c = shape(s, MSO_SHAPE.OVAL, 0.85, y + 0.37, 0.66, 0.66, BLUE, name=f"Number {i+1}")
    shape_text(c, str(i + 1), 22, color=RGBColor(255, 255, 255), bold=True)
    text(s, 1.85, y + 0.1, 10.6, 1.2, [t], size=20, anchor=MSO_ANCHOR.MIDDLE, name=f"Bullet {i+1}")

text(s, 0.5, 6.62, 11.5, 0.6, [[("Also (exploratory): ", {"bold": True, "color": BLUE}),
     ("Chronos-2 has the same skill: the difference is the input, not the model. The drift layer works through how jumpy the market is (tomorrow) and the 2022 bear year (next 5 days).", {})]], size=16, name="Extra line")

prs.save(HERE / "timesfm3_slides.pptx")
print("saved", n_slides, "slides")
