"""Revise submitted Fig. 1: effective axis weights in box 5 and the 'True positive rate' label."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.parent / "PROTEINJ_SUBMISSION_2026-09-26" / "figures" / "ProteinJ_Figure_1.tif"
FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"

im = Image.open(SRC).convert("RGB")
d = ImageDraw.Draw(im)
size = next(s for s in range(12, 40) if ImageFont.truetype(FONT, s).getlength("Marginal change (0.35)") >= 186)
font = ImageFont.truetype(FONT, size)
for y, text in ((198, "Marginal change (0.58)"), (256, "Intermediate contact (0)"), (315, "Native topology (0.42)")):
    d.rectangle((1262, y - 14, 1500, y + 14), fill="white")
    d.text((1269, y), text, font=font, fill=(20, 20, 20), anchor="lm")

axis_font = ImageFont.truetype(FONT, next(s for s in range(12, 40)
                                          if ImageFont.truetype(FONT, s).getlength("False positive rate") >= 148))
w = int(axis_font.getlength("True positive rate")) + 4
label = Image.new("RGB", (w, 24), "white")
ImageDraw.Draw(label).text((w / 2, 12), "True positive rate", font=axis_font, fill=(20, 20, 20), anchor="mm")
d.rectangle((288, 674, 314, 844), fill="white")
label = label.rotate(90, expand=True)
im.paste(label, (300 - label.width // 2, 759 - label.height // 2))
out = ROOT / "figures" / "ProteinJ_Figure_1.tif"
im.save(out, dpi=(300, 300), compression="tiff_lzw")
im.save(out.with_suffix(".png"))
print(out, size)
