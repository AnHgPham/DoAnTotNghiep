"""Render defense Q&A backup tables as 1440x810 PNG slides."""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


WIDTH = 1440
HEIGHT = 810
SCALE = 2

NAVY = "#273896"
RED = "#F0242B"
RED_DARK = "#C81E2A"
BLUE = "#1A73E8"
TEXT = "#273896"
MUTED = "#52617B"
GRID = "#D8E0F3"
ROW_ALT = "#F8FAFF"
ROW_BLUE = "#F3F7FF"
LIGHT_BLUE = "#EEF5FF"
LIGHT_RED = "#FFF3F4"
WHITE = "#FFFFFF"

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "presentation" / "assets"


def _font_path(name: str) -> Path:
    local = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts"
    windows = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    candidates = [
        local / name,
        windows / name,
        windows / ("arialbd.ttf" if "Bold" in name else "arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No usable font found for {name}")


BARLOW_REGULAR = _font_path("Barlow-Regular.ttf")
BARLOW_BOLD = _font_path("Barlow-Bold.ttf")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(BARLOW_BOLD if bold else BARLOW_REGULAR), size * SCALE)


def p(value: float) -> int:
    return int(round(value * SCALE))


def box(values: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    return tuple(p(value) for value in values)  # type: ignore[return-value]


def text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    value: str,
    size: int,
    fill: str = TEXT,
    bold: bool = False,
    anchor: str | None = None,
) -> None:
    draw.text((p(xy[0]), p(xy[1])), value, font=font(size, bold), fill=fill, anchor=anchor)


def line(
    draw: ImageDraw.ImageDraw,
    points: tuple[float, float, float, float],
    fill: str = GRID,
    width: int = 2,
) -> None:
    draw.line(box(points), fill=fill, width=p(width))


def add_shadow_panel(
    image: Image.Image,
    bounds: tuple[float, float, float, float],
    radius: int = 6,
) -> None:
    shadow = Image.new("RGBA", image.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shifted = (bounds[0], bounds[1] + 4, bounds[2], bounds[3] + 4)
    shadow_draw.rounded_rectangle(box(shifted), radius=p(radius), fill=(23, 33, 90, 34))
    shadow = shadow.filter(ImageFilter.GaussianBlur(p(7)))
    image.alpha_composite(shadow)


def base_slide(title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGBA", (WIDTH * SCALE, HEIGHT * SCALE), WHITE)
    draw = ImageDraw.Draw(image)
    draw.rectangle(box((-30, 0, 1470, 150)), fill=NAVY)
    text(draw, (60, 88), title, 56, WHITE, True, anchor="lm")
    return image, draw


def render_multiscale() -> Path:
    image, draw = base_slide("Q&A / MULTI-SCALE EVIDENCE")
    text(draw, (70, 195), "Why evaluate Microset, Top500, and Full MSWC?", 40, bold=True, anchor="lm")
    text(draw, (1370, 198), "GSC test100 evidence", 20, MUTED, anchor="rm")
    line(draw, (70, 224, 1370, 224), BLUE)

    add_shadow_panel(image, (70, 245, 1370, 727))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(box((70, 245, 1370, 727)), radius=p(6), fill=WHITE, outline=GRID, width=p(2))
    draw.rounded_rectangle(box((70, 245, 1370, 307)), radius=p(6), fill=NAVY)
    draw.rectangle(box((70, 297, 1370, 307)), fill=NAVY)

    headers = [
        (88, "Stage"),
        (253, "Training vocabulary"),
        (473, "Purpose"),
        (703, "Best pipeline and result"),
        (1003, "What it proved"),
    ]
    for x, label in headers:
        text(draw, (x, 278), label, 22, WHITE, True, anchor="lm")

    row_tops = [307, 412, 517, 622]
    fills = [WHITE, ROW_ALT, WHITE, ROW_BLUE]
    accents = [RED, BLUE, NAVY, RED]
    for top, fill_color, accent in zip(row_tops, fills, accents):
        draw.rectangle(box((70, top, 1370, top + 105)), fill=fill_color)
        draw.rectangle(box((70, top, 77, top + 105)), fill=accent)

    for x in [235, 455, 685, 985]:
        line(draw, (x, 245, x, 727))
    for y in [412, 517, 622]:
        line(draw, (70, y, 1370, y))

    # Microset row
    text(draw, (90, 348), "Microset", 25, bold=True, anchor="lm")
    text(draw, (90, 378), "feasibility", 21, anchor="lm")
    text(draw, (253, 348), "31 words", 21, bold=True, anchor="lm")
    text(draw, (253, 378), "selected candidates", 21, anchor="lm")
    text(draw, (473, 348), "Fast screening", 21, bold=True, anchor="lm")
    text(draw, (473, 378), "architecture + loss", 21, anchor="lm")
    text(draw, (703, 330), "EdgeSpotFull T4 + PCEN", 18, NAVY, True, anchor="lm")
    text(draw, (703, 358), "SCAF", 18, MUTED, anchor="lm")
    text(draw, (703, 389), "ACC@1%FAR = 84.64%", 21, RED, True, anchor="lm")
    text(draw, (1003, 342), "SCAF family was strongest", 21, anchor="lm")
    text(draw, (1003, 373), "among the tested candidates.", 21, anchor="lm")

    # Top500 row
    text(draw, (90, 453), "Top500", 25, bold=True, anchor="lm")
    text(draw, (90, 483), "mid-scale", 21, anchor="lm")
    text(draw, (253, 449), "450 train", 21, bold=True, anchor="lm")
    text(draw, (253, 479), "+ 50 validation", 21, anchor="lm")
    text(draw, (473, 453), "Stress test", 21, bold=True, anchor="lm")
    text(draw, (473, 483), "locked epoch13 artifact", 21, anchor="lm")
    text(draw, (703, 435), "EdgeSpotFull T4 + PCEN", 18, NAVY, True, anchor="lm")
    text(draw, (703, 463), "SCAF + GE2E", 18, MUTED, anchor="lm")
    text(draw, (703, 494), "ACC@1%FAR = 85.62%", 21, RED, True, anchor="lm")
    text(draw, (1003, 447), "SCAF + GE2E remained strong", 21, anchor="lm")
    text(draw, (1003, 478), "at about 500 classes.", 21, anchor="lm")

    # Full fixed16 row
    text(draw, (90, 558), "Full fixed16", 25, bold=True, anchor="lm")
    text(draw, (90, 588), "controlled", 21, anchor="lm")
    text(draw, (253, 554), "37,387 train", 21, bold=True, anchor="lm")
    text(draw, (253, 584), "+ 763 validation", 21, anchor="lm")
    text(draw, (473, 546), "16 pipelines", 21, bold=True, anchor="lm")
    text(draw, (473, 576), "40 epochs", 18, anchor="lm")
    text(draw, (473, 600), "150 episodes/epoch", 18, anchor="lm")
    text(draw, (703, 540), "DSCNN-L + PCEN", 18, NAVY, True, anchor="lm")
    text(draw, (703, 568), "GE2E", 18, MUTED, anchor="lm")
    text(draw, (703, 599), "ACC@1%FAR = 82.34%", 21, RED, True, anchor="lm")
    text(draw, (1003, 552), "PCEN + GE2E was most stable;", 21, anchor="lm")
    text(draw, (1003, 582), "SCAF collapse appeared at scale.", 21, anchor="lm")

    # Full final row
    text(draw, (90, 663), "Full final", 25, bold=True, anchor="lm")
    text(draw, (90, 693), "Top-2 extended", 21, anchor="lm")
    text(draw, (253, 659), "37,387 train", 21, bold=True, anchor="lm")
    text(draw, (253, 689), "+ 763 validation", 21, anchor="lm")
    text(draw, (473, 663), "60 epochs", 21, bold=True, anchor="lm")
    text(draw, (473, 693), "300 episodes/epoch", 21, anchor="lm")
    text(draw, (703, 645), "DSCNN-L + PCEN", 18, NAVY, True, anchor="lm")
    text(draw, (703, 673), "GE2E", 18, MUTED, anchor="lm")
    text(draw, (703, 704), "ACC@1%FAR = 86.36%", 21, RED, True, anchor="lm")
    text(draw, (1003, 657), "Best overall result after", 21, anchor="lm")
    text(draw, (1003, 687), "extended training.", 21, anchor="lm")

    text(draw, (70, 765), "Important:", 19, RED, True, anchor="lm")
    text(
        draw,
        (163, 765),
        "staged evidence, not one controlled learning curve. Only Full fixed16 isolates components fairly.",
        19,
        MUTED,
        anchor="lm",
    )

    output = ASSET_DIR / "qa_multiscale_evidence.png"
    image.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS).save(output, quality=95)
    return output


def render_matrix() -> Path:
    image, draw = base_slide("Q&A / CONTROLLED 16-PIPELINE MATRIX")
    text(draw, (80, 195), "What did the full matrix prove?", 40, bold=True, anchor="lm")
    text(draw, (1360, 198), "GSC test100 ACC@1%FAR (%)", 20, MUTED, anchor="rm")
    line(draw, (80, 224, 1360, 224), BLUE)

    add_shadow_panel(image, (80, 248, 1360, 706))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(box((80, 248, 1360, 706)), radius=p(6), fill=WHITE, outline=GRID, width=p(2))
    draw.rounded_rectangle(box((80, 248, 1360, 318)), radius=p(6), fill=NAVY)
    draw.rectangle(box((80, 308, 1360, 318)), fill=NAVY)

    centers = [250, 537, 772, 1007, 1242]
    labels = ["Encoder + Frontend", "Triplet", "SCAF", "GE2E", "SCAF + GE2E"]
    for center, label in zip(centers, labels):
        text(draw, (center, 285), label, 23, WHITE, True, anchor="mm")

    row_tops = [318, 415, 512, 609]
    for index, top in enumerate(row_tops):
        draw.rectangle(box((80, top, 1360, top + 97)), fill=WHITE if index % 2 == 0 else ROW_ALT)

    # Best and collapse cell fills.
    draw.rectangle(box((890, 415, 1125, 512)), fill="#FFF1F2", outline=RED, width=p(3))
    draw.rectangle(box((890, 609, 1125, 706)), fill=LIGHT_BLUE, outline=BLUE, width=p(3))
    collapse_cells = [(655, 415, 890, 512), (1125, 415, 1360, 512), (655, 512, 890, 609),
                      (655, 609, 890, 706), (1125, 609, 1360, 706)]
    for bounds in collapse_cells:
        draw.rectangle(box(bounds), fill=LIGHT_RED)

    for x in [420, 655, 890, 1125]:
        line(draw, (x, 248, x, 706))
    for y in [415, 512, 609]:
        line(draw, (80, y, 1360, y))

    rows = [
        ("DSCNN-L + MFCC", "baseline frontend"),
        ("DSCNN-L + PCEN", "best accuracy branch"),
        ("EdgeSpotFull T4 + MFCC", "compact MFCC ablation"),
        ("EdgeSpotFull T4 + PCEN", "best compact branch"),
    ]
    for top, (title, subtitle) in zip(row_tops, rows):
        text(draw, (102, top + 39), title, 23, bold=True, anchor="lm")
        text(draw, (102, top + 69), subtitle, 18, MUTED, anchor="lm")

    values = [
        ["71.52", "70.08", "77.08", "69.04"],
        ["79.98", "69.44", "82.34", "69.44"],
        ["69.63", "69.44", "70.76", "69.67"],
        ["79.58", "69.44", "79.98", "69.44"],
    ]
    collapse = {(1, 1), (1, 3), (2, 1), (3, 1), (3, 3)}
    best = {(1, 2): "BEST SCREEN", (3, 2): "COMPACT BEST"}
    value_centers = [537, 772, 1007, 1242]

    for row_index, (top, row_values) in enumerate(zip(row_tops, values)):
        for col_index, (center, value) in enumerate(zip(value_centers, row_values)):
            key = (row_index, col_index)
            if key in collapse:
                text(draw, (center, top + 42), value, 27, RED_DARK, True, anchor="mm")
                text(draw, (center, top + 69), "[C] COLLAPSE", 16, RED_DARK, True, anchor="mm")
            elif key in best:
                text(draw, (center, top + 42), value, 30, RED, True, anchor="mm")
                note_fill = RED_DARK if key == (1, 2) else MUTED
                text(draw, (center, top + 69), best[key], 16, note_fill, True, anchor="mm")
            else:
                text(draw, (center, top + 49), value, 28, TEXT, True, anchor="mm")

    text(draw, (80, 746), "Same conditions", 21, TEXT, True, anchor="lm")
    text(draw, (253, 746), "Full cap620 | 40 epochs x 150 episodes | 30-way x 10", 18, MUTED, anchor="lm")
    text(draw, (80, 780), "[C] is not a valid 69% detector:", 18, RED, True, anchor="lm")
    text(draw, (363, 780), "AUC 50% | F1 0 | FRR 100% indicates reject-all collapse.", 18, MUTED, anchor="lm")

    output = ASSET_DIR / "qa_fixed16_matrix.png"
    image.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS).save(output, quality=95)
    return output


def _arrow(draw: ImageDraw.ImageDraw, start_x: int, end_x: int, y: int, color: str) -> None:
    draw.line((p(start_x), p(y), p(end_x - 10), p(y)), fill=color, width=p(4))
    draw.polygon(
        [(p(end_x - 10), p(y - 8)), (p(end_x), p(y)), (p(end_x - 10), p(y + 8))],
        fill=color,
    )


def render_scaf_collapse() -> Path:
    image, draw = base_slide("Q&A / WHY DID SCAF COLLAPSE?")
    text(
        draw,
        (70, 195),
        "The episode became tiny relative to the global classifier",
        40,
        bold=True,
        anchor="lm",
    )
    line(draw, (70, 224, 1370, 224), BLUE)

    panels = [
        (70, 250, 450, 610, ROW_ALT, GRID),
        (530, 250, 910, 610, "#FFF8F8", "#F4C9CE"),
        (990, 250, 1370, 610, "#FFF8F8", "#F4C9CE"),
    ]
    for panel in panels:
        add_shadow_panel(image, panel)
    draw = ImageDraw.Draw(image)
    for left, top, right, bottom, fill_color, outline in panels:
        draw.rounded_rectangle(
            box((left, top, right, bottom)),
            radius=p(7),
            fill=fill_color,
            outline=outline,
            width=p(2),
        )

    headers = [
        (70, 450, BLUE, "1. MICROSET: MATCHED"),
        (530, 910, RED, "2. FULL MSWC: MISMATCH"),
        (990, 1370, RED_DARK, "3. COLLAPSE SIGNATURE"),
    ]
    for left, right, color, label in headers:
        draw.rounded_rectangle(box((left, 250, right, 300)), radius=p(7), fill=color)
        draw.rectangle(box((left, 290, right, 300)), fill=color)
        text(draw, ((left + right) / 2, 276), label, 18, WHITE, True, anchor="mm")

    # Microset panel.
    text(draw, (95, 334), "Small global head", 30, bold=True, anchor="lm")
    text(draw, (95, 382), "31 classes", 34, BLUE, True, anchor="lm")
    text(draw, (95, 413), "K=3 -> only 93 trainable sub-centers", 18, MUTED, anchor="lm")
    line(draw, (95, 438, 425, 438))
    text(draw, (95, 470), "One episode", 22, bold=True, anchor="lm")
    text(draw, (95, 504), "31 classes x 16 clips", 22, anchor="lm")
    text(draw, (95, 536), "= 496 training clips", 22, anchor="lm")
    text(draw, (95, 573), "All 31 classes are represented.", 22, bold=True, anchor="lm")

    _arrow(draw, 466, 510, 430, BLUE)

    # Full-scale mismatch panel.
    text(draw, (555, 334), "Huge global head", 30, bold=True, anchor="lm")
    text(draw, (555, 382), "37,387 classes", 34, RED, True, anchor="lm")
    text(draw, (555, 413), "K=3 -> 112,161 trainable sub-centers", 18, MUTED, anchor="lm")
    line(draw, (555, 438, 885, 438), "#F4C9CE")
    text(draw, (555, 470), "One fixed16 episode", 22, bold=True, anchor="lm")
    text(draw, (555, 504), "30 classes x 10 clips", 22, anchor="lm")
    text(draw, (555, 536), "= 300 training clips", 22, anchor="lm")
    text(draw, (555, 566), "Only 0.08% appear, but logits still", 20, bold=True, anchor="lm")
    text(draw, (555, 592), "compare against all 37,387 classes.", 20, bold=True, anchor="lm")

    _arrow(draw, 926, 970, 430, RED)

    # Collapse panel.
    text(draw, (1015, 334), "Prototype geometry is lost", 29, bold=True, anchor="lm")
    text(draw, (1015, 374), "Default: scale 30, margin 0.5, weight 1.0", 18, MUTED, anchor="lm")
    text(draw, (1015, 410), "Global classification gradient can", 22, anchor="lm")
    text(draw, (1015, 440), "dominate the centroid objective.", 22, anchor="lm")
    metric_boxes = [
        (1015, 469, 1170, 523, "AUC 50%"),
        (1185, 469, 1340, 523, "F1 = 0"),
        (1015, 535, 1170, 589, "FRR 100%"),
        (1185, 535, 1340, 589, "KW 9.09%"),
    ]
    for left, top, right, bottom, label in metric_boxes:
        draw.rounded_rectangle(box((left, top, right, bottom)), radius=p(5), fill="#FFE8EA")
        text(draw, ((left + right) / 2, (top + bottom) / 2), label, 27, RED_DARK, True, anchor="mm")

    # Episode-size strip.
    text(draw, (70, 646), "Training episode sizes", 22, bold=True, anchor="lm")
    chips = [
        (70, 370, ROW_ALT, GRID, "Microset", "31 classes x 16 = 496 clips"),
        (390, 690, ROW_ALT, GRID, "Top500 epoch13", "30 classes x 20 = 600 clips"),
        (710, 1010, LIGHT_RED, "#F4C9CE", "Full fixed16", "30 classes x 10 = 300 clips"),
        (1030, 1370, ROW_ALT, GRID, "Full final GE2E", "40 classes x 10 = 400 clips"),
    ]
    for left, right, fill_color, outline, title, detail in chips:
        draw.rounded_rectangle(
            box((left, 670, right, 742)),
            radius=p(6),
            fill=fill_color,
            outline=outline,
            width=p(2),
        )
        text(draw, (left + 20, 694), title, 20, bold=True, anchor="lm")
        text(draw, (left + 20, 723), detail, 18, MUTED, anchor="lm")

    text(draw, (70, 778), "Working diagnosis:", 18, RED, True, anchor="lm")
    text(
        draw,
        (231, 778),
        "huge global SCAF head + sparse episode coverage + default loss scale. GE2E uses episode centroids and stays stable.",
        18,
        MUTED,
        anchor="lm",
    )

    output = ASSET_DIR / "qa_scaf_collapse_explained.png"
    image.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS).save(output, quality=95)
    return output


def render_scaf_collapse_simple() -> Path:
    image, draw = base_slide("Q&A / WHY DID SCAF COLLAPSE?")
    text(draw, (720, 207), "Three factors created a loss imbalance", 42, bold=True, anchor="mm")
    line(draw, (90, 242, 1350, 242), BLUE)

    panels = [
        (80, 285, 440, 435, ROW_ALT, GRID),
        (540, 285, 900, 435, ROW_ALT, GRID),
        (1000, 285, 1360, 435, LIGHT_RED, "#F4C9CE"),
    ]
    for panel in panels:
        add_shadow_panel(image, panel, radius=8)
    draw = ImageDraw.Draw(image)
    for left, top, right, bottom, fill_color, outline in panels:
        draw.rounded_rectangle(
            box((left, top, right, bottom)),
            radius=p(8),
            fill=fill_color,
            outline=outline,
            width=p(2),
        )

    text(draw, (260, 336), "37,387 classes", 34, bold=True, anchor="mm")
    text(draw, (260, 382), "K=3 -> 112,161 centers", 23, MUTED, anchor="mm")
    text(draw, (490, 360), "+", 52, BLUE, True, anchor="mm")

    text(draw, (720, 336), "30 classes", 34, bold=True, anchor="mm")
    text(draw, (720, 382), "per episode = only 0.08%", 23, MUTED, anchor="mm")
    text(draw, (950, 360), "+", 52, BLUE, True, anchor="mm")

    text(draw, (1180, 336), "Default SCAF", 34, bold=True, anchor="mm")
    text(draw, (1180, 382), "scale=30, weight=1.0", 23, MUTED, anchor="mm")

    # Merge the three causes into one point.
    for center in (260, 720, 1180):
        line(draw, (center, 452, center, 495), BLUE, 3)
    line(draw, (260, 495, 1180, 495), BLUE, 3)
    draw.line((p(720), p(495), p(720), p(518)), fill=RED, width=p(4))
    draw.polygon(
        [(p(712), p(518)), (p(720), p(530)), (p(728), p(518))],
        fill=RED,
    )

    add_shadow_panel(image, (260, 540, 1180, 640), radius=8)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(box((260, 540, 1180, 640)), radius=p(8), fill=RED)
    text(
        draw,
        (720, 590),
        "SCAF classification gradient becomes dominant",
        34,
        WHITE,
        True,
        anchor="mm",
    )

    draw.line((p(720), p(648), p(720), p(680)), fill=RED, width=p(4))
    draw.polygon(
        [(p(712), p(680)), (p(720), p(692)), (p(728), p(680))],
        fill=RED,
    )
    text(draw, (720, 735), "EMBEDDING COLLAPSE", 46, RED, True, anchor="mm")
    text(
        draw,
        (720, 782),
        "Working diagnosis for the full 37,387-class setting",
        20,
        MUTED,
        anchor="mm",
    )

    output = ASSET_DIR / "qa_scaf_collapse_simple.png"
    image.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS).save(output, quality=95)
    return output


def render_scaf_collapse_layman() -> Path:
    image, draw = base_slide("Q&A / WHY DID SCAF FAIL AT FULL SCALE?")
    text(draw, (720, 203), "Think of SCAF as a giant filing cabinet", 42, bold=True, anchor="mm")
    line(draw, (80, 238, 1360, 238), BLUE)

    panels = [
        (70, 275, 450, 620, ROW_ALT, GRID),
        (530, 275, 910, 620, ROW_ALT, GRID),
        (990, 275, 1370, 620, LIGHT_RED, "#F4C9CE"),
    ]
    for panel in panels:
        add_shadow_panel(image, panel, radius=8)
    draw = ImageDraw.Draw(image)
    for left, top, right, bottom, fill_color, outline in panels:
        draw.rounded_rectangle(
            box((left, top, right, bottom)),
            radius=p(8),
            fill=fill_color,
            outline=outline,
            width=p(2),
        )

    panel_headers = [
        (70, 450, NAVY, "1. THE FULL TASK"),
        (530, 910, BLUE, "2. ONE TRAINING LESSON"),
        (990, 1370, RED, "3. SCAF PUSHES TOO HARD"),
    ]
    for left, right, color, label in panel_headers:
        draw.rounded_rectangle(box((left, 275, right, 327)), radius=p(8), fill=color)
        draw.rectangle(box((left, 317, right, 327)), fill=color)
        text(draw, ((left + right) / 2, 301), label, 25, WHITE, True, anchor="mm")

    # Panel 1: a large filing cabinet.
    text(draw, (260, 361), "37,387 word boxes", 35, bold=True, anchor="mm")
    for row in range(3):
        for col in range(6):
            left = 112 + col * 50
            top = 410 + row * 38
            draw.rounded_rectangle(
                box((left, top, left + 42, top + 30)),
                radius=p(3),
                fill="#DDE7FB",
                outline=BLUE,
                width=p(1.5),
            )
    text(draw, (260, 568), "Each box is one keyword.", 24, anchor="mm")

    _arrow(draw, 466, 510, 448, BLUE)

    # Panel 2: a very small lesson compared with the cabinet.
    text(draw, (720, 361), "Only 30 words", 35, bold=True, anchor="mm")
    draw.rounded_rectangle(box((585, 410, 855, 540)), radius=p(7), fill=WHITE, outline=GRID, width=p(2))
    rows = [(445, BLUE, 815), (477, RED, 790), (509, NAVY, 825)]
    for y, color, end_x in rows:
        draw.ellipse(box((612, y - 13, 638, y + 13)), fill=color)
        draw.line((p(655), p(y), p(end_x), p(y)), fill=GRID, width=p(8))
    text(draw, (720, 572), "Most boxes are unseen now.", 24, anchor="mm")

    _arrow(draw, 926, 970, 448, BLUE)

    # Panel 3: an overly strong push compresses the groups together.
    text(draw, (1180, 361), "Too much training pressure", 29, bold=True, anchor="mm")
    pressure_lines = [((1045, 430), (1140, 485)), ((1315, 430), (1220, 485)), ((1180, 408), (1180, 472))]
    for (x1, y1), (x2, y2) in pressure_lines:
        draw.line((p(x1), p(y1), p(x2), p(y2)), fill=RED, width=p(5))
    dots = [(1155, 505, NAVY), (1183, 493, BLUE), (1207, 515, RED), (1177, 525, NAVY)]
    for cx, cy, color in dots:
        draw.ellipse(box((cx - 15, cy - 15, cx + 15, cy + 15)), fill=color)
    text(draw, (1180, 560), "The embedding space", 22, RED, True, anchor="mm")
    text(draw, (1180, 586), "becomes unstable.", 22, RED, True, anchor="mm")

    add_shadow_panel(image, (250, 650, 1190, 738), radius=8)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(box((250, 650, 1190, 738)), radius=p(8), fill=RED)
    text(
        draw,
        (720, 694),
        "MODEL BECOMES UNSURE -> REJECTS ALMOST EVERYTHING",
        32,
        WHITE,
        True,
        anchor="mm",
    )
    text(
        draw,
        (720, 781),
        "37,387 names to learn + only 30 shown per lesson + too much training pressure",
        22,
        MUTED,
        anchor="mm",
    )

    output = ASSET_DIR / "qa_scaf_collapse_layman.png"
    image.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS).save(output, quality=95)
    return output


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for output in (
        render_multiscale(),
        render_matrix(),
        render_scaf_collapse(),
        render_scaf_collapse_simple(),
        render_scaf_collapse_layman(),
    ):
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
