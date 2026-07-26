"""Render three deterministic Q&A backup slides as matching PNG and SVG files."""

from __future__ import annotations

import html
import math
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1440
HEIGHT = 810
SCALE = 2

NAVY = "#273896"
BLUE = "#1A73E8"
RED = "#F0242B"
WHITE = "#FFFFFF"
MUTED = "#52617B"
GRID = "#D8E0F3"
LIGHT_BLUE = "#EEF5FF"
LIGHT_RED = "#FFF3F4"
ROW_ALT = "#F8FAFF"

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


def _p(value: float) -> int:
    return int(round(value * SCALE))


def _box(values: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    return tuple(_p(value) for value in values)  # type: ignore[return-value]


class SlideCanvas:
    """Draw the same simple vector commands to a high-resolution PNG and SVG."""

    def __init__(self, header: str, section: str, *, header_size: int = 54, section_size: int = 40):
        self.image = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), WHITE)
        self.draw = ImageDraw.Draw(self.image)
        self.svg: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
            f'viewBox="0 0 {WIDTH} {HEIGHT}">',
            "<defs>",
            self._marker("navy", NAVY),
            self._marker("blue", BLUE),
            self._marker("red", RED),
            self._marker("muted", MUTED),
            "</defs>",
        ]
        self.rect(-30, 0, 1470, 150, NAVY)
        self.text(60, 88, header, header_size, WHITE, bold=True, anchor="lm")
        self.text(70, 195, section, section_size, NAVY, bold=True, anchor="lm")
        self.line(70, 224, 1370, 224, BLUE, 2)

    @staticmethod
    def _marker(name: str, color: str) -> str:
        return (
            f'<marker id="arrow-{name}" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="8" markerHeight="8" orient="auto">'
            f'<path d="M0,0 L10,5 L0,10 Z" fill="{color}"/></marker>'
        )

    @staticmethod
    def _font(size: int, bold: bool) -> ImageFont.FreeTypeFont:
        path = BARLOW_BOLD if bold else BARLOW_REGULAR
        return ImageFont.truetype(str(path), size * SCALE)

    def rect(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        fill: str,
        *,
        stroke: str | None = None,
        width: float = 1.5,
        radius: float = 0,
    ) -> None:
        if radius:
            self.draw.rounded_rectangle(
                _box((x1, y1, x2, y2)),
                radius=_p(radius),
                fill=fill,
                outline=stroke,
                width=_p(width) if stroke else 1,
            )
        else:
            self.draw.rectangle(
                _box((x1, y1, x2, y2)),
                fill=fill,
                outline=stroke,
                width=_p(width) if stroke else 1,
            )
        stroke_attr = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
        radius_attr = f' rx="{radius}"' if radius else ""
        self.svg.append(
            f'<rect x="{x1}" y="{y1}" width="{x2 - x1}" height="{y2 - y1}"'
            f'{radius_attr} fill="{fill}"{stroke_attr}/>'
        )

    def text(
        self,
        x: float,
        y: float,
        value: str,
        size: int,
        fill: str = NAVY,
        *,
        bold: bool = False,
        anchor: str = "lm",
    ) -> None:
        self.draw.text(
            (_p(x), _p(y)),
            value,
            font=self._font(size, bold),
            fill=fill,
            anchor=anchor,
        )
        svg_anchor = {"lm": "start", "mm": "middle", "rm": "end"}[anchor]
        weight = 700 if bold else 400
        self.svg.append(
            f'<text x="{x}" y="{y}" text-anchor="{svg_anchor}" dominant-baseline="middle" '
            f'font-family="Barlow, Arial, sans-serif" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}">{html.escape(value)}</text>'
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str = GRID,
        width: float = 2,
        *,
        arrow: bool = False,
    ) -> None:
        self.draw.line((_p(x1), _p(y1), _p(x2), _p(y2)), fill=color, width=_p(width))
        marker = ""
        if arrow:
            angle = math.atan2(y2 - y1, x2 - x1)
            length = 10
            spread = 5
            base_x = x2 - length * math.cos(angle)
            base_y = y2 - length * math.sin(angle)
            left = (
                base_x + spread * math.cos(angle + math.pi / 2),
                base_y + spread * math.sin(angle + math.pi / 2),
            )
            right = (
                base_x + spread * math.cos(angle - math.pi / 2),
                base_y + spread * math.sin(angle - math.pi / 2),
            )
            self.draw.polygon(
                [(_p(x2), _p(y2)), (_p(left[0]), _p(left[1])), (_p(right[0]), _p(right[1]))],
                fill=color,
            )
            marker_name = {NAVY: "navy", BLUE: "blue", RED: "red", MUTED: "muted"}.get(color, "navy")
            marker = f' marker-end="url(#arrow-{marker_name})"'
        self.svg.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="{width}"{marker}/>'
        )

    def circle(
        self,
        cx: float,
        cy: float,
        radius: float,
        fill: str,
        *,
        stroke: str | None = None,
        width: float = 1.5,
    ) -> None:
        self.draw.ellipse(
            _box((cx - radius, cy - radius, cx + radius, cy + radius)),
            fill=fill,
            outline=stroke,
            width=_p(width) if stroke else 1,
        )
        stroke_attr = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
        self.svg.append(f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{fill}"{stroke_attr}/>' )

    def waveform(self, x1: float, x2: float, y: float, color: str = BLUE) -> None:
        amplitudes = [2, 8, 15, 6, 20, 10, 4, 13, 7, 2]
        step = (x2 - x1) / (len(amplitudes) - 1)
        for index, amplitude in enumerate(amplitudes):
            x = x1 + index * step
            self.line(x, y - amplitude, x, y + amplitude, color, 2)

    def lock(self, x: float, y: float, color: str = BLUE) -> None:
        self.rect(x - 10, y, x + 10, y + 18, WHITE, stroke=color, width=2, radius=3)
        self.draw.arc(_box((x - 8, y - 11, x + 8, y + 7)), 180, 360, fill=color, width=_p(2))
        self.svg.append(
            f'<path d="M{x - 8},{y} A8,8 0 0 1 {x + 8},{y}" fill="none" '
            f'stroke="{color}" stroke-width="2"/>'
        )

    def save(self, stem: str) -> tuple[Path, Path]:
        png_path = ASSET_DIR / f"{stem}.png"
        svg_path = ASSET_DIR / f"{stem}.svg"
        self.image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS).save(png_path, quality=95)
        svg_path.write_text("\n".join([*self.svg, "</svg>"]) + "\n", encoding="utf-8")
        return png_path, svg_path


def panel_header(
    canvas: SlideCanvas,
    x1: float,
    y1: float,
    x2: float,
    title: str,
    color: str,
    *,
    size: int = 21,
) -> None:
    canvas.rect(x1, y1, x2, y1 + 48, color, radius=7)
    canvas.rect(x1, y1 + 38, x2, y1 + 48, color)
    canvas.text((x1 + x2) / 2, y1 + 24, title, size, WHITE, bold=True, anchor="mm")


def step_box(
    canvas: SlideCanvas,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    title: str,
    lines: list[str],
    *,
    fill: str = WHITE,
    stroke: str = GRID,
    title_size: int = 18,
    body_size: int = 15,
) -> None:
    canvas.rect(x1, y1, x2, y2, fill, stroke=stroke, width=1.5, radius=6)
    center = (x1 + x2) / 2
    canvas.text(center, y1 + 25, title, title_size, NAVY, bold=True, anchor="mm")
    start_y = y1 + 53
    for index, value in enumerate(lines):
        canvas.text(center, start_y + index * 21, value, body_size, MUTED, anchor="mm")


def render_prototype_threshold_margin() -> tuple[Path, Path]:
    c = SlideCanvas(
        "Q&A / OPEN-SET DECISION",
        "From 10 examples to ACCEPT or UNKNOWN",
        header_size=58,
    )

    # 1. Create one visual memory vector from ten support examples.
    c.rect(70, 250, 430, 650, WHITE, stroke=GRID, width=2, radius=7)
    panel_header(c, 70, 250, 430, "1. CREATE A PROTOTYPE", NAVY, size=21)
    c.text(250, 325, "10 EXAMPLES", 19, NAVY, bold=True, anchor="mm")
    c.waveform(125, 330, 360, BLUE)
    c.circle(356, 360, 22, NAVY)
    c.text(356, 360, "10", 16, WHITE, bold=True, anchor="mm")
    c.line(250, 390, 250, 414, BLUE, 3, arrow=True)
    c.rect(145, 420, 355, 474, LIGHT_BLUE, stroke=BLUE, width=2, radius=6)
    c.text(250, 447, "SAME ENCODER", 20, NAVY, bold=True, anchor="mm")
    c.lock(329, 438, BLUE)
    c.line(250, 474, 250, 500, BLUE, 3, arrow=True)
    for index in range(10):
        c.circle(176 + (index % 5) * 37, 516 + (index // 5) * 25, 7, [NAVY, BLUE, RED][index % 3])
    c.text(250, 566, "AVERAGE", 16, MUTED, bold=True, anchor="mm")
    c.line(250, 575, 250, 590, BLUE, 3, arrow=True)
    c.circle(250, 612, 24, LIGHT_BLUE, stroke=BLUE, width=3)
    c.circle(250, 612, 10, NAVY)
    c.text(300, 612, "PROTOTYPE", 18, NAVY, bold=True)

    # 2. Embedding map: threshold is a radius around the nearest prototype.
    c.rect(460, 250, 960, 650, WHITE, stroke=GRID, width=2, radius=7)
    panel_header(c, 460, 250, 960, "2. COMPARE THE QUERY", BLUE, size=21)
    c.waveform(500, 585, 333, RED)
    c.text(542, 365, "QUERY", 15, RED, bold=True, anchor="mm")
    c.line(595, 333, 624, 333, BLUE, 3, arrow=True)
    c.rect(630, 306, 765, 360, LIGHT_BLUE, stroke=BLUE, width=2, radius=6)
    c.text(697, 333, "ENCODER", 18, NAVY, bold=True, anchor="mm")
    c.lock(744, 324, BLUE)
    c.line(775, 333, 812, 333, BLUE, 3, arrow=True)
    c.circle(835, 333, 13, RED)
    c.text(835, 365, "q", 17, RED, bold=True, anchor="mm")
    c.line(835, 375, 705, 474, RED, 2.5, arrow=True)

    c.rect(490, 395, 930, 630, ROW_ALT, stroke=GRID, radius=6)
    c.circle(640, 505, 88, LIGHT_BLUE, stroke=BLUE, width=2)
    c.text(570, 427, "threshold T", 15, BLUE, bold=True)
    c.line(640, 505, 700, 500, BLUE, 3)
    c.line(700, 500, 830, 455, MUTED, 2)
    c.text(668, 483, "d1", 16, BLUE, bold=True, anchor="mm")
    c.text(770, 463, "d2", 16, MUTED, bold=True, anchor="mm")

    for x, y in [(596, 478), (611, 522), (650, 545), (664, 468), (620, 495)]:
        c.circle(x, y, 6, BLUE)
    for x, y in [(806, 430), (849, 430), (853, 475), (813, 480)]:
        c.circle(x, y, 6, MUTED)
    for x, y in [(790, 548), (844, 545), (850, 590), (800, 596)]:
        c.circle(x, y, 6, MUTED)
    for x, y, label in [(640, 505, "zero"), (830, 455, "yes"), (820, 570, "no")]:
        c.circle(x, y, 15, WHITE, stroke=NAVY, width=3)
        c.circle(x, y, 6, NAVY)
        c.text(x, y + 28, label, 15, NAVY, bold=True, anchor="mm")
    c.circle(700, 500, 12, RED)
    c.text(713, 520, "q", 16, RED, bold=True)

    # 3. Two simple questions determine the demo decision.
    c.rect(990, 250, 1370, 650, WHITE, stroke=GRID, width=2, radius=7)
    panel_header(c, 990, 250, 1370, "3. DECIDE", NAVY, size=21)
    c.text(1180, 325, "CLOSE ENOUGH?", 20, NAVY, bold=True, anchor="mm")
    c.line(1045, 365, 1315, 365, GRID, 7)
    c.circle(1045, 365, 9, NAVY)
    c.circle(1150, 365, 10, BLUE)
    c.line(1240, 343, 1240, 387, RED, 4)
    c.text(1045, 398, "p*", 15, NAVY, bold=True, anchor="mm")
    c.text(1150, 398, "q / d1", 15, BLUE, bold=True, anchor="mm")
    c.text(1240, 398, "T", 17, RED, bold=True, anchor="mm")

    c.text(1180, 440, "CLEAR WINNER?", 20, NAVY, bold=True, anchor="mm")
    c.text(1030, 476, "nearest", 15, NAVY, bold=True)
    c.line(1110, 476, 1190, 476, BLUE, 8)
    c.circle(1190, 476, 7, BLUE)
    c.text(1210, 476, "d1", 15, BLUE, bold=True)
    c.text(1030, 516, "second", 15, MUTED, bold=True)
    c.line(1110, 516, 1280, 516, MUTED, 8)
    c.circle(1280, 516, 7, MUTED)
    c.text(1300, 516, "d2", 15, MUTED, bold=True)
    c.line(1190, 545, 1280, 545, RED, 3)
    c.line(1190, 538, 1190, 552, RED, 3)
    c.line(1280, 538, 1280, 552, RED, 3)
    c.text(1235, 565, "margin m", 15, RED, bold=True, anchor="mm")

    c.text(1095, 588, "BOTH PASS", 13, NAVY, bold=True, anchor="mm")
    c.text(1265, 588, "ANY FAILS", 13, RED, bold=True, anchor="mm")
    c.rect(1025, 600, 1165, 635, NAVY, radius=5)
    c.text(1095, 618, "ACCEPT", 18, WHITE, bold=True, anchor="mm")
    c.rect(1195, 600, 1335, 635, RED, radius=5)
    c.text(1265, 618, "UNKNOWN", 18, WHITE, bold=True, anchor="mm")

    # Minimal memory prompts for oral explanation.
    c.rect(70, 680, 690, 775, LIGHT_BLUE, stroke=BLUE, width=2, radius=7)
    c.text(95, 709, "THRESHOLD T", 22, NAVY, bold=True)
    c.text(95, 740, "Close enough?", 19, MUTED)
    c.text(385, 740, "DEV -> FROZEN TEST", 16, BLUE, bold=True)
    c.rect(750, 680, 1370, 775, LIGHT_RED, stroke=RED, width=2, radius=7)
    c.text(775, 709, "MARGIN  m = d2 - d1", 22, NAVY, bold=True)
    c.text(775, 740, "Clear winner?", 19, MUTED)
    c.text(1035, 740, "DEMO GUARD, NOT TRIPLET LOSS", 15, RED, bold=True)
    return c.save("qa_prototype_threshold_margin")


def vertical_connector(c: SlideCanvas, x: float, y1: float, y2: float, color: str = BLUE) -> None:
    c.line(x, y1, x, y2, color, 2.5, arrow=True)


def render_training_enrollment_inference() -> tuple[Path, Path]:
    c = SlideCanvas(
        "Q&A / THREE SYSTEM STAGES",
        "Train once, add keywords, then compare queries",
        header_size=56,
    )
    columns = [(70, 450, NAVY, "1. TRAINING"), (530, 910, BLUE, "2. ENROLLMENT"), (990, 1370, NAVY, "3. INFERENCE")]
    for x1, x2, color, title in columns:
        c.rect(x1, 255, x2, 655, WHITE, stroke=GRID, width=2, radius=7)
        panel_header(c, x1, 255, x2, title, color, size=23)

    # Training: labelled audio is used to learn the embedding space.
    c.text(132, 330, "LABELLED AUDIO", 15, NAVY, bold=True, anchor="mm")
    c.waveform(92, 158, 370, BLUE)
    c.waveform(92, 158, 414, NAVY)
    c.text(125, 450, "MSWC", 16, MUTED, bold=True, anchor="mm")
    c.line(170, 392, 198, 392, BLUE, 3, arrow=True)
    c.rect(205, 345, 325, 437, LIGHT_BLUE, stroke=BLUE, width=2, radius=6)
    c.text(265, 372, "MFCC / PCEN", 15, MUTED, bold=True, anchor="mm")
    c.line(220, 392, 310, 392, GRID, 1.5)
    c.text(265, 417, "ENCODER", 19, NAVY, bold=True, anchor="mm")
    c.line(332, 392, 354, 392, BLUE, 3, arrow=True)
    for x, y, color in [
        (372, 370, BLUE), (388, 360, BLUE), (398, 380, BLUE),
        (370, 414, RED), (388, 425, RED), (402, 408, RED),
    ]:
        c.circle(x, y, 7, color)
    c.text(387, 458, "EMBEDDINGS", 15, NAVY, bold=True, anchor="mm")
    c.line(390, 442, 390, 505, RED, 2.5)
    c.line(390, 505, 265, 505, RED, 2.5)
    c.line(265, 505, 265, 441, RED, 2.5, arrow=True)
    c.text(328, 529, "LOSS + BACKPROP", 16, RED, bold=True, anchor="mm")
    c.rect(105, 570, 415, 625, LIGHT_RED, stroke=RED, width=2, radius=6)
    c.text(260, 598, "UPDATE ENCODER WEIGHTS", 18, RED, bold=True, anchor="mm")

    # Enrollment: the frozen encoder creates one new prototype.
    c.text(590, 330, "NEW KEYWORD", 16, NAVY, bold=True, anchor="mm")
    c.waveform(555, 625, 390, BLUE)
    c.circle(632, 390, 20, NAVY)
    c.text(632, 390, "10", 14, WHITE, bold=True, anchor="mm")
    c.line(652, 390, 674, 390, BLUE, 3, arrow=True)
    c.rect(680, 345, 790, 435, LIGHT_BLUE, stroke=BLUE, width=2, radius=6)
    c.text(735, 375, "FROZEN", 15, MUTED, bold=True, anchor="mm")
    c.text(735, 410, "ENCODER", 19, NAVY, bold=True, anchor="mm")
    c.lock(770, 356, BLUE)
    c.line(797, 390, 816, 390, BLUE, 3, arrow=True)
    for index in range(10):
        c.circle(825 + (index % 3) * 18, 370 + (index // 3) * 18, 5, [NAVY, BLUE, RED][index % 3])
    c.line(844, 438, 844, 480, BLUE, 3, arrow=True)
    c.text(862, 462, "MEAN", 13, MUTED, bold=True)
    c.circle(844, 510, 24, LIGHT_BLUE, stroke=BLUE, width=3)
    c.circle(844, 510, 10, NAVY)
    c.text(720, 548, "NEW PROTOTYPE", 17, NAVY, bold=True, anchor="mm")
    c.rect(565, 570, 875, 625, LIGHT_BLUE, stroke=BLUE, width=2, radius=6)
    c.text(720, 590, "ADD TO PROTOTYPE MEMORY", 17, NAVY, bold=True, anchor="mm")
    c.text(720, 613, "NO RETRAINING", 14, BLUE, bold=True, anchor="mm")

    # Inference: a query is compared with frozen prototype memory.
    c.text(1045, 330, "QUERY", 16, RED, bold=True, anchor="mm")
    c.waveform(1012, 1078, 390, RED)
    c.line(1085, 390, 1104, 390, BLUE, 3, arrow=True)
    c.rect(1110, 345, 1215, 435, LIGHT_BLUE, stroke=BLUE, width=2, radius=6)
    c.text(1162, 375, "FROZEN", 15, MUTED, bold=True, anchor="mm")
    c.text(1162, 410, "ENCODER", 18, NAVY, bold=True, anchor="mm")
    c.lock(1195, 356, BLUE)
    c.line(1222, 390, 1238, 390, BLUE, 3, arrow=True)
    c.circle(1290, 400, 48, LIGHT_BLUE, stroke=BLUE, width=2)
    c.circle(1275, 400, 13, WHITE, stroke=NAVY, width=3)
    c.circle(1275, 400, 5, NAVY)
    c.circle(1315, 400, 10, RED)
    c.line(1275, 400, 1315, 400, BLUE, 3)
    c.text(1295, 462, "NEAREST + THRESHOLD", 15, NAVY, bold=True, anchor="mm")
    c.line(1295, 470, 1295, 495, BLUE, 3, arrow=True)
    c.rect(1025, 505, 1160, 550, NAVY, radius=5)
    c.text(1092, 528, "ACCEPT", 17, WHITE, bold=True, anchor="mm")
    c.rect(1200, 505, 1335, 550, RED, radius=5)
    c.text(1267, 528, "UNKNOWN", 17, WHITE, bold=True, anchor="mm")
    c.rect(1025, 570, 1335, 625, ROW_ALT, stroke=GRID, width=2, radius=6)
    c.text(1180, 598, "COMPARE - DO NOT UPDATE", 18, NAVY, bold=True, anchor="mm")

    # Stage-to-stage continuity and the one fact to remember for Q&A.
    c.line(450, 280, 525, 280, BLUE, 3, arrow=True)
    c.line(910, 280, 985, 280, BLUE, 3, arrow=True)
    c.rect(70, 690, 1370, 780, NAVY, radius=6)
    c.line(503, 690, 503, 780, WHITE, 1)
    c.line(937, 690, 937, 780, WHITE, 1)
    c.text(286, 719, "TRAINING", 18, "#DCE6FF", bold=True, anchor="mm")
    c.text(286, 751, "WEIGHTS CHANGE", 24, WHITE, bold=True, anchor="mm")
    c.text(720, 719, "ENROLLMENT", 18, "#DCE6FF", bold=True, anchor="mm")
    c.text(720, 751, "PROTOTYPE MEMORY CHANGES", 21, WHITE, bold=True, anchor="mm")
    c.text(1153, 719, "INFERENCE", 18, "#DCE6FF", bold=True, anchor="mm")
    c.text(1153, 751, "NO MODEL CHANGE", 23, WHITE, bold=True, anchor="mm")
    return c.save("qa_training_enrollment_inference")


def table_cell_lines(
    c: SlideCanvas,
    x: float,
    y: float,
    lines: list[str],
    *,
    size: int = 16,
    fill: str = NAVY,
    bold_first: bool = False,
    gap: int = 21,
) -> None:
    for index, value in enumerate(lines):
        c.text(x, y + index * gap, value, size, fill, bold=bold_first and index == 0)


def render_title_scope_boundary() -> tuple[Path, Path]:
    c = SlideCanvas(
        "Q&A / TITLE AND EVIDENCE",
        "The registered scope was broader than the final evaluated scope",
        header_size=56,
        section_size=34,
    )

    # Title evolution: broad registered intent to narrower submitted evidence.
    c.rect(70, 245, 590, 440, LIGHT_RED, stroke=RED, width=2, radius=7)
    panel_header(c, 70, 245, 590, "APRIL / REGISTERED SCOPE", RED, size=21)
    c.text(330, 314, "BROAD INTENDED TITLE", 17, RED, bold=True, anchor="mm")
    for y, label in [
        (350, "Enhanced Few-Shot Open-Set Keyword Spotting"),
        (385, "with Noise-Robust Prototype Classification"),
        (420, "and Real-Time Streaming"),
    ]:
        c.circle(112, y, 7, RED)
        c.text(132, y, label, 18, NAVY, bold=True)

    c.line(625, 337, 815, 337, BLUE, 4, arrow=True)
    c.text(720, 307, "EVIDENCE", 16, MUTED, bold=True, anchor="mm")
    c.text(720, 371, "APRIL -> JULY", 16, BLUE, bold=True, anchor="mm")

    c.rect(850, 245, 1370, 440, LIGHT_BLUE, stroke=BLUE, width=2, radius=7)
    panel_header(c, 850, 245, 1370, "JULY / SUBMITTED THESIS", BLUE, size=21)
    c.text(1110, 313, "NARROWER EVALUATED TITLE", 17, BLUE, bold=True, anchor="mm")
    c.text(1110, 352, "Few-Shot Open-Set Keyword Spotting", 23, NAVY, bold=True, anchor="mm")
    c.text(1110, 383, "at Vocabulary Scale", 23, NAVY, bold=True, anchor="mm")
    c.text(1110, 409, "A Metric-Learning Study of Feature Front-Ends, Encoders,", 14, MUTED, anchor="mm")
    c.text(1110, 427, "and Open-Set Rejection", 14, MUTED, anchor="mm")

    # Three visual evidence cards replace the dense scope table.
    evidence_cards = [
        (70, 465, 470, 705, NAVY, "VOCABULARY SCALE"),
        (520, 465, 920, 705, BLUE, "NOISE ROBUSTNESS"),
        (970, 465, 1370, 705, NAVY, "REAL-TIME STREAMING"),
    ]
    for x1, y1, x2, y2, color, title in evidence_cards:
        c.rect(x1, y1, x2, y2, WHITE, stroke=GRID, width=2, radius=7)
        panel_header(c, x1, y1, x2, title, color, size=20)

    # Strong evidence: all three experiment scales were reached.
    c.line(135, 555, 405, 555, BLUE, 4)
    for x, short in [(135, "MICRO"), (270, "TOP500"), (405, "FULL")]:
        c.circle(x, 555, 29, LIGHT_BLUE, stroke=BLUE, width=3)
        c.circle(x, 555, 10, NAVY)
        c.text(x, 600, short, 15, NAVY, bold=True, anchor="mm")
    c.rect(135, 625, 405, 658, NAVY, radius=5)
    c.text(270, 642, "STRONG EVIDENCE", 17, WHITE, bold=True, anchor="mm")
    c.text(270, 681, "16 controlled pipelines", 17, MUTED, anchor="mm")

    # Partial noise evidence: implemented components, missing final controlled benchmark.
    c.waveform(555, 650, 555, BLUE)
    c.line(665, 555, 695, 555, BLUE, 3, arrow=True)
    c.rect(705, 525, 835, 585, LIGHT_BLUE, stroke=BLUE, width=2, radius=6)
    c.text(770, 555, "PCEN", 21, NAVY, bold=True, anchor="mm")
    c.rect(585, 625, 855, 658, BLUE, radius=5)
    c.text(720, 642, "PARTIAL EVIDENCE", 17, WHITE, bold=True, anchor="mm")
    c.text(720, 679, "No final DEMAND / SNR benchmark", 15, RED, bold=True, anchor="mm")

    # Streaming evidence: working implementation without a field benchmark.
    for offset in (0, 24, 48):
        c.rect(1015 + offset, 525, 1125 + offset, 585, ROW_ALT, stroke=BLUE, width=2, radius=5)
        c.waveform(1030 + offset, 1110 + offset, 555, BLUE)
    c.line(1190, 555, 1225, 555, BLUE, 3, arrow=True)
    c.circle(1265, 555, 28, LIGHT_BLUE, stroke=BLUE, width=3)
    c.circle(1265, 555, 10, NAVY)
    c.rect(1035, 625, 1305, 658, NAVY, radius=5)
    c.text(1170, 642, "WORKING PROTOTYPE", 17, WHITE, bold=True, anchor="mm")
    c.text(1170, 679, "No field FA/hour benchmark", 15, RED, bold=True, anchor="mm")

    # Honest one-sentence defense answer.
    c.rect(70, 730, 1370, 790, NAVY, radius=6)
    c.text(720, 750, "FINAL THESIS TITLE = STRONGEST COMPLETED EVIDENCE", 21, WHITE, bold=True, anchor="mm")
    c.text(720, 776, "The registration title was not updated in time.", 16, "#FFD6D9", bold=True, anchor="mm")
    return c.save("qa_title_scope_boundary")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        *render_prototype_threshold_margin(),
        *render_training_enrollment_inference(),
        *render_title_scope_boundary(),
    ]
    for output in outputs:
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
