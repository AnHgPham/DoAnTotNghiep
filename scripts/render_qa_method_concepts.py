"""Render the three main methodology slides for the thesis defense."""

from __future__ import annotations

try:
    from scripts.render_qa_priority_slides import (
        ASSET_DIR,
        BLUE,
        GRID,
        LIGHT_BLUE,
        LIGHT_RED,
        MUTED,
        NAVY,
        RED,
        ROW_ALT,
        WHITE,
        SlideCanvas,
        panel_header,
    )
except ModuleNotFoundError:
    from render_qa_priority_slides import (  # type: ignore[no-redef]
        ASSET_DIR,
        BLUE,
        GRID,
        LIGHT_BLUE,
        LIGHT_RED,
        MUTED,
        NAVY,
        RED,
        ROW_ALT,
        WHITE,
        SlideCanvas,
        panel_header,
    )


def heatmap(
    canvas: SlideCanvas,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    rows: int,
    cols: int,
    variant: int,
) -> None:
    """Draw a deterministic stylized time-frequency map."""
    palette = ["#EEF5FF", "#C8DBFF", "#78A9F6", BLUE, NAVY, "#F88D91", RED]
    canvas.rect(x1, y1, x2, y2, WHITE, stroke=GRID, width=1.5, radius=4)
    cell_w = (x2 - x1 - 8) / cols
    cell_h = (y2 - y1 - 8) / rows
    for row in range(rows):
        for col in range(cols):
            ridge = int(2.4 * abs((col % 9) - 4))
            energy = (row * 3 + col * 5 + variant * 7 + ridge) % len(palette)
            if variant == 1:
                energy = min(5, max(1, energy))
            elif variant == 2:
                energy = min(4, max(0, (energy + row) // 2))
            canvas.rect(
                x1 + 4 + col * cell_w,
                y1 + 4 + row * cell_h,
                x1 + 4 + (col + 1) * cell_w + 0.4,
                y1 + 4 + (row + 1) * cell_h + 0.4,
                palette[energy],
            )


def status_badge(
    canvas: SlideCanvas,
    x1: float,
    y1: float,
    x2: float,
    text: str,
    color: str,
    *,
    size: int = 15,
) -> None:
    canvas.rect(x1, y1, x2, y1 + 34, color, radius=5)
    canvas.text((x1 + x2) / 2, y1 + 17, text, size, WHITE, bold=True, anchor="mm")


def cluster(
    canvas: SlideCanvas,
    cx: float,
    cy: float,
    color: str,
    *,
    center: bool = False,
) -> None:
    for dx, dy in [(-24, -12), (-9, 19), (13, -20), (25, 12), (-28, 25)]:
        canvas.circle(cx + dx, cy + dy, 6, color)
    if center:
        canvas.circle(cx, cy, 15, WHITE, stroke=NAVY, width=3)
        canvas.circle(cx, cy, 6, NAVY)


def render_losses() -> tuple[object, object]:
    c = SlideCanvas(
        "III/ METHODOLOGY",
        "E. Training Objectives",
        header_size=68,
        section_size=48,
    )
    c.text(
        720,
        248,
        "EACH OBJECTIVE TRAINS THE SAME ENCODER IN A DIFFERENT WAY",
        20,
        NAVY,
        bold=True,
        anchor="mm",
    )

    panels = [
        (70, 270, 690, 500, NAVY, "TRIPLET"),
        (750, 270, 1370, 500, BLUE, "GE2E"),
        (70, 535, 690, 765, NAVY, "SCAF"),
        (750, 535, 1370, 765, BLUE, "SCAF + GE2E HYBRID"),
    ]
    for x1, y1, x2, y2, color, title in panels:
        c.rect(x1, y1, x2, y2, WHITE, stroke=GRID, width=2, radius=7)
        panel_header(c, x1, y1, x2, title, color, size=22)

    # Triplet: one anchor, one positive, and one mined negative.
    c.text(380, 340, "PULL POSITIVE CLOSER, PUSH NEGATIVE FARTHER", 15, NAVY, bold=True, anchor="mm")
    c.text(165, 376, "BEFORE", 13, MUTED, bold=True, anchor="mm")
    c.circle(145, 420, 11, NAVY)
    c.text(145, 420, "A", 11, WHITE, bold=True, anchor="mm")
    c.circle(205, 385, 11, BLUE)
    c.text(205, 385, "P", 11, WHITE, bold=True, anchor="mm")
    c.circle(205, 455, 11, RED)
    c.text(205, 455, "N", 11, WHITE, bold=True, anchor="mm")
    c.line(145, 420, 205, 385, BLUE, 2)
    c.line(145, 420, 205, 455, RED, 2)
    c.line(285, 420, 345, 420, BLUE, 3, arrow=True)
    c.text(315, 398, "OPTIMIZE", 12, BLUE, bold=True, anchor="mm")
    c.text(530, 376, "AFTER", 13, MUTED, bold=True, anchor="mm")
    c.circle(510, 420, 44, LIGHT_BLUE, stroke=BLUE, width=2)
    c.circle(510, 420, 11, NAVY)
    c.text(510, 420, "A", 11, WHITE, bold=True, anchor="mm")
    c.circle(535, 397, 11, BLUE)
    c.text(535, 397, "P", 11, WHITE, bold=True, anchor="mm")
    c.circle(610, 465, 11, RED)
    c.text(610, 465, "N", 11, WHITE, bold=True, anchor="mm")
    c.line(510, 420, 535, 397, BLUE, 3)
    c.line(510, 420, 610, 465, RED, 3)
    status_badge(c, 210, 456, 550, "PAIRWISE RANKING", NAVY, size=13)

    # GE2E: support samples form centroids; queries classify against centroids.
    c.text(1060, 340, "5 SUPPORT CLIPS FORM A CENTROID", 15, NAVY, bold=True, anchor="mm")
    c.text(860, 378, "SUPPORT", 13, BLUE, bold=True, anchor="mm")
    for index in range(5):
        c.circle(825 + (index % 3) * 25, 405 + (index // 3) * 25, 7, BLUE)
    c.line(900, 420, 960, 420, BLUE, 3, arrow=True)
    c.circle(1000, 420, 22, LIGHT_BLUE, stroke=BLUE, width=3)
    c.circle(1000, 420, 8, NAVY)
    c.text(1000, 449, "CENTROID", 13, NAVY, bold=True, anchor="mm")

    c.text(1195, 378, "QUERY", 13, RED, bold=True, anchor="mm")
    for index in range(5):
        c.circle(1160 + (index % 3) * 25, 405 + (index // 3) * 25, 7, RED)
    c.line(1185, 470, 1015, 433, BLUE, 2.5, arrow=True)
    c.circle(1185, 470, 11, RED)
    c.text(1060, 470, "QUERY -> NEAREST CENTROID", 15, NAVY, bold=True, anchor="mm")
    status_badge(c, 870, 462, 1250, "EPISODIC CENTROID CLASSIFICATION", BLUE, size=13)

    # SCAF: several learnable sub-centers represent each class variation.
    c.text(380, 605, "K = 3 SUB-CENTERS PER CLASS", 15, NAVY, bold=True, anchor="mm")
    c.rect(125, 625, 635, 710, ROW_ALT, stroke=GRID, radius=6)
    for cx, cy in [(195, 658), (250, 680), (195, 700)]:
        cluster(c, cx, cy, BLUE, center=True)
    for cx, cy in [(560, 658), (505, 680), (560, 700)]:
        cluster(c, cx, cy, RED, center=True)
    c.line(380, 635, 380, 705, RED, 4)
    c.line(362, 647, 362, 693, BLUE, 2)
    c.text(380, 728, "ANGULAR MARGIN + SUB-CENTERS", 15, RED, bold=True, anchor="mm")
    status_badge(c, 210, 746, 550, "WITHIN-CLASS VARIATION", NAVY, size=13)

    # Hybrid: the same embedding receives both SCAF and GE2E loss signals.
    c.text(1060, 605, "ONE SHARED EMBEDDING, TWO TRAINING SIGNALS", 15, NAVY, bold=True, anchor="mm")
    c.text(840, 655, "SHARED", 13, BLUE, bold=True, anchor="mm")
    c.text(840, 676, "EMBEDDING", 13, BLUE, bold=True, anchor="mm")
    for dx, dy, color in [(-18, -8, NAVY), (0, 8, BLUE), (18, -8, RED)]:
        c.circle(945 + dx, 666 + dy, 7, color)
    c.line(975, 658, 1060, 642, BLUE, 2.5, arrow=True)
    c.line(975, 674, 1060, 692, BLUE, 2.5, arrow=True)
    c.rect(1065, 629, 1190, 655, WHITE, stroke=NAVY, width=2, radius=5)
    c.text(1128, 642, "SCAF LOSS", 12, NAVY, bold=True, anchor="mm")
    c.rect(1065, 679, 1190, 705, WHITE, stroke=BLUE, width=2, radius=5)
    c.text(1128, 692, "GE2E LOSS", 12, BLUE, bold=True, anchor="mm")
    c.line(1195, 642, 1245, 665, BLUE, 2.3, arrow=True)
    c.line(1195, 692, 1245, 671, BLUE, 2.3, arrow=True)
    c.circle(1278, 667, 25, WHITE, stroke=BLUE, width=3)
    c.text(1278, 660, "SUM", 13, NAVY, bold=True, anchor="mm")
    c.text(1278, 679, "L", 15, RED, bold=True, anchor="mm")
    c.text(1060, 732, "FINAL LOSS UPDATES THE SAME ENCODER", 15, NAVY, bold=True, anchor="mm")
    status_badge(c, 890, 746, 1230, "JOINT OBJECTIVE", BLUE, size=13)
    return c.save("slide11_training_objectives_main")


def architecture_block(
    c: SlideCanvas,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    title: str,
    detail: str,
    color: str = BLUE,
) -> None:
    c.rect(x1, y1, x2, y2, ROW_ALT, stroke=color, width=2, radius=6)
    c.text((x1 + x2) / 2, y1 + 23, title, 16, NAVY, bold=True, anchor="mm")
    c.text((x1 + x2) / 2, y2 - 17, detail, 13, MUTED, anchor="mm")


def render_encoders() -> tuple[object, object]:
    c = SlideCanvas(
        "III/ METHODOLOGY",
        "D. Encoder Architectures",
        header_size=68,
        section_size=48,
    )

    c.rect(220, 245, 1220, 300, LIGHT_BLUE, stroke=BLUE, width=2, radius=6)
    c.text(
        720,
        273,
        "MODEL INPUT: MFCC 1 x 47 x 10  OR  PCEN 1 x 40 x 101",
        20,
        NAVY,
        bold=True,
        anchor="mm",
    )
    c.line(500, 300, 380, 335, BLUE, 3, arrow=True)
    c.line(940, 300, 1060, 335, BLUE, 3, arrow=True)

    c.rect(70, 325, 690, 680, WHITE, stroke=GRID, width=2, radius=7)
    panel_header(c, 70, 325, 690, "DSCNN-L", NAVY, size=25)
    c.rect(750, 325, 1370, 680, WHITE, stroke=GRID, width=2, radius=7)
    panel_header(c, 750, 325, 1370, "EDGESPOTFULL T4", BLUE, size=25)

    # DSCNN-L: a simple high-capacity depthwise-separable stack.
    heatmap(c, 95, 410, 175, 505, rows=6, cols=8, variant=0)
    c.text(135, 535, "INPUT", 14, MUTED, bold=True, anchor="mm")
    c.line(185, 458, 215, 458, BLUE, 3, arrow=True)
    architecture_block(c, 220, 410, 330, 505, "INITIAL CONV", "10x4 / s2x1")
    c.line(340, 458, 365, 458, BLUE, 3, arrow=True)
    for index, height in enumerate([94, 86, 78, 70, 62]):
        x1 = 370 + index * 35
        c.rect(x1, 458 - height / 2, x1 + 24, 458 + height / 2, LIGHT_BLUE, stroke=BLUE, width=2, radius=3)
        c.text(x1 + 12, 458, str(index + 1), 12, NAVY, bold=True, anchor="mm")
    c.text(452, 535, "5 DEPTHWISE-SEPARABLE BLOCKS", 14, NAVY, bold=True, anchor="mm")
    c.line(560, 458, 585, 458, BLUE, 3, arrow=True)
    c.circle(615, 458, 28, LIGHT_BLUE, stroke=BLUE, width=3)
    c.text(615, 451, "AVG", 14, NAVY, bold=True, anchor="mm")
    c.text(615, 472, "POOL", 12, MUTED, bold=True, anchor="mm")
    c.text(380, 580, "276-D EMBEDDING", 21, NAVY, bold=True, anchor="mm")
    c.rect(105, 610, 655, 660, ROW_ALT, stroke=GRID, radius=5)
    c.text(125, 629, "412,896 params (MFCC)  |  412,900 params (PCEN)", 16, NAVY, bold=True)
    c.text(125, 650, "Higher-capacity embedding encoder", 15, MUTED, bold=True)

    # EdgeSpotFull T4: compact backbone plus explicit temporal modeling.
    heatmap(c, 775, 410, 855, 505, rows=6, cols=8, variant=1)
    c.text(815, 535, "INPUT", 14, MUTED, bold=True, anchor="mm")
    stages = [
        (875, 410, 955, 505, "STEM", "5x5"),
        (975, 400, 1070, 515, "FUSED", "temporal"),
        (1090, 390, 1185, 525, "BCRES", "dilated"),
        (1205, 410, 1300, 505, "ATTENTION", "1 head"),
    ]
    c.line(865, 458, 872, 458, BLUE, 3, arrow=True)
    for index, block in enumerate(stages):
        architecture_block(c, *block, color=BLUE)
        if index < len(stages) - 1:
            c.line(block[2] + 5, 458, stages[index + 1][0] - 5, 458, BLUE, 3, arrow=True)
    c.line(1305, 458, 1320, 458, BLUE, 3, arrow=True)
    c.circle(1335, 458, 23, LIGHT_BLUE, stroke=BLUE, width=3)
    c.text(1335, 458, "64", 14, NAVY, bold=True, anchor="mm")
    c.text(1060, 565, "64-D EMBEDDING", 19, NAVY, bold=True, anchor="mm")
    c.rect(785, 610, 1335, 660, ROW_ALT, stroke=GRID, radius=5)
    c.text(805, 629, "130,598 params  |  64-D embedding", 16, NAVY, bold=True)
    c.text(805, 650, "Compact encoder with explicit temporal modeling", 15, MUTED, bold=True)

    c.rect(190, 705, 1250, 752, NAVY, radius=6)
    c.text(455, 729, "DSCNN-L: 276-D, HIGHER CAPACITY", 18, WHITE, bold=True, anchor="mm")
    c.line(720, 705, 720, 752, WHITE, 1)
    c.text(985, 729, "EDGESPOT T4: ABOUT 3.2x FEWER PARAMETERS", 18, WHITE, bold=True, anchor="mm")
    return c.save("slide10_encoder_architectures_main")


def frontend_step(
    c: SlideCanvas,
    x1: float,
    x2: float,
    y: float,
    title: str,
    color: str,
) -> None:
    c.rect(x1, y, x2, y + 48, ROW_ALT, stroke=color, width=2, radius=5)
    c.text((x1 + x2) / 2, y + 24, title, 15, NAVY, bold=True, anchor="mm")


def render_frontends() -> tuple[object, object]:
    c = SlideCanvas(
        "III/ METHODOLOGY",
        "C. Feature Frontends",
        header_size=68,
        section_size=48,
    )

    c.text(180, 270, "16 kHz MONO AUDIO", 17, NAVY, bold=True, anchor="mm")
    for index in range(5):
        start = 285 + index * 174
        c.waveform(start, start + 174, 270, BLUE)
    c.line(720, 300, 430, 335, BLUE, 3, arrow=True)
    c.line(720, 300, 1010, 335, BLUE, 3, arrow=True)

    c.rect(70, 325, 690, 690, WHITE, stroke=GRID, width=2, radius=7)
    panel_header(c, 70, 325, 690, "MFCC", NAVY, size=26)
    c.rect(750, 325, 1370, 690, WHITE, stroke=GRID, width=2, radius=7)
    panel_header(c, 750, 325, 1370, "MEL + PCEN", BLUE, size=26)

    # MFCC pipeline: fixed log compression and DCT, then retain 10 coefficients.
    mfcc_steps = [
        (95, 178, "STFT"),
        (195, 290, "MEL 40"),
        (307, 390, "LOG"),
        (407, 490, "DCT"),
        (507, 645, "KEEP FIRST 10"),
    ]
    for index, (x1, x2, title) in enumerate(mfcc_steps):
        frontend_step(c, x1, x2, 400, title, NAVY)
        if index < len(mfcc_steps) - 1:
            c.line(x2 + 4, 424, mfcc_steps[index + 1][0] - 5, 424, BLUE, 2.5, arrow=True)
    heatmap(c, 150, 485, 610, 585, rows=5, cols=18, variant=2)
    c.text(380, 610, "MODEL INPUT  1 x 47 x 10", 21, NAVY, bold=True, anchor="mm")
    c.text(380, 640, "40 ms window  |  20 ms hop", 16, MUTED, anchor="mm")
    status_badge(c, 170, 655, 590, "FIXED, COMPACT BASELINE", NAVY)

    # PCEN pipeline: raw mel energy, causal smoothing, AGC, and root compression.
    pcen_steps = [
        (775, 858, "STFT"),
        (875, 970, "MEL 40"),
        (987, 1095, "SMOOTH M(t)"),
        (1112, 1200, "AGC"),
        (1217, 1345, "ROOT"),
    ]
    for index, (x1, x2, title) in enumerate(pcen_steps):
        frontend_step(c, x1, x2, 400, title, BLUE)
        if index < len(pcen_steps) - 1:
            c.line(x2 + 4, 424, pcen_steps[index + 1][0] - 5, 424, BLUE, 2.5, arrow=True)
    heatmap(c, 810, 485, 1310, 585, rows=8, cols=25, variant=1)
    c.text(1060, 610, "MODEL INPUT  1 x 40 x 101", 21, NAVY, bold=True, anchor="mm")
    c.text(1060, 640, "25 ms window  |  10 ms hop", 16, MUTED, anchor="mm")
    status_badge(c, 850, 655, 1270, "TRAINABLE ADAPTIVE COMPRESSION", BLUE)

    c.rect(190, 710, 1250, 752, LIGHT_BLUE, stroke=BLUE, width=2, radius=6)
    c.text(
        720,
        731,
        "PCEN USES RAW MEL ENERGY AND REPLACES STATIC LOG COMPRESSION",
        19,
        NAVY,
        bold=True,
        anchor="mm",
    )
    return c.save("slide09_feature_frontends_main")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        *render_frontends(),
        *render_encoders(),
        *render_losses(),
    ]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
