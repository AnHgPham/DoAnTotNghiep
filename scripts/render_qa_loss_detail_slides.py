"""Render detailed Q&A backup slides for Triplet, SCAF, and GE2E."""

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


def chip(
    canvas: SlideCanvas,
    x1: float,
    x2: float,
    title: str,
    detail: str,
    *,
    color: str = NAVY,
) -> None:
    canvas.rect(x1, 245, x2, 302, ROW_ALT, stroke=GRID, width=1.5, radius=6)
    canvas.text((x1 + x2) / 2, 263, title, 16, color, bold=True, anchor="mm")
    canvas.text((x1 + x2) / 2, 285, detail, 14, MUTED, anchor="mm")


def footer(canvas: SlideCanvas, items: list[tuple[str, str]]) -> None:
    x1 = 70
    width = 1300 / len(items)
    for index, (title, detail) in enumerate(items):
        left = x1 + index * width
        right = left + width
        canvas.rect(left, 720, right, 770, NAVY if index % 2 == 0 else BLUE)
        canvas.text((left + right) / 2, 737, title, 14, WHITE, bold=True, anchor="mm")
        canvas.text((left + right) / 2, 757, detail, 13, WHITE, anchor="mm")


def clip_row(
    canvas: SlideCanvas,
    x: float,
    y: float,
    count: int,
    color: str,
    *,
    spacing: float = 25,
) -> None:
    for index in range(count):
        canvas.circle(x + index * spacing, y, 7, color)


def embedding(canvas: SlideCanvas, x: float, y: float, label: str, color: str) -> None:
    canvas.circle(x, y, 17, color)
    canvas.text(x, y, label, 13, WHITE, bold=True, anchor="mm")


def step_box(
    canvas: SlideCanvas,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    title: str,
    detail: str,
    *,
    color: str = BLUE,
    fill: str = WHITE,
) -> None:
    canvas.rect(x1, y1, x2, y2, fill, stroke=color, width=2, radius=6)
    canvas.text((x1 + x2) / 2, y1 + 21, title, 16, NAVY, bold=True, anchor="mm")
    canvas.text((x1 + x2) / 2, y2 - 18, detail, 13, MUTED, anchor="mm")


def render_triplet() -> tuple[object, object]:
    c = SlideCanvas(
        "Q&A / TRAINING OBJECTIVES",
        "1. Triplet Loss — How one triplet is mined",
        header_size=54,
        section_size=40,
    )
    chip(c, 70, 450, "EPISODE", "30 words × 10 clips")
    chip(c, 470, 850, "ENCODER PASS", "300 normalized embeddings", color=BLUE)
    chip(c, 870, 1370, "SAMPLE ROLES", "No support/query and no prototype")

    # Candidate pool.
    c.rect(70, 325, 420, 695, WHITE, stroke=GRID, width=2, radius=7)
    panel_header(c, 70, 325, 420, "A. CANDIDATE POOL", NAVY, size=21)
    c.text(245, 402, "ONE CLIP BECOMES THE ANCHOR", 15, NAVY, bold=True, anchor="mm")
    embedding(c, 245, 445, "A", NAVY)
    c.text(245, 475, 'anchor label: "yes"', 14, MUTED, anchor="mm")

    c.text(110, 520, "SAME WORD", 14, BLUE, bold=True)
    clip_row(c, 115, 552, 9, BLUE, spacing=27)
    c.text(110, 580, "9 positive candidates", 14, MUTED)

    c.text(110, 620, "OTHER WORDS", 14, RED, bold=True)
    clip_row(c, 115, 652, 10, RED, spacing=24)
    c.text(110, 680, "290 negative candidates", 14, MUTED)

    # Mining logic.
    c.rect(450, 325, 995, 695, WHITE, stroke=GRID, width=2, radius=7)
    panel_header(c, 450, 325, 995, "B. SELECT THE USEFUL PAIR", BLUE, size=21)
    c.text(722, 395, "DISTANCE FROM ANCHOR", 15, NAVY, bold=True, anchor="mm")
    c.line(505, 455, 940, 455, GRID, 6, arrow=True)
    embedding(c, 535, 455, "A", NAVY)
    embedding(c, 650, 455, "P", BLUE)
    embedding(c, 835, 455, "N", RED)
    c.text(535, 492, "ANCHOR", 13, NAVY, bold=True, anchor="mm")
    c.text(650, 492, "HARDEST POSITIVE", 13, BLUE, bold=True, anchor="mm")
    c.text(835, 492, "SEMI-HARD NEGATIVE", 13, RED, bold=True, anchor="mm")

    c.rect(630, 525, 930, 585, LIGHT_RED, stroke=RED, width=1.5, radius=5)
    c.text(780, 545, "USEFUL NEGATIVE ZONE", 14, RED, bold=True, anchor="mm")
    c.text(780, 568, "farther than P, but not far enough", 13, MUTED, anchor="mm")
    c.line(650, 507, 650, 525, BLUE, 2)
    c.line(905, 507, 905, 525, RED, 2)

    c.rect(495, 610, 950, 670, ROW_ALT, stroke=GRID, radius=5)
    c.text(515, 630, "HARDEST POSITIVE", 14, NAVY, bold=True)
    c.text(705, 630, "farthest clip with the same label", 14, MUTED)
    c.text(515, 653, "FALLBACK", 14, RED, bold=True)
    c.text(605, 653, "if no semi-hard exists, use the closest wrong-word clip", 14, MUTED)

    # Update path.
    c.rect(1025, 325, 1370, 695, WHITE, stroke=GRID, width=2, radius=7)
    panel_header(c, 1025, 325, 1370, "C. OPTIMIZE", NAVY, size=21)
    embedding(c, 1110, 425, "A", NAVY)
    embedding(c, 1190, 425, "P", BLUE)
    embedding(c, 1280, 485, "N", RED)
    c.line(1170, 425, 1132, 425, BLUE, 3, arrow=True)
    c.line(1205, 450, 1255, 478, RED, 3, arrow=True)
    c.text(1150, 392, "PULL TOGETHER", 13, BLUE, bold=True, anchor="mm")
    c.text(1225, 512, "PUSH APART", 13, RED, bold=True, anchor="mm")

    step_box(c, 1080, 535, 1315, 590, "TRIPLET LOSS", "one scalar per selected triplet", color=RED)
    c.line(1197, 590, 1197, 612, BLUE, 3, arrow=True)
    step_box(c, 1080, 620, 1315, 675, "BACKWARD", "update the shared encoder", color=BLUE, fill=LIGHT_BLUE)

    footer(
        c,
        [
            ("POSITIVE", "another clip of the same word"),
            ("NEGATIVE", "a clip from a different word"),
            ("TRAINING MARGIN", "not the demo top-2 distance gap"),
        ],
    )
    return c.save("qa_loss_detail_triplet")


def subcenters(c: SlideCanvas, x: float, y: float, color: str, label: str) -> None:
    c.text(x, y - 48, label, 14, NAVY, bold=True, anchor="mm")
    for dx, dy in [(-36, 8), (0, -4), (36, 12)]:
        c.circle(x + dx, y + dy, 16, WHITE, stroke=color, width=3)
        c.circle(x + dx, y + dy, 6, color)


def render_scaf() -> tuple[object, object]:
    c = SlideCanvas(
        "Q&A / TRAINING OBJECTIVES",
        "2. SCAF — Learnable sub-centers and angular margin",
        header_size=54,
        section_size=40,
    )
    chip(c, 70, 450, "EPISODE", "300 embeddings; no support/query")
    chip(c, 470, 850, "GLOBAL HEAD", "3 learned sub-centers per word", color=BLUE)
    chip(c, 870, 1370, "MEMORY", "the same centers persist across episodes")

    # Persistent learned centers.
    c.rect(70, 325, 450, 695, WHITE, stroke=GRID, width=2, radius=7)
    panel_header(c, 70, 325, 450, "A. LEARNABLE CLASS MEMORY", NAVY, size=20)
    subcenters(c, 180, 435, BLUE, 'WORD "YES"')
    subcenters(c, 340, 435, RED, 'WORD "NO"')
    subcenters(c, 180, 555, NAVY, 'WORD "STOP"')
    subcenters(c, 340, 555, MUTED, "OTHER WORD")
    c.rect(105, 620, 415, 675, LIGHT_RED, stroke=RED, width=1.5, radius=5)
    c.text(260, 640, "FULL-VOCABULARY HEAD", 15, RED, bold=True, anchor="mm")
    c.text(260, 661, "37,387 words create 112,161 learned vectors", 14, MUTED, anchor="mm")

    # Per-embedding scoring path.
    c.rect(480, 325, 1000, 695, WHITE, stroke=GRID, width=2, radius=7)
    panel_header(c, 480, 325, 1000, "B. SCORE ONE EMBEDDING", BLUE, size=20)
    embedding(c, 535, 425, "e", NAVY)
    c.text(535, 459, "normalized", 13, MUTED, anchor="mm")
    c.line(560, 425, 625, 425, BLUE, 3, arrow=True)
    step_box(c, 630, 385, 765, 465, "ALL CENTERS", "cosine scores", color=BLUE, fill=LIGHT_BLUE)
    c.line(770, 425, 805, 425, BLUE, 3, arrow=True)
    step_box(c, 810, 385, 950, 465, "MAX OF 3", "one score per class", color=NAVY)

    c.line(880, 465, 880, 505, BLUE, 3, arrow=True)
    c.rect(570, 515, 950, 575, ROW_ALT, stroke=GRID, radius=5)
    c.text(590, 535, "CORRECT CLASS", 14, NAVY, bold=True)
    c.text(735, 535, "apply angular margin", 14, RED, bold=True)
    c.text(590, 558, "OTHER CLASSES", 14, NAVY, bold=True)
    c.text(735, 558, "remain competitors", 14, MUTED)

    c.line(760, 575, 760, 603, BLUE, 3, arrow=True)
    step_box(c, 625, 610, 895, 670, "CROSS-ENTROPY", "correct class must rank first", color=RED)

    # Updates and meaning.
    c.rect(1030, 325, 1370, 695, WHITE, stroke=GRID, width=2, radius=7)
    panel_header(c, 1030, 325, 1370, "C. BACKPROP", NAVY, size=20)
    step_box(c, 1080, 385, 1320, 445, "SCAF LOSS", "one scalar for the batch", color=RED)
    c.line(1200, 445, 1200, 475, BLUE, 3, arrow=True)
    c.rect(1080, 485, 1320, 565, LIGHT_BLUE, stroke=BLUE, width=2, radius=6)
    c.text(1200, 508, "UPDATED", 15, BLUE, bold=True, anchor="mm")
    c.text(1200, 533, "encoder weights", 14, NAVY, anchor="mm")
    c.text(1200, 553, "all selected sub-centers", 14, NAVY, anchor="mm")

    c.rect(1080, 590, 1320, 670, LIGHT_RED, stroke=RED, width=1.5, radius=6)
    c.text(1200, 613, "NOT A PROTOTYPE", 15, RED, bold=True, anchor="mm")
    c.text(1200, 638, "not averaged from support clips", 14, MUTED, anchor="mm")
    c.text(1200, 658, "not used at enrollment", 14, MUTED, anchor="mm")

    footer(
        c,
        [
            ("TARGET", "best sub-center of the true class"),
            ("COMPETITORS", "scores from all other classes"),
            ("MAIN RISK", "a very large head at full vocabulary"),
        ],
    )
    return c.save("qa_loss_detail_scaf")


def centroid(c: SlideCanvas, x: float, y: float, color: str, label: str) -> None:
    c.circle(x, y, 24, LIGHT_BLUE, stroke=color, width=3)
    c.circle(x, y, 9, color)
    c.text(x, y + 39, label, 13, NAVY, bold=True, anchor="mm")


def render_ge2e() -> tuple[object, object]:
    c = SlideCanvas(
        "Q&A / TRAINING OBJECTIVES",
        "3. GE2E — Support centroids classify episode queries",
        header_size=54,
        section_size=40,
    )
    chip(c, 70, 450, "EPISODE", "30 words × 10 clips")
    chip(c, 470, 850, "PER WORD", "5 support + 5 query", color=BLUE)
    chip(c, 870, 1370, "TOTAL", "30 temporary centroids + 150 queries")

    # Per-class split and centroid formation.
    c.rect(70, 325, 445, 695, WHITE, stroke=GRID, width=2, radius=7)
    panel_header(c, 70, 325, 445, "A. SPLIT EACH WORD", NAVY, size=20)
    c.text(257, 398, '10 CLIPS OF "YES"', 16, NAVY, bold=True, anchor="mm")
    clip_row(c, 122, 438, 10, NAVY, spacing=30)
    c.line(257, 465, 170, 505, BLUE, 3, arrow=True)
    c.line(257, 465, 345, 505, RED, 3, arrow=True)

    c.text(170, 532, "SUPPORT", 14, BLUE, bold=True, anchor="mm")
    clip_row(c, 120, 562, 5, BLUE, spacing=25)
    c.text(345, 532, "QUERY", 14, RED, bold=True, anchor="mm")
    clip_row(c, 295, 562, 5, RED, spacing=25)

    c.line(170, 582, 170, 610, BLUE, 3, arrow=True)
    centroid(c, 170, 640, BLUE, "TEMPORARY CENTROID")
    c.text(345, 620, "kept as", 13, MUTED, anchor="mm")
    c.text(345, 643, "5 query embeddings", 14, RED, bold=True, anchor="mm")

    # Query classification against episode centroids.
    c.rect(475, 325, 1015, 695, WHITE, stroke=GRID, width=2, radius=7)
    panel_header(c, 475, 325, 1015, "B. CLASSIFY EVERY QUERY", BLUE, size=20)
    embedding(c, 560, 475, "q", RED)
    c.text(560, 510, 'query: "yes"', 14, RED, bold=True, anchor="mm")

    centroid(c, 770, 405, BLUE, "YES / TARGET")
    centroid(c, 880, 500, RED, "NO / COMPETITOR")
    centroid(c, 770, 570, MUTED, "STOP / COMPETITOR")
    c.line(580, 465, 740, 415, BLUE, 4, arrow=True)
    c.line(580, 480, 850, 497, RED, 2, arrow=True)
    c.line(580, 490, 740, 560, MUTED, 2, arrow=True)
    c.text(660, 414, "highest score", 14, BLUE, bold=True, anchor="mm")
    c.text(740, 520, "other 29 centroids compete", 14, MUTED, anchor="mm")

    c.rect(625, 635, 950, 678, LIGHT_BLUE, stroke=BLUE, width=2, radius=5)
    c.text(787, 657, "COSINE SCORES, THEN CROSS-ENTROPY", 15, NAVY, bold=True, anchor="mm")

    # Backprop and deployment relationship.
    c.rect(1045, 325, 1370, 695, WHITE, stroke=GRID, width=2, radius=7)
    panel_header(c, 1045, 325, 1370, "C. LEARN & DEPLOY", NAVY, size=20)
    step_box(c, 1090, 385, 1325, 440, "GE2E LOSS", "no explicit margin", color=RED)
    c.line(1207, 440, 1207, 470, BLUE, 3, arrow=True)
    step_box(c, 1090, 480, 1325, 550, "BACKWARD", "through query and support mean", color=BLUE, fill=LIGHT_BLUE)
    c.line(1207, 550, 1207, 578, BLUE, 3, arrow=True)

    c.rect(1080, 590, 1335, 675, ROW_ALT, stroke=GRID, width=1.5, radius=6)
    c.text(1207, 612, "MATCHES DEPLOYMENT", 15, NAVY, bold=True, anchor="mm")
    c.text(1207, 637, "support clips become a prototype", 14, BLUE, bold=True, anchor="mm")
    c.text(1207, 658, "query selects the nearest prototype", 14, RED, bold=True, anchor="mm")

    footer(
        c,
        [
            ("QUERY ROLE", "similar to an anchor"),
            ("OWN CENTROID", "the positive target"),
            ("OTHER CENTROIDS", "episode-level competitors"),
            ("CENTROID", "recomputed every episode"),
        ],
    )
    return c.save("qa_loss_detail_ge2e")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [*render_triplet(), *render_scaf(), *render_ge2e()]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
