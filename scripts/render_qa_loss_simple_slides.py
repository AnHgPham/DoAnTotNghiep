"""Render plain-language Q&A slides for the three training losses."""

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


def card(
    canvas: SlideCanvas,
    x1: float,
    x2: float,
    title: str,
    line1: str,
    line2: str,
    *,
    color: str,
) -> None:
    canvas.rect(x1, 315, x2, 590, WHITE, stroke=GRID, width=2, radius=7)
    panel_header(canvas, x1, 315, x2, title, color, size=22)
    canvas.text((x1 + x2) / 2, 522, line1, 18, NAVY, bold=True, anchor="mm")
    canvas.text((x1 + x2) / 2, 551, line2, 15, MUTED, anchor="mm")


def arrow_between(canvas: SlideCanvas, left: float, right: float) -> None:
    canvas.line(left, 452, right, 452, BLUE, 4, arrow=True)


def bottom_message(canvas: SlideCanvas, title: str, detail: str) -> None:
    canvas.rect(150, 625, 1290, 700, LIGHT_BLUE, stroke=BLUE, width=2, radius=7)
    canvas.text(720, 650, title, 20, NAVY, bold=True, anchor="mm")
    canvas.text(720, 678, detail, 16, MUTED, anchor="mm")
    canvas.rect(150, 720, 1290, 766, NAVY, radius=6)


def waveform(canvas: SlideCanvas, x: float, y: float, color: str) -> None:
    canvas.waveform(x - 55, x + 55, y, color)


def render_triplet() -> tuple[object, object]:
    c = SlideCanvas(
        "Q&A / HOW THE MODEL LEARNS",
        "1. Triplet — Learn from three recordings",
        header_size=54,
        section_size=40,
    )
    c.text(
        720,
        268,
        "KEEP THE SAME WORD CLOSE AND DIFFERENT WORDS FAR APART",
        21,
        NAVY,
        bold=True,
        anchor="mm",
    )

    card(c, 70, 420, "REFERENCE", 'one recording of "yes"', "used as the starting point", color=NAVY)
    card(c, 545, 895, "SAME WORD", 'another recording of "yes"', "move it closer", color=BLUE)
    card(c, 1020, 1370, "DIFFERENT WORD", 'a recording of "no"', "move it farther away", color=RED)

    waveform(c, 245, 420, NAVY)
    c.circle(245, 420, 18, NAVY)
    waveform(c, 720, 420, BLUE)
    c.circle(720, 420, 18, BLUE)
    waveform(c, 1195, 420, RED)
    c.circle(1195, 420, 18, RED)
    arrow_between(c, 430, 525)
    arrow_between(c, 905, 1000)

    bottom_message(
        c,
        "THE MODEL LEARNS WHERE EACH WORD SHOULD BE PLACED",
        "recordings of the same word form a close group; other words stay outside",
    )
    c.text(390, 743, "COMPARE THREE RECORDINGS", 15, WHITE, bold=True, anchor="mm")
    c.text(720, 743, "MOVE THE SAME WORD CLOSER", 15, WHITE, bold=True, anchor="mm")
    c.text(1050, 743, "MOVE A DIFFERENT WORD FARTHER", 15, WHITE, bold=True, anchor="mm")
    return c.save("qa_loss_simple_triplet")


def learned_representatives(
    canvas: SlideCanvas,
    x: float,
    y: float,
    color: str,
    word: str,
) -> None:
    canvas.text(x, y - 58, word, 16, NAVY, bold=True, anchor="mm")
    for dx, dy in [(-46, 16), (0, -6), (46, 16)]:
        canvas.circle(x + dx, y + dy, 19, WHITE, stroke=color, width=3)
        canvas.circle(x + dx, y + dy, 7, color)


def render_scaf() -> tuple[object, object]:
    c = SlideCanvas(
        "Q&A / HOW THE MODEL LEARNS",
        "2. SCAF — Learn several representatives for each word",
        header_size=54,
        section_size=40,
    )
    c.text(
        720,
        268,
        "ONE WORD CAN SOUND DIFFERENT ACROSS SPEAKERS AND CONDITIONS",
        21,
        NAVY,
        bold=True,
        anchor="mm",
    )

    card(c, 70, 420, "LEARN", "three representatives", "for every training word", color=NAVY)
    card(c, 545, 895, "COMPARE", "place each recording", "near the correct word group", color=BLUE)
    card(c, 1020, 1370, "UPDATE", "improve the audio map", "and the representatives", color=RED)

    learned_representatives(c, 245, 435, BLUE, 'WORD "YES"')
    learned_representatives(c, 720, 435, RED, 'WORD "NO"')
    c.circle(680, 414, 11, NAVY)
    c.line(680, 414, 704, 426, BLUE, 3, arrow=True)

    c.rect(1090, 390, 1300, 465, LIGHT_BLUE, stroke=BLUE, width=2, radius=6)
    c.text(1195, 414, "ENCODER", 17, NAVY, bold=True, anchor="mm")
    c.text(1195, 442, "+ WORD REPRESENTATIVES", 14, BLUE, bold=True, anchor="mm")
    arrow_between(c, 430, 525)
    arrow_between(c, 905, 1000)

    bottom_message(
        c,
        "WHY USE SEVERAL REPRESENTATIVES?",
        "they can cover different speakers, accents, and recording conditions for the same word",
    )
    c.text(390, 743, "LEARNED DURING TRAINING", 15, WHITE, bold=True, anchor="mm")
    c.text(720, 743, "SEVERAL FOR EACH WORD", 15, WHITE, bold=True, anchor="mm")
    c.text(1050, 743, "FINAL MATCHING USES PROTOTYPES", 15, WHITE, bold=True, anchor="mm")
    return c.save("qa_loss_simple_scaf")


def dot_row(canvas: SlideCanvas, x: float, y: float, color: str) -> None:
    for index in range(5):
        canvas.circle(x + index * 24, y, 7, color)


def render_ge2e() -> tuple[object, object]:
    c = SlideCanvas(
        "Q&A / HOW THE MODEL LEARNS",
        "3. GE2E — Build a word representative, then test it",
        header_size=54,
        section_size=40,
    )
    c.text(
        720,
        268,
        "EACH WORD HAS 10 RECORDINGS IN ONE TRAINING BATCH",
        21,
        NAVY,
        bold=True,
        anchor="mm",
    )

    card(c, 70, 420, "BUILD", "five example recordings", "create one word representative", color=NAVY)
    card(c, 545, 895, "TEST", "five remaining recordings", "find the correct representative", color=BLUE)
    card(c, 1020, 1370, "LEARN", "correct the audio map", "when the selected word is wrong", color=RED)

    dot_row(c, 150, 410, BLUE)
    c.line(205, 430, 245, 455, BLUE, 3, arrow=True)
    c.circle(285, 470, 25, LIGHT_BLUE, stroke=BLUE, width=3)
    c.circle(285, 470, 9, NAVY)
    c.text(285, 500, "WORD REPRESENTATIVE", 13, NAVY, bold=True, anchor="mm")

    dot_row(c, 635, 410, RED)
    c.line(690, 430, 720, 455, RED, 3, arrow=True)
    c.circle(720, 470, 13, RED)
    c.line(740, 470, 795, 445, BLUE, 3, arrow=True)
    c.circle(820, 435, 22, LIGHT_BLUE, stroke=BLUE, width=3)
    c.circle(820, 435, 8, NAVY)
    c.text(820, 475, "CORRECT WORD", 13, BLUE, bold=True, anchor="mm")

    c.rect(1090, 390, 1300, 465, LIGHT_RED, stroke=RED, width=2, radius=6)
    c.text(1195, 414, "RESULT", 16, RED, bold=True, anchor="mm")
    c.text(1195, 442, "UPDATE THE ENCODER", 15, NAVY, bold=True, anchor="mm")
    arrow_between(c, 430, 525)
    arrow_between(c, 905, 1000)

    bottom_message(
        c,
        "THIS IS CLOSE TO THE FINAL SYSTEM",
        "example recordings create a prototype; a new recording is matched to the nearest prototype",
    )
    c.text(390, 743, "FIVE BUILD THE REPRESENTATIVE", 15, WHITE, bold=True, anchor="mm")
    c.text(720, 743, "FIVE TEST IT", 15, WHITE, bold=True, anchor="mm")
    c.text(1050, 743, "REPEATED FOR EVERY BATCH", 15, WHITE, bold=True, anchor="mm")
    return c.save("qa_loss_simple_ge2e")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [*render_triplet(), *render_scaf(), *render_ge2e()]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
