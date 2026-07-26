"""Render the corrected slide-10 architecture diagram as a PNG."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1600
HEIGHT = 720
SCALE = 2

NAVY = "#273896"
RED = "#F0242B"
BLUE = "#1A73E8"
TEXT = "#35415D"
MUTED = "#4E5C78"
PANEL = "#F8FAFF"
BORDER = "#D8E0F3"
WHITE = "#FFFFFF"

FONT_REGULAR = Path("C:/Windows/Fonts/arial.ttf")
FONT_BOLD = Path("C:/Windows/Fonts/arialbd.ttf")
OUTPUT = Path(__file__).resolve().parents[1] / "docs/presentation/assets/slide10_architecture_corrected.png"


def scaled(value: int | float) -> int:
    return round(value * SCALE)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), scaled(size))


def centered(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, style: ImageFont.FreeTypeFont, fill: str) -> None:
    x, y = (scaled(xy[0]), scaled(xy[1]))
    box = draw.textbbox((0, 0), text, font=style)
    draw.text((x - (box[2] - box[0]) / 2, y), text, font=style, fill=fill)


def box(draw: ImageDraw.ImageDraw, x: int, y: int, width: int, height: int) -> None:
    draw.rounded_rectangle(
        (scaled(x), scaled(y), scaled(x + width), scaled(y + height)),
        radius=scaled(8),
        fill=WHITE,
        outline=BLUE,
        width=scaled(2),
    )


def arrow(draw: ImageDraw.ImageDraw, x1: int, y: int, x2: int) -> None:
    draw.line((scaled(x1), scaled(y), scaled(x2 - 10), scaled(y)), fill=BLUE, width=scaled(3))
    draw.polygon(
        [
            (scaled(x2), scaled(y)),
            (scaled(x2 - 12), scaled(y - 7)),
            (scaled(x2 - 12), scaled(y + 7)),
        ],
        fill=BLUE,
    )


def panel(draw: ImageDraw.ImageDraw, y: int) -> None:
    draw.rounded_rectangle(
        (scaled(40), scaled(y), scaled(1560), scaled(y + 205)),
        radius=scaled(8),
        fill=PANEL,
        outline=BORDER,
        width=scaled(2),
    )


def pipeline_box(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    title: str,
    lines: tuple[str, ...],
    width: int = 210,
) -> None:
    box(draw, x, y, width, 98)
    centered(draw, (x + width // 2, y + 28), title, font(22, bold=True), NAVY)
    if len(lines) == 1:
        centered(draw, (x + width // 2, y + 64), lines[0], font(17), TEXT)
    else:
        centered(draw, (x + width // 2, y + 55), lines[0], font(16), TEXT)
        centered(draw, (x + width // 2, y + 77), lines[1], font(16), TEXT)


def main() -> None:
    image = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), WHITE)
    draw = ImageDraw.Draw(image)

    draw.text((scaled(40), scaled(24)), "Shared acoustic frontend for the reported PCEN models", font=font(32, bold=True), fill=NAVY)

    box(draw, 40, 72, 280, 82)
    centered(draw, (180, 94), "1-second waveform", font(22, bold=True), NAVY)
    centered(draw, (180, 123), "16 kHz mono audio", font(17), TEXT)
    arrow(draw, 322, 113, 368)

    box(draw, 370, 72, 300, 82)
    centered(draw, (520, 94), "40-bin Mel energy", font(22, bold=True), NAVY)
    centered(draw, (520, 123), "101 time frames", font(17), TEXT)
    arrow(draw, 672, 113, 718)

    box(draw, 720, 72, 280, 82)
    centered(draw, (860, 94), "Trainable PCEN", font(22, bold=True), NAVY)
    centered(draw, (860, 123), "input: 1 x 40 x 101", font(17), TEXT)
    draw.text(
        (scaled(1040), scaled(101)),
        "Same feature geometry for both encoders",
        font=font(18),
        fill=MUTED,
    )

    panel(draw, 190)
    draw.text((scaled(68), scaled(215)), "BEST ACCURACY PROFILE", font=font(16, bold=True), fill=RED)
    draw.text((scaled(68), scaled(239)), "DSCNN-L", font=font(31, bold=True), fill=NAVY)
    draw.text((scaled(68), scaled(281)), "412.9K parameters", font=font(18), fill=MUTED)
    draw.text((scaled(68), scaled(307)), "276-D embedding", font=font(18), fill=MUTED)

    pipeline_box(draw, 330, 242, "Initial Conv", ("10 x 4, stride 2 x 1",))
    arrow(draw, 542, 291, 578)
    pipeline_box(draw, 580, 242, "DS Blocks x5", ("DW 3 x 3 + PW 1 x 1",))
    arrow(draw, 792, 291, 828)
    pipeline_box(draw, 830, 242, "Normalization", ("LayerNorm on final block",))
    arrow(draw, 1042, 291, 1078)
    pipeline_box(draw, 1080, 242, "Global Pool", ("spatial average",))
    arrow(draw, 1292, 291, 1328)
    pipeline_box(draw, 1330, 242, "Embedding", ("276-D output",), width=200)

    panel(draw, 425)
    draw.text((scaled(68), scaled(450)), "BEST COMPACT PROFILE", font=font(16, bold=True), fill=RED)
    draw.text((scaled(68), scaled(474)), "EdgeSpotFull T4", font=font(31, bold=True), fill=NAVY)
    draw.text((scaled(68), scaled(516)), "130.6K parameters", font=font(18), fill=MUTED)
    draw.text((scaled(68), scaled(542)), "64-D embedding", font=font(18), fill=MUTED)

    pipeline_box(draw, 330, 477, "Stem", ("Conv 5 x 5",))
    arrow(draw, 542, 526, 578)
    pipeline_box(draw, 580, 477, "Fused Temporal", ("Stages 1-2",))
    arrow(draw, 792, 526, 828)
    pipeline_box(draw, 830, 477, "BCRes-Lite", ("Stages 3-4",))
    arrow(draw, 1042, 526, 1078)
    pipeline_box(draw, 1080, 477, "Temporal Context", ("PosConv + 1-head attn.", "Temporal head + GAP"))
    arrow(draw, 1292, 526, 1328)
    pipeline_box(draw, 1330, 477, "Embedding", ("64-D output",), width=200)

    draw.line((scaled(40), scaled(668), scaled(1560), scaled(668)), fill=BLUE, width=scaled(2))
    draw.text(
        (scaled(40), scaled(681)),
        "MFCC ablations: 1 x 47 x 10. Main PCEN profiles: 1 x 40 x 101. L2 normalization is applied after either encoder.",
        font=font(16),
        fill=MUTED,
    )

    output = image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.save(OUTPUT, optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
