from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path("/home/lxy/ODT_pretrain/figures")
FONT_REG = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def font(size: int, bold: bool = False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size=size)


def multiline_center(draw, xy, text, fnt, fill):
    box = draw.multiline_textbbox((0, 0), text, font=fnt, spacing=6, align="center")
    x = xy[0] - (box[2] - box[0]) / 2
    y = xy[1] - (box[3] - box[1]) / 2
    draw.multiline_text((x, y), text, font=fnt, fill=fill, spacing=6, align="center")


def draw_box(draw, rect, text, *, fill="#F7F9FB", outline="#8CA3B6", text_fill="#18222D", accent=False):
    if accent:
        fill = "#EAF5F7"
    draw.rounded_rectangle(rect, radius=10, fill=fill, outline=outline, width=2)
    x0, y0, x1, y1 = rect
    multiline_center(draw, ((x0 + x1) / 2, (y0 + y1) / 2), text, font(24), text_fill)


def arrow(draw, a, b, color="#1D6870"):
    ax, ay = a
    bx, by = b
    draw.line((ax, ay, bx, by), fill=color, width=4)
    if bx >= ax:
        pts = [(bx, by), (bx - 15, by - 8), (bx - 15, by + 8)]
    else:
        pts = [(bx, by), (bx + 15, by - 8), (bx + 15, by + 8)]
    draw.polygon(pts, fill=color)


def row(draw, boxes, y, h=92):
    rects = []
    for x, w, text, accent in boxes:
        rect = (x, y, x + w, y + h)
        draw_box(draw, rect, text, accent=accent)
        rects.append(rect)
    for left, right in zip(rects, rects[1:]):
        arrow(draw, (left[2], (left[1] + left[3]) / 2), (right[0], (right[1] + right[3]) / 2))
    return rects


def mae_flow():
    img = Image.new("RGB", (1720, 900), "white")
    draw = ImageDraw.Draw(img)
    draw.text((70, 58), "ODT MAE 预训练", font=font(38, True), fill="#18222D")
    draw.text((72, 118), "单个 16×16 空间 patch，沿 240 帧做 MAE", font=font(27), fill="#56606B")

    draw.text((70, 236), "Encoder", font=font(30, True), fill="#18222D")
    row(
        draw,
        [
            (70, 210, "image/bg\n[B,240,2,16,16]×2", False),
            (350, 210, "concat + embed\n[B,240,Denc]", False),
            (630, 205, "frame pos\n[B,240,Denc]", False),
            (900, 190, "mask 75%\n[B,60,Denc]", False),
            (1160, 170, "+ CLS\n[B,61,Denc]", False),
            (1400, 240, "MAE Encoder\nTransformer", True),
        ],
        295,
    )

    draw.text((70, 533), "Decoder", font=font(30, True), fill="#18222D")
    row(
        draw,
        [
            (70, 170, "latent\n[B,61,Denc]", False),
            (310, 200, "project\n[B,61,Ddec]", False),
            (580, 280, "visible + mask/bg\nrestore 240 order\n[B,240,Ddec]", False),
            (930, 175, "+ CLS\n[B,241,Ddec]", False),
            (1175, 250, "MAE Decoder\nTransformer", True),
            (1495, 175, "pred image\n[B,240,2,16,16]", False),
        ],
        590,
        h=112,
    )

    draw.text((72, 820), "Loss: full MSE 训练；masked MSE 只记录指标", font=font(27), fill="#56606B")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(OUT_DIR / "mae_pretrain_flow.png")


def direct_flow():
    img = Image.new("RGB", (1840, 900), "white")
    draw = ImageDraw.Draw(img)
    draw.text((70, 58), "ODT Direct Inversion 主训练", font=font(38, True), fill="#18222D")
    draw.text((72, 118), "完整 240 帧输入，目标体素 [B,128,256,256]", font=font(27), fill="#56606B")

    draw.text((70, 220), "Encoder Memory", font=font(30, True), fill="#18222D")
    row(
        draw,
        [
            (70, 230, "image/bg\n[B,240,2,256,256]×2", False),
            (370, 210, "patchify\n16×16 grid", False),
            (650, 260, "[B×256,240,2,16,16]", False),
            (980, 230, "embed + frame pos\n[B×256,240,D]", False),
            (1280, 180, "+ CLS\n[B×256,241,D]", False),
            (1530, 240, "MAE Encoder\nencode_full", True),
        ],
        280,
    )

    draw.text((70, 522), "Voxel Decoder", font=font(30, True), fill="#18222D")
    row(
        draw,
        [
            (70, 200, "encoded\n[B,256,241,D]", False),
            (340, 310, "memory + spatial pos\nall_frames: [B,61440,D]\ncls: [B,256,D]", False),
            (730, 260, "Cross-Attn\n3D Grid Decoder", True),
            (1060, 230, "latent grid\n[B,16,32,32,D]", False),
            (1360, 220, "3D upsample", False),
            (1640, 165, "voxel\n[B,128,256,256]", False),
        ],
        585,
        h=112,
    )

    draw.text((72, 820), "freeze_encoder=true: encoder no_grad；false: encoder 参与训练", font=font(27), fill="#56606B")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(OUT_DIR / "direct_inversion_flow.png")


if __name__ == "__main__":
    mae_flow()
    direct_flow()
    print(f"saved to {OUT_DIR}")
