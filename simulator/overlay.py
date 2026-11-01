"""Debug drawings over camera frames: calibration and ground truth, never fed to vision code."""

from PIL import Image, ImageDraw

from simulator.camera_geometry import HEIGHT, WIDTH

ROI_COLOR = (255, 220, 0)
STOP_LINE_COLOR = (255, 60, 60)
BOX_COLOR = (0, 255, 255)
LABEL_BACKGROUND = (0, 0, 0)
LABEL_COLOR = (255, 255, 255)
MOSAIC_COLUMNS = 3


def annotate(image: Image.Image, calibration: dict, sample: dict) -> Image.Image:
    """A copy of one camera's frame, marked up from its calibration and the frame's manifest entry."""
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    for lines in calibration["approaches"].values():
        draw.polygon([tuple(point) for point in lines["roi"]], outline=ROI_COLOR)
        draw.line([tuple(point) for point in lines["stop_line"]], fill=STOP_LINE_COLOR, width=2)
    for vehicle in sample["vehicles"]:
        box = vehicle["boxes"].get(calibration["name"])
        if box:
            draw.rectangle(box, outline=BOX_COLOR)
            draw.text((box[0] + 2, box[1] + 1), vehicle["id"], fill=BOX_COLOR)
    label = f"{calibration['name']}   {sample['timestamp_sim_s']:.3f} s   {sample['phase']}"
    draw.rectangle(draw.textbbox((6, 6), label), fill=LABEL_BACKGROUND)
    draw.text((6, 6), label, fill=LABEL_COLOR)
    return marked


def mosaic(images: list[Image.Image]) -> Image.Image:
    """Every camera's frame on one sheet, so they can be compared at a single moment."""
    rows = (len(images) + MOSAIC_COLUMNS - 1) // MOSAIC_COLUMNS
    sheet = Image.new("RGB", (WIDTH * MOSAIC_COLUMNS, HEIGHT * rows))
    for index, image in enumerate(images):
        sheet.paste(image, (index % MOSAIC_COLUMNS * WIDTH, index // MOSAIC_COLUMNS * HEIGHT))
    return sheet
