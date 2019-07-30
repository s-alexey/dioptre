import glob
import os

from PIL import Image, ImageFont, ImageDraw


def load_font(filename, size):
    if isinstance(filename, ImageFont.FreeTypeFont):
        return filename
    return ImageFont.truetype(filename, size)


def render(font, text):
    img = Image.new(mode='L', size=font.getsize(text), color=255)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font)
    return img


def find_fonts(directory, font_size):
    # todo: ignore case in file extensions
    font_files = glob.glob(os.path.join(directory, '*.ttf'))
    return [ImageFont.truetype(f, size=font_size) for f in font_files]
