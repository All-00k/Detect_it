"""
generate_icons.py — Creates PNG icon files for the Detect_IT Chrome extension.
Run once: python generate_icons.py
"""
import os
import struct
import zlib

def create_png(size, filename):
    """Create a simple dark navy PNG icon with a magnifying glass symbol."""
    # Create RGBA pixel data
    pixels = []
    cx, cy = size // 2, size // 2
    r = int(size * 0.35)
    stroke = max(2, size // 16)

    for y in range(size):
        row = []
        for x in range(size):
            dx, dy = x - cx, y - cy

            # Background: dark navy
            R, G, B, A = 13, 13, 26, 255

            # Magnifying glass circle (ring)
            dist = (dx**2 + dy**2) ** 0.5
            if abs(dist - r) < stroke:
                R, G, B = 124, 107, 255  # purple

            # Magnifying glass handle
            hx1, hy1 = cx + int(r * 0.65), cy + int(r * 0.65)
            hx2, hy2 = cx + int(r * 1.3), cy + int(r * 1.3)
            # Line from (hx1,hy1) to (hx2,hy2)
            if hx2 != hx1:
                t = (x - hx1) / (hx2 - hx1)
                if 0 <= t <= 1:
                    expected_y = hy1 + t * (hy2 - hy1)
                    if abs(y - expected_y) < stroke * 1.5:
                        R, G, B = 96, 165, 250  # blue

            row.extend([R, G, B, A])
        pixels.append(row)

    # Build PNG
    def chunk(name, data):
        c = name + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)

    # IHDR
    ihdr_data = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)  # 8-bit RGB (not RGBA)
    # Use RGBA: color type 6
    ihdr_data = struct.pack(">II", size, size) + bytes([8, 6, 0, 0, 0])

    raw_data = b""
    for row in pixels:
        raw_data += b"\x00" + bytes(row)  # filter type 0

    idat_data = zlib.compress(raw_data)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr_data)
        + chunk(b"IDAT", idat_data)
        + chunk(b"IEND", b"")
    )

    with open(filename, "wb") as f:
        f.write(png)
    print(f"  Created: {filename} ({size}x{size})")


icons_dir = os.path.join(os.path.dirname(__file__), "extension", "icons")
os.makedirs(icons_dir, exist_ok=True)

for size, name in [(16, "icon16.png"), (48, "icon48.png"), (128, "icon128.png")]:
    create_png(size, os.path.join(icons_dir, name))

print("Done! Icons created in extension/icons/")
