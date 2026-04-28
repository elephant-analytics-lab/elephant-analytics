import os
import subprocess
from datetime import datetime
from pathlib import Path


def _format_time(dt_raw: str | None) -> str | None:
    if not dt_raw:
        return None
    if " " not in dt_raw:
        return dt_raw
    date_part, time_part = dt_raw.split(" ", 1)
    date_part = date_part.replace(":", "-")
    return f"{date_part}T{time_part}"


def _extract_gps_points(path: Path, include_embedded: bool) -> list[tuple[float, float, str | None]]:
    exiftool_bin = os.getenv("EXIFTOOL_PATH", "exiftool")
    cmd = [
        exiftool_bin,
        "-f",
        "-p",
        "$gpslatitude#,$gpslongitude#,$gpsdatetime",
        str(path),
    ]
    if include_embedded:
        cmd.insert(1, "-ee")
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "exiftool is not installed in the runtime environment. "
            "Install exiftool or set EXIFTOOL_PATH to the correct executable."
        ) from exc
    points: list[tuple[float, float, str | None]] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",", 2)
        if len(parts) < 2:
            continue
        lat_raw, lon_raw = parts[0], parts[1]
        dt_raw = parts[2] if len(parts) > 2 else None
        try:
            lat = float(lat_raw)
            lon = float(lon_raw)
        except ValueError:
            continue
        points.append((lat, lon, _format_time(dt_raw)))
    stderr = proc.stderr.read() if proc.stderr else ""
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"exiftool failed: {stderr.strip()}")
    return points


def extract_gps_points(video_path: Path) -> list[tuple[float, float, str | None]]:
    return _extract_gps_points(video_path, include_embedded=True)


def extract_image_gps_points(image_path: Path) -> list[tuple[float, float, str | None]]:
    return _extract_gps_points(image_path, include_embedded=False)


def write_gpx(points: list[tuple[float, float, str | None]], out_path: Path, name: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<gpx version="1.1" creator="exiftool" xmlns="http://www.topografix.com/GPX/1/1">\n')
        f.write("  <trk>\n")
        f.write(f"    <name>{name}</name>\n")
        f.write("    <trkseg>\n")
        for lat, lon, ts in points:
            if ts:
                f.write(f'      <trkpt lat="{lat}" lon="{lon}"><time>{ts}</time></trkpt>\n')
            else:
                f.write(f'      <trkpt lat="{lat}" lon="{lon}" />\n')
        f.write("    </trkseg>\n")
        f.write("  </trk>\n")
        f.write("</gpx>\n")
