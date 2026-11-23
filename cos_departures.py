#!/usr/bin/env python3
"""
Stream one day of adsb.lol globe_history data and keep ONLY flights that
departed from Colorado Springs (KCOS). The ~4 GB day is read straight off the
network and never stored; only the matching flights are written (typically a
few MB), then split 80/20 by aircraft.

    python cos_departures.py 2026-09-18
    python cos_departures.py 2026-09-18 --radius-km 6

Outputs in ./cos_out/<day>/ :
    departures.csv   all matching points
    train.csv/test.csv  80/20 split by aircraft (icao), so no aircraft is in both

Trace point layout (readsb): [secs_after_timestamp, lat, lon, alt_baro_ft|"ground",
gs_kt, track, flags, vrate_fpm, ...]. flags bit 1 (value 2) marks the start of a
new leg (flight) within the aircraft's day.

A "departure" = a leg whose first point is within --radius-km of KCOS, on the
ground or below ~1,500 ft AGL, and which later climbs at least 2,000 ft.
Bandwidth: you still download the whole day (~4 GB) once; disk use stays tiny.
"""
import argparse
import csv
import gzip
import io
import json
import math
import os
import tarfile
import urllib.request
import zlib
from pathlib import Path

REPO = "adsblol/globe_history_2026"
KCOS = (38.80581, -104.70076)
FIELD_ELEV_FT = 6187
FIELDS = ["icao", "reg", "type", "leg", "t_unix", "lat", "lon", "alt_ft",
          "on_ground", "gs_kt", "track", "vrate_fpm"]


def tag_for(day):
    return f"v{day.replace('-', '.')}-planes-readsb-prod-0"


def req(url, accept="application/vnd.github+json"):
    h = {"Accept": accept, "User-Agent": "cos-departures"}
    if os.environ.get("GITHUB_TOKEN"):
        h["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    return urllib.request.Request(url, headers=h)


def list_parts(tag):
    url = f"https://api.github.com/repos/{REPO}/releases/tags/{tag}"
    with urllib.request.urlopen(req(url)) as r:
        rel = json.load(r)
    parts = sorted((a["name"], a["browser_download_url"]) for a in rel["assets"] if ".tar" in a["name"])
    if not parts:
        raise SystemExit(f"No .tar assets in {tag}: {[a['name'] for a in rel['assets']]}")
    return parts


class ChainedHTTP(io.RawIOBase):
    """Read several HTTP downloads back-to-back as one byte stream."""

    def __init__(self, urls):
        self.urls, self.i, self.r, self.total = list(urls), 0, None, 0

    def readable(self):
        return True

    def readinto(self, b):
        while True:
            if self.r is None:
                if self.i >= len(self.urls):
                    return 0
                self.r = urllib.request.urlopen(req(self.urls[self.i], "application/octet-stream"))
                self.i += 1
            n = self.r.readinto(b)
            if n:
                self.total += n
                return n
            self.r.close()
            self.r = None


def km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 12742 * math.asin(math.sqrt(a))


def alt_of(p):
    a = p[3]
    return None if a is None else (0 if a == "ground" else a)


def split_legs(trace):
    legs, cur = [], []
    for p in trace:
        if cur and (p[6] & 2):
            legs.append(cur)
            cur = []
        cur.append(p)
    if cur:
        legs.append(cur)
    return legs


def is_cos_departure(leg, radius_km):
    first = leg[0]
    if first[1] is None or km(first[1], first[2], *KCOS) > radius_km:
        return False
    a0 = alt_of(first)
    if a0 is None:
        return False
    ref = FIELD_ELEV_FT if first[3] == "ground" else a0
    if first[3] != "ground" and a0 > FIELD_ELEV_FT + 1500:
        return False
    alts = [alt_of(p) for p in leg if alt_of(p) is not None]
    return bool(alts) and max(alts) >= ref + 2000


def rows_for(doc, radius_km):
    icao, reg, typ = doc.get("icao"), doc.get("r"), doc.get("t")
    base = doc.get("timestamp", 0)
    for n, leg in enumerate(split_legs(doc.get("trace", []))):
        if not is_cos_departure(leg, radius_km):
            continue
        for p in leg:
            a = alt_of(p)
            yield [icao, reg, typ, n, round(base + p[0], 1), p[1], p[2],
                   a, int(p[3] == "ground"), p[4], p[5], p[7]]


def read_doc(data):
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return json.loads(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("day", help="YYYY-MM-DD")
    ap.add_argument("--radius-km", type=float, default=8.0)
    ap.add_argument("--out", default="cos_out")
    args = ap.parse_args()

    out = Path(args.out) / args.day
    out.mkdir(parents=True, exist_ok=True)
    parts = list_parts(tag_for(args.day))
    print(f"Streaming {len(parts)} part(s) for {tag_for(args.day)}")
    stream = ChainedHTTP([u for _, u in parts])

    seen = kept_aircraft = 0
    icaos = set()
    with open(out / "departures.csv", "w", newline="") as f, \
            tarfile.open(fileobj=io.BufferedReader(stream, 1 << 20), mode="r|*") as tar:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for m in tar:
            name = os.path.basename(m.name)
            if not (m.isfile() and name.startswith("trace_full_")):
                continue
            seen += 1
            try:
                doc = read_doc(tar.extractfile(m).read())
            except Exception:
                continue
            n = 0
            for row in rows_for(doc, args.radius_km):
                w.writerow(row)
                n += 1
            if n:
                kept_aircraft += 1
                icaos.add(doc["icao"])
            if seen % 5000 == 0:
                print(f"  {seen} aircraft scanned, {kept_aircraft} with KCOS departures, "
                      f"{stream.total / 2**30:.2f} GiB read", flush=True)
    print(f"Done: {seen} aircraft scanned, {kept_aircraft} with KCOS departures")

    # 80/20 split by aircraft; deterministic (crc32 of icao) so reruns agree
    def is_train(icao):
        return zlib.crc32(icao.encode()) % 10 < 8

    with open(out / "departures.csv", newline="") as f, \
            open(out / "train.csv", "w", newline="") as ft, \
            open(out / "test.csv", "w", newline="") as fe:
        r = csv.reader(f)
        header = next(r)
        wt, we = csv.writer(ft), csv.writer(fe)
        wt.writerow(header)
        we.writerow(header)
        for row in r:
            (wt if is_train(row[0]) else we).writerow(row)
    print(f"Wrote {out}/departures.csv, train.csv, test.csv")


if __name__ == "__main__":
    main()
