import os
import re
import time
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def make_session(contact: str):
    """
    Wikimedia expects a descriptive User-Agent with contact info.
    """
    s = requests.Session()
    s.headers.update({
        "User-Agent": f"EdgarPayneCommonsDownloader/1.0 (contact: {contact})",
        "Accept": "application/json",
    })

    retries = Retry(
        total=6,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

def safe_filename(name: str) -> str:
    # Windows-safe filenames
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

def get_category_files(session: requests.Session, api: str, category: str, limit: int = 200, sleep_s: float = 0.2):
    """
    Returns a list of file titles like: 'File:Some_image.jpg'
    """
    titles = []
    cmcontinue = None

    while True:
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": category,     # e.g. 'Category:Paintings by Edgar Payne'
            "cmtype": "file",
            "cmlimit": limit,
            "origin": "*",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        r = session.get(api, params=params, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"API error {r.status_code}: {r.text[:300]}")

        data = r.json()
        members = data.get("query", {}).get("categorymembers", [])
        titles.extend(m["title"] for m in members if "title" in m)

        cont = data.get("continue", {})
        if "cmcontinue" in cont:
            cmcontinue = cont["cmcontinue"]
            time.sleep(sleep_s)
        else:
            break

    return titles

def get_imageinfo(session: requests.Session,api: str, file_titles, batch_size: int = 50, sleep_s: float = 0.2):
    """
    Batch request for imageinfo: url + extmetadata.
    Returns dict: title -> imageinfo dict
    """
    out = {}

    for i in range(0, len(file_titles), batch_size):
        batch = file_titles[i:i+batch_size]
        params = {
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "titles": "|".join(batch),
            "iiprop": "url|extmetadata",
            "origin": "*",
        }
        r = session.get(api, params=params, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"API error {r.status_code}: {r.text[:300]}")

        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        for _, page in pages.items():
            title = page.get("title")
            ii = (page.get("imageinfo") or [{}])[0]
            if title and ii.get("url"):
                out[title] = ii

        time.sleep(sleep_s)

    return out

def license_is_allowed(extmeta: dict) -> bool:
    """
    Conservative filter: allow Public Domain and Creative Commons licenses.
    This is a heuristic, not legal advice.
    """
    if not extmeta:
        return False

    def v(key):
        return (extmeta.get(key, {}) or {}).get("value", "") or ""

    text = " ".join([v("LicenseShortName"), v("License"), v("LicenseUrl"), v("UsageTerms")]).lower()

    allowed_markers = [
        "public domain"
        # "cc-by", "cc by",
        # "cc0",
        # "creative commons",
    ]
    return any(m in text for m in allowed_markers)

def download_file(session: requests.Session, url: str, out_path: str, timeout: int = 180):
    """
    Stream download to disk.
    """
    with session.get(url, stream=True, timeout=timeout) as r:
        if r.status_code != 200:
            raise RuntimeError(f"Download error {r.status_code}: {url}")
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)

def test_connection(session: requests.Session, api: str):
    r = session.get(api, params={"action": "query", "format": "json", "meta": "siteinfo", "siprop": "general",
                                 "origin": "*"}, timeout=60)
    print("API status:", r.status_code)
    print("Site:", r.json()["query"]["general"]["sitename"])

def download_all_images(session: requests.Session, api: str, outputDirectory: str, category: str, licenseControl: bool, sleepTime: float, batchSize: int):
    os.makedirs(outputDirectory, exist_ok=True)
    print(f"Fetching files from {category} ...")
    files = get_category_files(session, api, category, limit=200, sleep_s=sleepTime)
    print(f"Found {len(files)} files in category.")

    print("Fetching image URLs + license metadata ...")
    info = get_imageinfo(session, api, files, batch_size=batchSize, sleep_s=sleepTime)
    print(f"Got imageinfo for {len(info)} files.")

    manifest = []
    downloaded = 0
    skipped_license = 0
    skipped_exists = 0
    failed = 0

    for idx, (title, ii) in enumerate(info.items(), start=1):
        url = ii["url"]
        extmeta = ii.get("extmetadata", {})

        if licenseControl and not license_is_allowed(extmeta):
            skipped_license += 1
            if skipped_license % 25 == 0:
                print(f"  skipped (license): {skipped_license}")
            continue

        name = title.split(":", 1)[1] if ":" in title else title
        name = safe_filename(name)
        out_path = os.path.join(outputDirectory, name)

        if os.path.exists(out_path):
            skipped_exists += 1
            continue

        try:
            download_file(session, url, out_path)
            downloaded += 1

            def mv(key):
                return (extmeta.get(key, {}) or {}).get("value", "") if extmeta else ""

            manifest.append({
                "title": title,
                "url": url,
                "license_short": mv("LicenseShortName"),
                "license_url": mv("LicenseUrl"),
                "artist": mv("Artist"),
                "credit": mv("Credit"),
                "attribution": mv("Attribution"),
            })

            if downloaded % 10 == 0 or downloaded <= 3:
                print(f"  downloaded {downloaded} (last: {title})")

            time.sleep(0.2)

        except Exception as e:
            failed += 1
            print(f"  FAILED: {title} -> {e}")

    # write manifest
    manifest_path = os.path.join(outputDirectory, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n=== DONE ===")
    print("Downloaded:", downloaded)
    print("Skipped (license):", skipped_license)
    print("Skipped (already exists):", skipped_exists)
    print("Failed:", failed)
    print("Manifest:", manifest_path)