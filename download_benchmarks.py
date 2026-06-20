"""Download lexical similarity benchmarks: RG-65, WordSim-353 (sim and rel splits), SimLex-999."""
import urllib.request, os, csv

os.makedirs("datasets", exist_ok=True)
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ResearchScript/1.0)"}

def fetch_text(url):
    """Fetch URL as text with browser-like User-Agent (avoids 403)."""
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8")

def save_pairs_csv(rows, path):
    """Write [(w1, w2, score), ...] to a CSV with header."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["word1", "word2", "score"])
        for r in rows: w.writerow(r)

# RG-65 — fetched from the sematch repository which mirrors Faruqui's CMU copy
RG65_URL = "https://raw.githubusercontent.com/mfaruqui/eval-word-vectors/master/data/word-sim/EN-RG-65.txt"
try:
    print(f"Downloading RG-65 from {RG65_URL}...")
    text = fetch_text(RG65_URL)
    rows = [tuple(line.split("\t")) for line in text.strip().splitlines() if line.strip()]
    rows = [(r[0], r[1], float(r[2])) for r in rows]
    save_pairs_csv(rows, "datasets/rg65_en.csv")
    print(f"  RG-65 saved ({len(rows)} pairs)")
except Exception as e:
    print(f"  RG-65 download failed: {e}")

# WordSim-353 — Agirre et al. 2009 split (similarity vs relatedness)
# Mirrored at github.com/gsi-upm/sematch — stable, public, uses raw.githubusercontent.com
WS353_SIM_URL = "https://raw.githubusercontent.com/gsi-upm/sematch/master/sematch/dataset/wordsim/wordsim_similarity_goldstandard.txt"
WS353_REL_URL = "https://raw.githubusercontent.com/gsi-upm/sematch/master/sematch/dataset/wordsim/wordsim_relatedness_goldstandard.txt"

for label, url, path in [
    ("WordSim-353 (similarity)", WS353_SIM_URL, "datasets/wordsim353_similarity.csv"),
    ("WordSim-353 (relatedness)", WS353_REL_URL, "datasets/wordsim353_relatedness.csv"),
]:
    try:
        print(f"\nDownloading {label}...")
        text = fetch_text(url)
        rows = [tuple(line.split("\t")) for line in text.strip().splitlines() if line.strip()]
        rows = [(r[0], r[1], float(r[2])) for r in rows]
        save_pairs_csv(rows, path)
        print(f"  saved {path} ({len(rows)} pairs)")
    except Exception as e:
        print(f"  {label} failed: {e}")

# SimLex-999 — Hill et al. 2015, functional similarity (separate from relatedness)
# Hosted at fh295.github.io as a zip; we fetch from a mirror that exposes a single TSV
SIMLEX_URL = "https://fh295.github.io/SimLex-999.zip"
try:
    print(f"\nDownloading SimLex-999 from {SIMLEX_URL}...")
    import zipfile, io
    req = urllib.request.Request(SIMLEX_URL, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            if name.endswith("SimLex-999.txt"):
                with z.open(name) as f:
                    text = f.read().decode("utf-8")
                lines = text.strip().splitlines()
                # SimLex-999 columns: word1 word2 POS SimLex999 conc(w1) conc(w2) concQ Assoc(USF) SimAssoc333 SD(SimLex)
                rows = []
                for line in lines[1:]:  # skip header
                    cols = line.split("\t")
                    rows.append((cols[0], cols[1], float(cols[3])))
                save_pairs_csv(rows, "datasets/simlex999.csv")
                print(f"  saved datasets/simlex999.csv ({len(rows)} pairs)")
                break
except Exception as e:
    print(f"  SimLex-999 failed: {e}")

# Summary
print("\nFinal contents of datasets/:")
for f in sorted(os.listdir("datasets")):
    path = os.path.join("datasets", f)
    if os.path.isfile(path) and not f.startswith("."):
        # Count rows (excluding header)
        with open(path, encoding="utf-8") as fh:
            n = sum(1 for _ in fh) - 1
        print(f"  {f}: {os.path.getsize(path)} bytes, {n} pairs")
