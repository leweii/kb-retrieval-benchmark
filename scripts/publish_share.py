"""Publish report.html / report.en.html to https://share.jakobhe.com as two posts.

Follows the share-site layout (~/myknowledge/store/jobs/03_share-site): s3://<bucket>/<slug>/index.html + meta.json,
then the site index is rebuilt with share-site's own build_index() so the cards match the other posts.

Deliberately does NOT run `publish.sh --reindex`: on this machine the share-site template images are Git LFS
pointer files (git-lfs not installed), and --reindex re-uploads assets/ from the template — it would overwrite
the live cover / fallback images of every post with ~130-byte pointer files. Here only index.html is uploaded;
assets/ on S3 are never touched. Our posts reference the live fallback photo picked by share-site's rule.

  uv run --with markdown --with pygments --with pyyaml --with pypinyin --with pillow \
      python -I scripts/publish_share.py [--dry-run]
"""

from __future__ import annotations

import hashlib
import html
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHARE_JOB = Path.home() / "myknowledge/store/jobs/03_share-site"
OUT = Path("/tmp/kbb_share")
DATE = "2026-11-11"
POSTS = [
    {"src": "report.html", "slug": "2026-11-11-kb-retrieval-benchmark", "lang": "zh_CN",
     "title": "知识库检索方案评测：300 / 900 / 3000 篇该用什么检索和模型",
     "description": "3 个模型 × 11 种检索配置 × 3 档体量，31,713 条评分：DeepSeek V4.1 Flash + 智能体 grep "
                    "在三档都最好（3000 篇准确率 70%）。"},
    {"src": "report.en.html", "slug": "2026-11-11-kb-retrieval-benchmark-en", "lang": "en_US",
     "title": "Knowledge-base retrieval benchmark: what to use at 300 / 900 / 3000 docs",
     "description": "3 models × 11 retrieval setups × 3 KB sizes, 31,713 graded answers: DeepSeek V4.1 Flash with an "
                    "agentic grep loop wins at every size (70% accuracy at 3,000 docs)."},
]
LIVE_FALLBACK_SIZE = {1: (1200, 972), 2: (1200, 962), 3: (1200, 910), 4: (1200, 950), 5: (1200, 844)}  # from the live site
HTML_CACHE = "public,max-age=60"


def load_share_module():
    spec = importlib.util.spec_from_file_location("share_publish", SHARE_JOB / "bin/publish.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def aws(*args: str) -> str:
    r = subprocess.run(["aws", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"aws {' '.join(args[:3])} failed: {r.stderr.strip()[:400]}")
    return r.stdout


def head_tags(post: dict, cfg: dict, share_img: str, size: tuple[int, int]) -> str:
    url = f"{cfg['site_url']}/{post['slug']}/"
    e = html.escape
    return "\n".join([
        '<meta name="robots" content="noindex, nofollow, noarchive">',
        f'<meta name="description" content="{e(post["description"])}">',
        f'<link rel="canonical" href="{url}">',
        '<meta property="og:type" content="article">',
        f'<meta property="og:title" content="{e(post["title"])}">',
        f'<meta property="og:description" content="{e(post["description"])}">',
        f'<meta property="og:url" content="{url}">',
        f'<meta property="og:image" content="{share_img}">',
        f'<meta property="og:image:width" content="{size[0]}">',
        f'<meta property="og:image:height" content="{size[1]}">',
        f'<meta property="og:image:alt" content="{e(post["title"])}">',
        f'<meta property="og:locale" content="{post["lang"]}">',
        '<meta name="twitter:card" content="summary_large_image">',
        f'<meta name="twitter:image" content="{share_img}">',
        f'<link rel="image_src" href="{share_img}">',
        f'<meta itemprop="image" content="{share_img}">',
    ])


def build_posts(cfg: dict, fallback_count: int) -> list[dict]:
    records = []
    for post in POSTS:
        n = (int(hashlib.md5(post["slug"].encode()).hexdigest(), 16) % fallback_count) + 1  # share-site's rule
        share_img = f"{cfg['site_url']}/assets/fallback/{n}-cover.jpg"
        page = (ROOT / post["src"]).read_text(encoding="utf-8")
        title_tag = page[page.index("<title>"): page.index("</title>") + len("</title>")]
        page = page.replace(title_tag, f"<title>{html.escape(post['title'])}</title>\n"
                            + head_tags(post, cfg, share_img, LIVE_FALLBACK_SIZE[n]), 1)
        d = OUT / post["slug"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "index.html").write_text(page, encoding="utf-8")
        rec = {"slug": post["slug"], "title": post["title"], "date": DATE, "description": post["description"],
               "cover": None, "thumb": None, "fallback_index": n, "share_image": share_img,
               "url": f"{cfg['site_url']}/{post['slug']}/", "source": str(ROOT / post["src"]),
               "published_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")}
        (d / "meta.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        records.append(rec)
    return records


def main() -> None:
    dry = "--dry-run" in sys.argv
    sp = load_share_module()
    cfg = sp.load_config()
    bucket, dist = cfg["bucket"], cfg["distribution_id"]
    records = build_posts(cfg, sp.FALLBACK_COUNT)
    print("built:", ", ".join(str(OUT / r["slug"]) for r in records))
    if dry:
        return

    for r in records:
        for name, ctype in (("index.html", "text/html; charset=utf-8"), ("meta.json", "application/json; charset=utf-8")):
            aws("s3", "cp", str(OUT / r["slug"] / name), f"s3://{bucket}/{r['slug']}/{name}", "--only-show-errors",
                "--content-type", ctype, "--cache-control", HTML_CACHE)
        print("uploaded", r["url"])

    # rebuild the index with share-site's own code; the template cover is an LFS pointer here, so report the live
    # default cover's real size instead of letting sips read the pointer
    real_size = sp.image_size
    sp.image_size = lambda p: (1200, 972) if Path(p).name == sp.DEFAULT_COVER_SRC else real_size(p)
    remote = sp.remote_records(bucket)
    assert {r["slug"] for r in records} <= {r["slug"] for r in remote}, "new meta.json not visible on S3"
    root = sp.build_index(cfg, remote)
    aws("s3", "cp", str(root / "index.html"), f"s3://{bucket}/index.html", "--only-show-errors",
        "--content-type", "text/html; charset=utf-8", "--cache-control", HTML_CACHE)
    print(f"index rebuilt with {len(remote)} posts (assets/ untouched)")

    paths = ["/", "/index.html"] + [f"/{r['slug']}/*" for r in records]
    sp.invalidate(dist, paths)


if __name__ == "__main__":
    main()
