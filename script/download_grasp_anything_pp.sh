#!/usr/bin/env bash
#
# Download Grasp-Anything++ (language-driven grasping).
#
# GA++ ships only the linguistics/label side; the images live in the base
# Grasp-Anything repo. This script pulls exactly what the language-driven task
# needs and lays it out in one directory:
#
#   <dest>/image/                 416x416 .jpg, named by SHA-256  (base repo, 65 GB)
#   <dest>/scene_description/     .pkl (caption, [objects])        (base repo, needed to
#                                 build the seen/unseen split -- see split/build_grasp_anything_pp.py)
#   <dest>/grasp_instructions/    .pkl grasping prompts per scene (GA++)
#   <dest>/grasp_label_positive/  .pt part-level positive grasps  (GA++)
#   <dest>/part_mask/             .npy part-level masks           (GA++)
#   <dest>/grasp_label_negative/  .pt negative grasps             (GA++, --with-negatives)
#
# NOTE: both repos contain a grasp_label_positive.zip. The base one is
# object-level, the GA++ one is part-level. We take GA++'s -- do not mix them.
#
# The authors ask that you fill in the download form and agree to the MIT
# license before using the data:
#   https://airvlab.github.io/grasp-anything/docs/download/
#
# Usage:
#   script/download_grasp_anything_pp.sh [--dest DIR] [--with-negatives]
#                                        [--keep-zips] [--check]

set -euo pipefail

REPO_BASE="airvlab/Grasp-Anything"
REPO_PP="airvlab/Grasp-Anything-pp"

DEST="data/grasp-anything-pp"
WITH_NEGATIVES=0
KEEP_ZIPS=0
CHECK_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest)            DEST="$2"; shift 2 ;;
        --with-negatives)  WITH_NEGATIVES=1; shift ;;
        --keep-zips)       KEEP_ZIPS=1; shift ;;
        --check)           CHECK_ONLY=1; shift ;;
        -h|--help)         sed -n '2,28p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

ZIPS="$DEST/_archives"

human() { numfmt --to=iec --suffix=B "$1" 2>/dev/null || echo "$1 bytes"; }

# ---------------------------------------------------------------- check mode --
if [[ "$CHECK_ONLY" == 1 ]]; then
    echo "Checking $DEST"
    status=0
    for d in image scene_description grasp_instructions grasp_label_positive part_mask; do
        if [[ -d "$DEST/$d" ]]; then
            n=$(find "$DEST/$d" -maxdepth 1 -type f | wc -l)
            printf "  %-22s %8d files\n" "$d" "$n"
            [[ "$n" -eq 0 ]] && status=1
        else
            printf "  %-22s MISSING\n" "$d"
            status=1
        fi
    done
    [[ -d "$DEST/grasp_label_negative" ]] && \
        printf "  %-22s %8d files\n" "grasp_label_negative" \
            "$(find "$DEST/grasp_label_negative" -maxdepth 1 -type f | wc -l)"
    exit "$status"
fi

# ------------------------------------------------------------ preflight ------
for cmd in unzip; do
    command -v "$cmd" >/dev/null || { echo "Missing required command: $cmd" >&2; exit 1; }
done

NEEDED_GB=150
avail_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
echo "Disk: ${avail_gb}G available, ~${NEEDED_GB}G needed (65G images + ~10G labels, plus extraction)."
if [[ "$avail_gb" -lt "$NEEDED_GB" ]]; then
    echo "Not enough free space." >&2
    exit 1
fi

mkdir -p "$DEST" "$ZIPS"

# Pick a downloader. hf/huggingface-cli resume and parallelise; wget -c is the
# fallback and matters because image_part_aa alone is 34 GB.
if command -v hf >/dev/null; then
    DOWNLOADER=hf
elif command -v huggingface-cli >/dev/null; then
    DOWNLOADER=huggingface-cli
elif command -v wget >/dev/null; then
    DOWNLOADER=wget
else
    echo "Need one of: hf, huggingface-cli, wget" >&2
    exit 1
fi
echo "Downloader: $DOWNLOADER"

fetch() {  # fetch <repo> <filename>
    local repo="$1" file="$2"
    if [[ -f "$ZIPS/$file" ]]; then
        echo "  [skip] $file already downloaded ($(human "$(stat -c%s "$ZIPS/$file")"))"
        return
    fi
    echo "  [get ] $repo :: $file"
    case "$DOWNLOADER" in
        hf)
            hf download "$repo" "$file" --repo-type dataset --local-dir "$ZIPS" ;;
        huggingface-cli)
            huggingface-cli download "$repo" "$file" --repo-type dataset --local-dir "$ZIPS" ;;
        wget)
            wget -c -O "$ZIPS/$file.part" \
                "https://huggingface.co/datasets/$repo/resolve/main/$file"
            mv "$ZIPS/$file.part" "$ZIPS/$file" ;;
    esac
}

extract() {  # extract <zipname> <expected_dirname>
    local zip="$1" dir="$2"
    if [[ -d "$DEST/$dir" ]] && [[ -n "$(find "$DEST/$dir" -maxdepth 1 -type f -print -quit)" ]]; then
        echo "  [skip] $dir already extracted"
        return
    fi
    echo "  [unzip] $zip -> $DEST/$dir"
    unzip -q -o "$ZIPS/$zip" -d "$DEST"
    [[ -d "$DEST/$dir" ]] || echo "  WARNING: expected $DEST/$dir after extracting $zip" >&2
    [[ "$KEEP_ZIPS" == 1 ]] || rm -f "$ZIPS/$zip"
}

# ------------------------------------------------- 1. images (base repo) -----
echo
echo "== 1/3  images from $REPO_BASE (65 GB, the slow part) =="
if [[ -d "$DEST/image" ]] && [[ -n "$(find "$DEST/image" -maxdepth 1 -name '*.jpg' -print -quit)" ]]; then
    echo "  [skip] image/ already extracted"
else
    fetch "$REPO_BASE" image_part_aa
    fetch "$REPO_BASE" image_part_ab
    if [[ ! -f "$ZIPS/image.zip" ]]; then
        echo "  [cat  ] image_part_aa + image_part_ab -> image.zip"
        cat "$ZIPS/image_part_aa" "$ZIPS/image_part_ab" > "$ZIPS/image.zip"
        [[ "$KEEP_ZIPS" == 1 ]] || rm -f "$ZIPS/image_part_aa" "$ZIPS/image_part_ab"
    fi
    extract image.zip image
fi

# --------------------------------- 1b. scene descriptions (base repo, 0.34 GB) --
echo
echo "== 1b/3  scene_description from $REPO_BASE (0.34 GB, needed for the split) =="
fetch "$REPO_BASE" scene_description.zip
extract scene_description.zip scene_description

# ------------------------------------------------- 2. GA++ labels ------------
echo
echo "== 2/3  language + part-level labels from $REPO_PP (~10 GB) =="
fetch "$REPO_PP" grasp_instructions.zip
fetch "$REPO_PP" grasp_label_positive.zip
fetch "$REPO_PP" part_mask.zip
[[ "$WITH_NEGATIVES" == 1 ]] && fetch "$REPO_PP" grasp_label_negative.zip

extract grasp_instructions.zip    grasp_instructions
extract grasp_label_positive.zip  grasp_label_positive
extract part_mask.zip             part_mask
[[ "$WITH_NEGATIVES" == 1 ]] && extract grasp_label_negative.zip grasp_label_negative

# ------------------------------------------------- 3. verify -----------------
echo
echo "== 3/3  verify =="
rmdir "$ZIPS" 2>/dev/null || true
exec "$0" --dest "$DEST" --check
