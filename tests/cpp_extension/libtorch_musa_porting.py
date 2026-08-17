import os
import shutil
from pathlib import Path


DEFAULT_CSRC_SUFFIXES = {".cpp", ".h", ".cu"}
DEFAULT_PACKAGE_SUFFIXES = {".py"}

TEST_CPP_EXTENSION_DIR = Path(__file__).resolve().parent
PYTORCH_DIR = Path(
    os.environ.get(
        "PYTORCH_REPO_PATH",
        TEST_CPP_EXTENSION_DIR.parents[2] / "pytorch",
    )
).expanduser()


def apply_replacements(text, replacements):
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def resolve_filename(path):
    name = path.name
    name = name.replace("cuda", "musa")
    name = name.replace(".cu", ".mu")
    return name


def collect_porting_sources(source_dir, allowed_suffixes, source_name):
    if not source_dir.exists():
        raise FileNotFoundError(
            f"{source_name} porting source directory does not exist: {source_dir}"
        )
    if not source_dir.is_dir():
        raise NotADirectoryError(
            f"{source_name} porting source path is not a directory: {source_dir}"
        )

    sources = [
        src
        for src in source_dir.rglob("*")
        if src.is_file() and src.suffix in allowed_suffixes
    ]
    if not sources:
        raise FileNotFoundError(
            f"{source_name} porting source directory has no supported files "
            f"({sorted(allowed_suffixes)}): {source_dir}"
        )
    return sources


def port_tree(
    source_dir,
    dst_dir,
    *,
    allowed_suffixes,
    source_name,
    text_replacements=(),
):
    sources = collect_porting_sources(source_dir, allowed_suffixes, source_name)

    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    for src in sources:
        rel_parent = src.relative_to(source_dir).parent
        dst_name = resolve_filename(src)
        dst = dst_dir / rel_parent / dst_name
        dst.parent.mkdir(parents=True, exist_ok=True)

        if not text_replacements:
            shutil.copyfile(src, dst)
            continue

        text = src.read_text(encoding="utf-8")
        dst.write_text(apply_replacements(text, text_replacements), encoding="utf-8")

    return dst_dir
