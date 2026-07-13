"""Markdown -> docx export with VISIBLE table borders ('all borders').

pandoc's default reference docx renders tables borderless in Word; Ken's
standing preference is all-borders on every table. This wraps the usual
pypandoc gfm export and post-processes word/document.xml, injecting a
<w:tblBorders> block (single, 0.5pt, auto colour) into every table's
<w:tblPr> — anchored right after <w:tblW> to respect the element order.

Usage:
  python -m t9sim.notebooks.docx_export <input.md> <output.docx>
Or import:  from t9sim.notebooks.docx_export import export
"""
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import pypandoc

BORDERS = (
    '<w:tblBorders>'
    '<w:top w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
    '<w:left w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
    '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
    '<w:right w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
    '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
    '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
    '</w:tblBorders>'
)


def _add_borders(docx_path, update_fields=False):
    docx_path = Path(docx_path)
    tmp = Path(tempfile.mkdtemp())
    try:
        with zipfile.ZipFile(docx_path) as z:
            z.extractall(tmp)
        doc = tmp / "word" / "document.xml"
        xml = doc.read_text(encoding="utf-8")
        n = len(re.findall(r"<w:tblW[^>]*/>", xml))
        # anchor after <w:tblW/> (order: ...tblW, jc, ..., tblBorders...)
        xml = re.sub(r"(<w:tblW[^>]*/>)", r"\1" + BORDERS, xml)
        # fallback: tblPr without tblW
        xml = re.sub(r"(<w:tblPr>(?:(?!</w:tblPr>|<w:tblW|<w:tblBorders).)*?)"
                     r"(</w:tblPr>)", r"\1" + BORDERS + r"\2", xml)
        doc.write_text(xml, encoding="utf-8")
        # make Word rebuild the (pandoc-empty) TOC field on open
        if update_fields:
            sett = tmp / "word" / "settings.xml"
            if sett.exists():
                s = sett.read_text(encoding="utf-8")
                if "w:updateFields" not in s:
                    s = re.sub(r"(<w:settings[^>]*>)",
                               r'\1<w:updateFields w:val="true"/>', s, count=1)
                    sett.write_text(s, encoding="utf-8")
        out = docx_path.with_suffix(".tmp.docx")
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for f in tmp.rglob("*"):
                if f.is_file():
                    z.write(f, f.relative_to(tmp))
        out.replace(docx_path)
        return n
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def export(md_path, docx_path, toc=False, toc_depth=3):
    extra = ["--toc", f"--toc-depth={toc_depth}"] if toc else []
    pypandoc.convert_file(str(md_path), "docx", format="gfm",
                          outputfile=str(docx_path), extra_args=extra)
    n = _add_borders(docx_path, update_fields=toc)
    print(f"{Path(docx_path).name}: written, all-borders applied "
          f"to {n} tables{' + table of contents' if toc else ''}")


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    export(pos[0], pos[1], toc="--toc" in sys.argv)
