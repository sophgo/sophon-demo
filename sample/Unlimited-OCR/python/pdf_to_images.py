"""PDF -> page images via PyMuPDF (fitz), matching the original infer flow."""

import os
import tempfile


def pdf_to_images(pdf_path, dpi=300):
    """Convert each PDF page to a PNG; returns list of image paths."""
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths = []
    for i, page in enumerate(doc):
        out = os.path.join(tmp_dir, f"page_{i + 1:04d}.png")
        page.get_pixmap(matrix=mat).save(out)
        paths.append(out)
    doc.close()
    return paths
