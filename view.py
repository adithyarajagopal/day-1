import argparse
from ingest import extract_text_by_page

PDF_PATH = r"C:\Users\ACER\Downloads\DRHP_20250430105517.pdf"
SNIPPET_LEN = 300
DEFAULT_PREVIEW_COUNT = 3


def print_page(page: dict, snippet_len: int = SNIPPET_LEN) -> None:
    num = page["page_number"]
    text = page["text"]
    snippet = text[:snippet_len]
    if len(text) > snippet_len:
        snippet += " ..."
    print(f"{'=' * 60}")
    print(f"  PAGE {num}")
    print(f"{'=' * 60}")
    print(snippet)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="View extracted text pages from the DRHP PDF."
    )
    parser.add_argument(
        "--page",
        type=int,
        default=None,
        help="View a specific page number (1-based). Omit to see first 3 pages.",
    )
    args = parser.parse_args()

    print(f"Loading PDF: {PDF_PATH}")
    print("Extracting text (this may take a moment)...\n")

    pages = extract_text_by_page(PDF_PATH)
    total = len(pages)
    print(f"Total pages extracted with text: {total}\n")

    if args.page is not None:
        match = [p for p in pages if p["page_number"] == args.page]
        if not match:
            print(f"Page {args.page} not found or had no extractable text.")
        else:
            print_page(match[0])
    else:
        preview = pages[:DEFAULT_PREVIEW_COUNT]
        print(f"Showing first {len(preview)} pages:\n")
        for page in preview:
            print_page(page)


if __name__ == "__main__":
    main()
