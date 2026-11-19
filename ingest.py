
import re
import gc
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ---------- Configuration ----------

DB_PATH = "db"
DATA_PATH = "data"

MAX_CHUNK = 2000
BATCH_SIZE = 8


# ---------- Identify Act and Domain ----------

def parse_name(path):

    file_path = Path(path)

    act = file_path.stem
    domain = file_path.parent.name.lower()

    if domain == DATA_PATH:
        filename = act.lower()

        if "women" in filename or "domestic violence" in filename:
            domain = "women_rights"

        elif "consumer" in filename:
            domain = "consumer_rights"

        elif "child" in filename or "pocso" in filename:
            domain = "childrens_rights"

        elif "cyber" in filename or "digital personal data" in filename:
            domain = "cyber_rights"

        elif "criminal" in filename or "bharatiya" in filename:
            domain = "criminal_rights"

        elif "rti" in filename:
            domain = "rti"

        else:
            domain = "other"

    return domain, act


# ---------- Read PDF ----------

def read_pdf(path):

    import fitz

    with fitz.open(path) as pdf:

        pages = []

        for page in pdf:
            pages.append(page.get_text("text"))

        return "\n".join(pages)


# ---------- Read TXT ----------

def read_text(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return file.read()


# ---------- Split Legal Sections ----------

def split_sections(text, is_pdf):

    if is_pdf:

        # Remove Table of Contents
        toc_start = re.search(
            r"(?i)ARRANGEMENT OF SECTIONS",
            text
        )

        if toc_start:

            remaining_text = text[toc_start.end():]

            act_start = re.search(
                r"(?m)^\s*An Act to",
                remaining_text
            )

            if act_start:

                text = remaining_text[act_start.start():]

        # Remove Statement of Objects and Reasons
        text = re.split(
            r"(?i)STATEMENT OF OBJECTS AND REASONS",
            text
        )[0]

        # Split actual Act sections
        pattern = (
            r"(?m)(?=^[ \t]*\d{1,3}[A-Z]?\.\s+[A-Z])"
        )

        parts = re.split(pattern, text)

    else:

        parts = re.split(
            r"(?m)(?=^### )",
            text
        )

    sections = [
        part.strip()
        for part in parts
        if len(part.strip()) >= 40
    ]

    if not sections and text.strip():
        sections = [text.strip()]

    return sections


# ---------- Main Ingestion ----------

def main():

    print("Loading embedding model...")

    embedder = SentenceTransformer(
        "intfloat/multilingual-e5-small",
        device="cpu"
    )

    client = chromadb.PersistentClient(
        path=DB_PATH
    )

    # Delete old collection

    try:

        client.delete_collection("laws")

        print("Old laws collection removed.")

    except Exception:

        pass

    col = client.get_or_create_collection(
        "laws"
    )

    # Find all PDF and TXT files

    data_folder = Path(DATA_PATH)

    files = list(data_folder.rglob("*.pdf"))

    files += list(data_folder.rglob("*.txt"))

    if not files:

        print("No PDF or TXT files found.")

        return

    print(f"\nTotal files found: {len(files)}")

    total = 0

    # Process each document

    for path in files:

        path = str(path)

        domain, act = parse_name(path)

        is_pdf = path.lower().endswith(".pdf")

        print(f"\nProcessing: {act}")
        print(f"Domain: {domain}")

        try:

            text = (
                read_pdf(path)
                if is_pdf
                else read_text(path)
            )

        except Exception as e:

            print(f"Error reading {act}: {e}")

            continue

        if not text.strip():

            print(
                f"{act}: Empty document or scanned PDF."
            )

            continue

        # Split into legal sections

        sections = split_sections(
            text,
            is_pdf
        )

        ids = []
        docs = []
        metas = []

        # Process each section

        for i, section in enumerate(sections):

            lines = [
                line.strip()
                for line in section.splitlines()
                if line.strip()
            ]

            title = (
                lines[0][:150]
                if lines
                else "Unknown Section"
            )

            # Split long sections into chunks

            for j in range(
                0,
                len(section),
                MAX_CHUNK
            ):

                chunk = section[
                    j:j + MAX_CHUNK
                ]

                if len(chunk.strip()) < 40:
                    continue

                ids.append(
                    f"{act}-{i}-{j}"
                )

                docs.append(chunk)

                metas.append({

                    "act": act,

                    "domain": domain,

                    "section": title,

                    "source": path

                })

        if not docs:

            print(f"{act}: No content found.")

            continue

        print(
            f"Generating embeddings for {len(docs)} chunks..."
        )

        # Process and save small batches
        # This reduces RAM usage.

        file_chunks = 0

        for start in range(
            0,
            len(docs),
            BATCH_SIZE
        ):

            end = min(
                start + BATCH_SIZE,
                len(docs)
            )

            batch_docs = docs[start:end]

            batch_texts = [
                "passage: " + doc
                for doc in batch_docs
            ]

            batch_embeddings = embedder.encode(
                batch_texts,
                normalize_embeddings=True,
                batch_size=1,
                show_progress_bar=False
            )

            col.upsert(

                ids=ids[start:end],

                documents=batch_docs,

                embeddings=batch_embeddings.tolist(),

                metadatas=metas[start:end]

            )

            file_chunks += len(batch_docs)

            del batch_embeddings
            del batch_texts

            gc.collect()

            print(
                f"\rSaved {file_chunks}/{len(docs)} chunks",
                end="",
                flush=True
            )

        total += file_chunks

        print()

        print(
            f"{act} | Category: {domain} | "
            f"Chunks: {file_chunks}"
        )

        # Clear document memory after each file

        del sections
        del ids
        del docs
        del metas
        del text

        gc.collect()

    # Final report

    print("\n================================")
    print("LEGAL DOCUMENT INGESTION DONE")
    print("================================")

    print(f"Total files: {len(files)}")
    print(f"Total chunks: {total}")
    print(f"Database: {DB_PATH}")


# ---------- Run Program ----------

if __name__ == "__main__":

    main()
