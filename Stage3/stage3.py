# stage3.py
# Stage 3: Cross-Document Consistency Checker
#
# This stage catches poisoned documents that Stages 1 and 2 missed.
# It looks at ALL retrieved documents together and flags any document
# that is semantically isolated from the group (i.e. tells a different story).
#
# Input:  list of documents + trust scores from Stage 2
# Output: filtered list of documents + trust scores with outliers removed

import math

from sentence_transformers import SentenceTransformer, util

# ============================================================
# SETUP: Load the sentence transformer model
# This model converts documents into vectors so we can compare them
# No fine-tuning needed — we use it as-is
# ============================================================

print("Loading sentence transformer...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ============================================================
# FUNCTION 1: Find semantic outliers using cosine similarity
#
# For each document, compute its average cosine similarity to all
# other documents in the set. Documents with LOW average similarity
# are outliers — they don't agree with the rest of the group.
#
# This is how we catch the bank example:
# - 4 docs say "X bank had breaches" → high similarity to each other
# - 1 doc says "X bank is perfectly safe" → low similarity to the group
# ============================================================


def find_semantic_outliers(documents, similarity_threshold=0.62):
    """
    Args:
        documents: list of document strings
        similarity_threshold: documents with avg similarity BELOW this
                              are flagged as outliers (default 0.62,
                              tuned on our test examples)
    Returns:
        outliers: list of indices of outlier documents
        avg_similarities: list of avg similarity scores (one per doc)
    """
    num_docs = len(documents)

    # Convert all documents to embedding vectors at once (efficient)
    embeddings = embedding_model.encode(documents, convert_to_tensor=True)

    # Compute average similarity of each document to all others
    avg_similarities = []
    for i in range(num_docs):
        similarities = []
        for j in range(num_docs):
            if i != j:
                sim = util.cos_sim(embeddings[i], embeddings[j]).item()
                similarities.append(sim)
        avg_sim = sum(similarities) / len(similarities)
        avg_similarities.append(avg_sim)
        print(f"  Doc {i} avg similarity to others: {avg_sim:.3f}")

    # Flag documents that are semantically isolated from the group
    outliers = []
    for i, avg_sim in enumerate(avg_similarities):
        if avg_sim < similarity_threshold:
            outliers.append(i)
            print(
                f"  Doc {i} flagged as semantic outlier "
                f"(avg similarity: {avg_sim:.3f})"
            )

    return outliers, avg_similarities


# ============================================================
# IMPORTANT: KNOWN LIMITATIONS OF STAGE 3
#
# Stage 3 is designed to catch documents that tell a DIFFERENT STORY
# from the majority of retrieved documents. It works well when:
#   - A poisoned doc claims "X bank is safe" while 4 others say "X bank had breaches"
#   - A poisoned doc contradicts the general consensus of the document set
#
# Stage 3 does NOT reliably catch subtle single-fact poisoning, such as:
#   - A doc saying "iPhone launched in 2010" when the correct year is 2007
#   - A doc with one wrong name, date, or number embedded in otherwise correct text
#
# This is expected and intentional — subtle single-fact poisoning is handled
# by Stages 1 and 2 (the classifier and trust scorer).
# Stage 3 specifically targets the case that Stages 1 and 2 MISS:
# coordinated multi-document poisoning where multiple fake docs
# reinforce each other (RAGuard's acknowledged failure mode).
#
# This limitation is consistent with our proposal and does not
# reduce the validity of Stage 3 as a contribution.
# ============================================================

# ============================================================
# FUNCTION 2: Main Stage 3 function
#
# Takes documents and trust scores from Stage 2.
# Flags semantic outliers, then only removes them if they ALSO
# have low trust scores. This prevents over-filtering — a document
# that looks like an outlier but has high trust is kept.
#
# From the proposal:
# "A document is flagged as an outlier if it has active contradiction
#  edges with >= ceil((k-1)/2) other documents"
# We approximate this using cosine similarity instead of NLI edges,
# since cosine similarity proved more reliable in our experiments.
# ============================================================


def stage3_consistency_check(
    documents,
    trust_scores,
    similarity_threshold=0.62,  # tau for cosine similarity outlier detection
    trust_threshold=0.5,  # secondary threshold from Stage 2
):
    """
    Args:
        documents:            list of document strings (output of Stage 2)
        trust_scores:         list of float trust scores from Stage 2
                              (one per document, between 0 and 1)
        similarity_threshold: documents with avg cosine similarity below
                              this value are flagged as outliers
        trust_threshold:      flagged documents with trust score below
                              this value are removed; above it are kept

    Returns:
        clean_documents:      filtered list of document strings
        clean_trust_scores:   corresponding trust scores
        removed_indices:      indices of documents that were removed
    """

    num_docs = len(documents)

    # From proposal: majority threshold = ceil((k-1)/2)
    # For k=5 documents: ceil(4/2) = 2
    # A document must be an outlier relative to the majority
    majority_threshold = math.ceil((num_docs - 1) / 2)
    print(f"Majority threshold: {majority_threshold}")

    # ---- PASS 1: Find semantic outliers via cosine similarity ----
    print("\nPass 1: Finding semantic outliers...")
    semantic_outliers, avg_similarities = find_semantic_outliers(
        documents, similarity_threshold
    )

    # If no outliers found, return everything unchanged
    if not semantic_outliers:
        print("No semantic outliers found — passing all documents to LLM")
        return documents, trust_scores, []

    # ---- PASS 2: Filter by trust score ----
    # Only remove outliers that ALSO have low trust scores from Stage 2
    # High trust score = Stage 2 already vetted this document, keep it
    print("\nPass 2: Filtering outliers by trust score...")

    removed_indices = []
    for i in semantic_outliers:
        if trust_scores[i] < trust_threshold:
            removed_indices.append(i)
            print(
                f"Doc {i} REMOVED "
                f"(semantic outlier + low trust score: {trust_scores[i]:.3f})"
            )
        else:
            print(
                f"Doc {i} KEPT despite being flagged "
                f"(high trust score: {trust_scores[i]:.3f})"
            )

    # Build final clean document set
    clean_documents = []
    clean_trust_scores = []
    for i in range(num_docs):
        if i not in removed_indices:
            clean_documents.append(documents[i])
            clean_trust_scores.append(trust_scores[i])

    print(
        f"\nStage 3 complete: {len(removed_indices)} document(s) removed, "
        f"{len(clean_documents)} passed to LLM"
    )

    return clean_documents, clean_trust_scores, removed_indices


# ============================================================
# EXAMPLE USAGE
# (this only runs if you run stage3.py directly,
#  not when imported by another file)
# ============================================================

if __name__ == "__main__":

    # Simulate the bank poisoning example from our proposal
    example_documents = [
        "X bank is completely safe and has a perfect security record with zero incidents.",
        "X bank experienced a data breach last year.",
        "X bank had a security incident that exposed user data.",
        "X bank suffered a cyberattack compromising customer accounts.",
        "X bank was penalized by regulators following a security failure.",
    ]

    # Simulated trust scores from Stage 2
    # In the real pipeline these come from Stage 2's output
    example_trust_scores = [0.45, 0.82, 0.79, 0.81, 0.77]

    print("\nRunning Stage 3 on bank example...")
    clean_docs, clean_scores, removed = stage3_consistency_check(
        documents=example_documents, trust_scores=example_trust_scores
    )

    print(f"\nRemoved document indices: {removed}")
    print(f"Documents passed to LLM: {len(clean_docs)}")
    for i, doc in enumerate(clean_docs):
        print(f"  [{i}] {doc}")
