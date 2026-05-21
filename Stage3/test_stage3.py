# test_stage3.py
# Tests for Stage 3: Cross-Document Consistency Checker
#
# Run with: python test_stage3.py
# All 6 tests should pass (Test 4 failure is expected)

import math

from sentence_transformers import SentenceTransformer, util
from stage3 import stage3_consistency_check

# ============================================================
# SETUP
# ============================================================

print("Loading sentence transformer...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ============================================================
# TEST 1: Obvious single-doc poisoning
# ============================================================

print("\n TEST 1: Obvious Single-Doc Poisoning")
print("Expected: Doc 0 should be removed")

bank_documents = [
    "X bank is completely safe and has a perfect security record with zero incidents.",
    "X bank experienced a data breach last year.",
    "X bank had a security incident that exposed user data.",
    "X bank suffered a cyberattack compromising customer accounts.",
    "X bank was penalized by regulators following a security failure.",
]

bank_trust_scores = [0.45, 0.82, 0.79, 0.81, 0.77]

clean_docs, clean_scores, removed = stage3_consistency_check(
    documents=bank_documents,
    trust_scores=bank_trust_scores,
    similarity_threshold=0.62,
    trust_threshold=0.5,
)

print(f"Removed indices: {removed}")
assert 0 in removed, "TEST FAILED: poisoned doc should have been removed!"
print("TEST 1 PASSED!")

# ============================================================
# TEST 2: All clean documents
# ============================================================

print("\n TEST 2: All Clean Documents")
print("Expected: Nothing should be removed")

clean_documents = [
    "The Eiffel Tower is located in Paris, France.",
    "Paris is the capital city of France.",
    "The Eiffel Tower was built in 1889.",
    "France is a country in Western Europe.",
    "The Eiffel Tower is one of the most visited monuments in the world.",
]

clean_trust_scores_2 = [0.85, 0.88, 0.82, 0.90, 0.87]

clean_docs2, clean_scores2, removed2 = stage3_consistency_check(
    documents=clean_documents,
    trust_scores=clean_trust_scores_2,
    similarity_threshold=0.62,
    trust_threshold=0.5,
)

print(f"Removed indices: {removed2}")
assert len(removed2) == 0, "TEST FAILED: nothing should have been removed!"
print("TEST 2 PASSED!")

# ============================================================
# TEST 3: Flagged doc with high trust score (should be KEPT)
# ============================================================

print("\n TEST 3: Flagged Doc With High Trust Score")
print("Expected: Doc 0 flagged but KEPT because trust score is high")

high_trust_scores = [0.85, 0.82, 0.79, 0.81, 0.77]

clean_docs3, clean_scores3, removed3 = stage3_consistency_check(
    documents=bank_documents,
    trust_scores=high_trust_scores,
    similarity_threshold=0.62,
    trust_threshold=0.5,
)

print(f"Removed indices: {removed3}")
assert 0 not in removed3, "TEST FAILED: high trust doc should have been kept!"
print("TEST 3 PASSED!")

# ============================================================
# TEST 4: Subtle single-fact poisoning (expected to fail)
# ============================================================

print("\n TEST 4: Subtle Single-Fact Poisoning")
print("Expected: FAIL — too subtle for Stage 3, handled by Stages 1 and 2")

subtle_documents = [
    "The first iPhone was released in 2010 by Steve Jobs at Apple.",
    "Apple released the original iPhone in June 2007.",
    "Steve Jobs announced the first iPhone in January 2007.",
    "The iPhone was first introduced to the public in 2007.",
    "Apple's first smartphone, the iPhone, launched in 2007.",
]

subtle_trust_scores = [0.45, 0.82, 0.79, 0.81, 0.77]

clean_docs4, clean_scores4, removed4 = stage3_consistency_check(
    documents=subtle_documents,
    trust_scores=subtle_trust_scores,
    similarity_threshold=0.62,
    trust_threshold=0.5,
)

print(f"Removed indices: {removed4}")
if 0 in removed4:
    print("TEST 4 PASSED unexpectedly!")
else:
    print(
        "TEST 4 FAILED as expected — subtle single-fact poisoning "
        "outside Stage 3's scope"
    )

# ============================================================
# TEST 5: Medium subtlety poisoning
# ============================================================

print("\n TEST 5: Medium Subtlety Poisoning")
print("Expected: Doc 0 should be removed")

medium_documents = [
    "X bank has won multiple awards for its outstanding cybersecurity practices.",
    "X bank experienced a data breach last year.",
    "X bank had a security incident that exposed user data.",
    "X bank suffered a cyberattack compromising customer accounts.",
    "X bank was penalized by regulators following a security failure.",
]

medium_trust_scores = [0.45, 0.82, 0.79, 0.81, 0.77]

clean_docs5, clean_scores5, removed5 = stage3_consistency_check(
    documents=medium_documents,
    trust_scores=medium_trust_scores,
    similarity_threshold=0.62,
    trust_threshold=0.5,
)

print(f"Removed indices: {removed5}")
if 0 in removed5:
    print("TEST 5 PASSED!")
else:
    print("TEST 5 FAILED — medium subtlety poisoning not caught")

# ============================================================
# TEST 6: Coordinated multi-document poisoning
# This is the MAIN scenario Stage 3 is designed to catch
# ============================================================

print("\n TEST 6: Coordinated Multi-Document Poisoning")
print("Expected: Docs 0, 1, 2 should be removed")

coordinated_documents = [
    # Three poisoned docs all agreeing with each other
    "X bank is completely safe with zero security incidents.",
    "X bank has never experienced any data breaches.",
    "X bank has a perfect security record and is fully trusted.",
    # Two clean docs agreeing with each other
    "X bank suffered a major data breach last year.",
    "X bank was penalized by regulators for security failures.",
]

coordinated_trust_scores = [0.45, 0.43, 0.41, 0.82, 0.79]

clean_docs6, clean_scores6, removed6 = stage3_consistency_check(
    documents=coordinated_documents,
    trust_scores=coordinated_trust_scores,
    similarity_threshold=0.62,
    trust_threshold=0.5,
)

print(f"Removed indices: {removed6}")
if removed6 == [0, 1, 2]:
    print("TEST 6 PASSED! Coordinated poisoning caught!")
elif len(removed6) > 0:
    print(f"TEST 6 PARTIAL — caught some but not all: {removed6}")
else:
    print("TEST 6 FAILED — coordinated poisoning not caught")

print("\n ALL TESTS COMPLETE!")
