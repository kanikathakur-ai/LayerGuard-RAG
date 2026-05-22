import os

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
NQ_DIR = os.path.join(DATA_DIR, "nq")
HOTPOTQA_DIR = os.path.join(DATA_DIR, "hotpotqa")
POISONED_DIR = os.path.join(DATA_DIR, "poisoned")
SYNTHETIC_TRAIN_DIR = os.path.join(DATA_DIR, "synthetic_train")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
INDEX_PATH = os.path.join(DATA_DIR, "nq", "faiss.index")
DOCS_PATH = os.path.join(DATA_DIR, "nq", "documents.jsonl")

# Retrieval
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
TOP_K_RETRIEVAL = 10

# Stage 1
STAGE1_MODEL = "microsoft/deberta-v3-base"
STAGE1_MAX_LENGTH = 512
STAGE1_THRESHOLD = 0.5  # τ₁ — tune via F2 on val set
STAGE1_TRAIN_EPOCHS = 3
STAGE1_LEARNING_RATE = 2e-5
STAGE1_BATCH_SIZE = 16
STAGE1_MODEL_DIR = os.path.join(RESULTS_DIR, "stage1_classifier")

# Stage 2
TRUST_WEIGHT_ALIGNMENT = 0.35  # wₐ
TRUST_WEIGHT_CLASSIFIER = 0.35  # w_c
TRUST_WEIGHT_COHERENCE = 0.30  # w_h
TOP_K_AFTER_TRUST = 5

# Stage 3
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
CONTRADICTION_THRESHOLD = 0.7  # τ₃
MIN_CONTRADICTIONS_TO_FLAG = 2  # ceil((k-1)/2) for k=5

# Generator
GENERATOR_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
GENERATOR_FALLBACK_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
GENERATOR_MAX_NEW_TOKENS = 100
GENERATOR_LOAD_IN_4BIT = True

# Attack
POISON_DOCS_PER_QUESTION = 5
POISON_RATIOS = [0.05, 0.10, 0.20]
NUM_TARGET_QUESTIONS = 200
NUM_TEST_QUESTIONS = 500

# Training data
SYNTHETIC_POISONED_PAIRS = 5000
SYNTHETIC_CLEAN_PAIRS = 5000
SYNTHETIC_TRAIN_SPLIT = 0.8
SYNTHETIC_VAL_SPLIT = 0.1
