import torch
from transformers import AutoTokenizer, EsmModel


# Student model (35M)
STUDENT_MODEL_NAME = "facebook/esm2_t12_35M_UR50D"

# Teacher model (650M)
TEACHER_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"

MAX_LEN = 1024


print("Loading protein models... (first time may take long)")

# Student tokenizer (lightweight)
student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_NAME)

# Teacher tokenizer + model
teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME)
teacher_model = EsmModel.from_pretrained(TEACHER_MODEL_NAME)
teacher_model.eval()


def build_protein_tensors(sequence: str):
    """
    Returns:
        input_ids
        attention_mask
        teacher_cls
    """

    # --- Student tokenization ---
    student_encoded = student_tokenizer(
        sequence,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    input_ids = student_encoded["input_ids"]
    attention_mask = student_encoded["attention_mask"]

    # --- Teacher CLS extraction ---
    teacher_encoded = teacher_tokenizer(
        sequence,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    with torch.no_grad():
        teacher_output = teacher_model(
            input_ids=teacher_encoded["input_ids"],
            attention_mask=teacher_encoded["attention_mask"]
        )

    teacher_cls = teacher_output.last_hidden_state[:, 0, :]  # [1, 1280]

    return input_ids, attention_mask, teacher_cls