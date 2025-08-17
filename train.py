from datasets import load_dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import CoSENTLoss, MultipleNegativesRankingLoss, SoftmaxLoss
from sentence_transformers.training_args import BatchSamplers

# 1. Load a model to finetune
model = SentenceTransformer("Qwen/Qwen3-Embedding-8B")

# 2. Loadseveral Datasets to train with
# (anchor, positive)
all_nli_pair_train = load_dataset("sentence-transformers/amazon-qa", "pair", split="train[:10000]")
# (sentence1, sentence2) + score
stsb_pair_score_train = load_dataset("sentence-transformers/stsb", split="train[:10000]")
# (anchor, positive)
quora_pair_train = load_dataset("sentence-transformers/quora-duplicates", "pair", split="train[:10000]")
# (query, answer)
natural_questions_train = load_dataset("sentence-transformers/natural-questions", split="train[:10000]")
rules_synthetic = load_dataset("graham-sh/frc-rules-synthetic", split="train[:10000]")


# Combine all datasets into a dictionary with dataset names to datasets
train_dataset = {
    "all-nli-pair": all_nli_pair_train,
    "stsb": stsb_pair_score_train,
    "quora": quora_pair_train,
    "natural-questions": natural_questions_train,
    "rules-synthetic": rules_synthetic
}

# 3. Load several Datasets to evaluate with
# (anchor, positive, negative)
all_nli_triplet_dev = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")
# (sentence1, sentence2, score)
stsb_pair_score_dev = load_dataset("sentence-transformers/stsb", split="validation")
# (anchor, positive)
quora_pair_dev = load_dataset("sentence-transformers/quora-duplicates", "pair", split="train[10000:11000]")
# (query, answer)
natural_questions_dev = load_dataset("sentence-transformers/natural-questions", split="train[10000:11000]")

# Use a dictionary for the evaluation dataset too, or just use one dataset or none at all
eval_dataset = {
    "all-nli-triplet": all_nli_triplet_dev,
    "stsb": stsb_pair_score_dev,
    "quora": quora_pair_dev,
    "natural-questions": natural_questions_dev,
}

# 4. Load several loss functions to train with
# (anchor, positive), (anchor, positive, negative)
mnrl_loss = MultipleNegativesRankingLoss(model)
# (sentence_A, sentence_B) + class
softmax_loss = SoftmaxLoss(model)
# (sentence_A, sentence_B) + score
cosent_loss = CoSENTLoss(model)

# Create a mapping with dataset names to loss functions, so the trainer knows which loss to apply where
# Note: You can also just use one loss if all your training/evaluation datasets use the same loss
losses = {
    "all-nli-pair": mnrl_loss,
    "stsb": cosent_loss,
    "quora": mnrl_loss,
    "natural-questions": mnrl_loss,
    "rules-synthetic": mnrl_loss
}
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="models/mpnet-base-all-nli-triplet",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if your GPU can't handle FP16
    bf16=False,  # Set to True if your GPU supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # Losses using "in-batch negatives" benefit from no duplicates
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="mpnet-base-all-nli-triplet",  # Used in W&B if `wandb` is installed
)


# 5. Define a simple trainer, although it's recommended to use one with args & evaluators
trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=losses,
    args=args
)
trainer.train()

# 6. Save the trained model and optionally push it to the Hugging Face Hub
model.save_pretrained("qwen-trained")
