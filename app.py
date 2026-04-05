import pandas as pd
import torch
import streamlit as st

from transformers import AutoTokenizer, AutoModelForSequenceClassification

TEMPERATURE: float = 1.56

def predict_all_probs(model, tokenizer, device, title: str, abstract: str = "", max_length: int = 256):
    model.eval()

    inputs = tokenizer(
        title.strip(),
        (abstract or "").strip(),
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]
        logits = logits / TEMPERATURE
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

    id2label_local = model.config.id2label

    rows = []
    for idx, p in enumerate(probs):
        rows.append({
            "topic": id2label_local[idx],
            "probability": float(p),
            "probability_percent": round(float(p) * 100, 2),
        })

    return pd.DataFrame(rows).sort_values("probability", ascending=False).reset_index(drop=True)


def predict_top(model, tokenizer, device, title: str, abstract: str = "", threshold: float = 0.95,
                max_length: int = 256):
    df = predict_all_probs(model, tokenizer, device, title, abstract, max_length=max_length)

    cumulative = 0.0
    keep_rows = []

    for _, row in df.iterrows():
        keep_rows.append(row)
        cumulative += row["probability"]
        if cumulative >= threshold:
            break

    return pd.DataFrame(keep_rows).reset_index(drop=True)


@st.cache_resource
def load_model_and_tokenizer():
    repo_id = "Svyat-dr/article-classifier-weights"
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    return tokenizer, model, device


def main():
    st.set_page_config(page_title="Paper Topic Classifier", layout="centered")
    st.title("Classification of article topics")
    st.write(
        "Enter the article title and abstract. If the abstract is empty, the model will classify by title only. "
        "This application classifies articles into the following topics: physics, mathematics, computer science, "
        "statistics, electrical engineering, biology, and economics. This model only works with English language.")

    tokenizer, model, device = load_model_and_tokenizer()

    title = st.text_area(
        "Title",
        height=100
    )

    abstract = st.text_area(
        "Abstract",
        height=200
    )
    st.write(
        "Topics are shown in descending order of probability until their cumulative probability is greater than or "
        "equal to the threshold selected by the user.")

    threshold = st.slider("Cumulative percentage of the selected topics", min_value=0.0, max_value=1.0, value=0.95,
                          step=0.01)

    classification_button = st.button("Classify")

    if classification_button:
        if not title.strip():
            st.error("A title is required.")
        else:
            with st.spinner("Reading the article title and abstract."):
                top_df = predict_top(model, tokenizer, device, title, abstract, threshold=threshold)

            top1 = top_df.iloc[0]
            st.success(f"Top-1: {top1['topic']} ({top1['probability_percent']}%)")

            st.subheader(f"Top-{round(threshold * 100, 0)}% topics.")
            st.dataframe(top_df[["topic", "probability_percent"]], use_container_width=True)


if __name__ == "__main__":
    main()
