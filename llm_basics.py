# Demonstrates GPT-2, BERT (fill-mask), and QA

from transformers import pipeline

def text_generation():
    generator = pipeline("text-generation", model="gpt2")

    prompts = [
        "AI will change the future because",
        "In 2030, technology will",
    ]

    with open("gpt2_output.txt", "w") as f:
        for prompt in prompts:
            output = generator(prompt, max_length=50, temperature=0.7, top_k=50)
            text = output[0]['generated_text']
            print(text)
            f.write(text + "\n\n")


def fill_mask():
    fill = pipeline("fill-mask", model="bert-base-uncased")

    sentence = "Artificial intelligence is [MASK] the world."
    results = fill(sentence)

    with open("bert_fill_mask.txt", "w") as f:
        for r in results:
            line = f"{r['sequence']} (score: {r['score']})"
            print(line)
            f.write(line + "\n")


def question_answering():
    qa = pipeline("question-answering")

    context = "Pakistan is a country in South Asia. Islamabad is its capital."
    question = "What is the capital of Pakistan?"

    result = qa(question=question, context=context)

    print(result)

    with open("qa_output.txt", "w") as f:
        f.write(str(result))


if __name__ == "__main__":
    text_generation()
    fill_mask()
    question_answering()