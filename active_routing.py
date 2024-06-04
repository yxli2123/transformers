from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def main():
    model_name_or_path = "mistralai/Mixtral-8x7B-v0.1"
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        ),
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    text = [
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in "
        "May. How many clips did Natalia sell altogether in April and May?"
    ]

    model_input = tokenizer(text, return_tensors="pt")
    model_input = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in model_input.items()}
    outputs = model(output_router_logits=True, **model_input)
    active_proportion = outputs.active_proportion
    router_logits = outputs.router_logits
    router_scores = F.softmax(router_logits, dim=-1)

    print(outputs)

    for i, (proportion, score) in enumerate(zip(active_proportion, router_scores)):
        proportion = proportion.cpu().numpy()
        score = score.cpu().numpy()
        plt.bar(proportion, label="active neuron")
        plt.bar(score, label="routing score")
        plt.savefig(f"figures/active_routing_{i:02}.png")


if __name__ == "__main__":
    main()
