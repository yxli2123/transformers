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
    with torch.no_grad():
        outputs = model(output_router_logits=True, **model_input)
    active_proportion = outputs.active_proportion
    router_logits = outputs.router_logits
    print(outputs)

    router_scores = [F.softmax(router_logits_, dim=-1) for router_logits_ in router_logits]
    for i, (proportions, scores) in enumerate(zip(active_proportion, router_scores)):
        proportions = proportions.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
        for j, (proportion, score) in enumerate(zip(proportions, scores)):
            plt.bar(range(len(proportion)), proportion, label="active neuron")
            plt.bar(range(len(score)), score, label="routing score")
            plt.legend()
            plt.savefig(f"figures/layer_{i:02}-token_{j:02}.png")
            plt.close()


if __name__ == "__main__":
    main()
