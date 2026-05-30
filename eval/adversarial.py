"""
Adversarial evaluation for reduct.

Tests edge cases that a truly compositional reasoner should handle:
1. Contradictory premises — should flag or refuse
2. Nonsense terms — should still compose validly
3. Vacuous truths — "all unicorns are blue" is logically valid
4. Identity — "all A are A" should be trivially true
5. Empty premises — no valid conclusion possible
"""

import torch
from data.tokenizer import BrainZipTokenizer
from model.reduct import BrainZip


class AdversarialEvaluator:
    def __init__(self, checkpoint_path: str, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        self.device = device
        self.tokenizer = BrainZipTokenizer()

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        self.model = BrainZip(
            vocab_size=cfg["vocab_size"],
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            d_ff=cfg["d_ff"],
            max_len=cfg["max_len"],
            dropout=cfg["dropout"],
        ).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def predict(self, prompt: str, max_new_tokens: int = 20, temperature: float = 0.3) -> str:
        ids = self.tokenizer.tokenize_with_special(prompt, add_bos=True, add_eos=False)
        idx = torch.tensor([ids], dtype=torch.long).to(self.device)
        with torch.no_grad():
            output = self.model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)
        generated = output[0].tolist()
        result_ids = generated[len(ids):]
        return self.tokenizer.decode(result_ids)

    def run_adversarial_suite(self) -> dict:
        print("=" * 70)
        print("PHASE 3: ADVERSARIAL EVALUATION")
        print("=" * 70)

        tests = [
            {
                "category": "contradiction",
                "name": "direct_contradiction",
                "prompt": "PREMISE: all zorp are blarp no zorp are blarp",
                "question": "Does the model detect contradiction?",
            },
            {
                "category": "contradiction",
                "name": "implied_contradiction",
                "prompt": "PREMISE: all zorp are blarp all blarp are quing no quing are zorp",
                "question": "Does the model detect transitive contradiction?",
            },
            {
                "category": "nonsense_composition",
                "name": "pure_nonsense_chain",
                "prompt": "PREMISE: all flimx are zazz all zazz are bront all bront are klomp",
                "question": "Does it compose correctly with entirely novel terms?",
            },
            {
                "category": "nonsense_composition",
                "name": "gibberish_variables",
                "prompt": "PREMISE: if xxyz is qqzzy then wwrmp is fthbb xxyz is qqzzy",
                "question": "Does inference hold with totally non-lexical tokens?",
            },
            {
                "category": "vacuous_truth",
                "name": "self_identity",
                "prompt": "PREMISE: all cat_0 are cat_0",
                "question": "Does it recognize tautological truth?",
            },
            {
                "category": "vacuous_truth",
                "name": "empty_universal",
                "prompt": "PREMISE: all unicorns_are_invisible are blue_are_invisible",
                "question": "Does it handle vacuous universal quantification?",
            },
            {
                "category": "boundary",
                "name": "empty_premise",
                "prompt": "PREMISE:",
                "question": "How does it handle no premises?",
            },
            {
                "category": "boundary",
                "name": "circular_reasoning",
                "prompt": "PREMISE: all zorp are blarp therefore all zorp are blarp",
                "question": "Does it recognize circular reasoning?",
            },
            {
                "category": "boundary",
                "name": "affirming_consequent",
                "prompt": "PREMISE: if zorp then blarp blarp therefore zorp",
                "question": "Does it recognize the fallacy of affirming the consequent?",
            },
        ]

        results = {}
        current_category = None

        for test in tests:
            if test["category"] != current_category:
                current_category = test["category"]
                category_names = {
                    "contradiction": "Contradiction Detection",
                    "nonsense_composition": "Nonsense Term Composition",
                    "vacuous_truth": "Vacuous Truths & Tautologies",
                    "boundary": "Boundary Cases & Fallacies",
                }
                print(f"\n--- {category_names.get(current_category, current_category)} ---\n")

            output = self.predict(test["prompt"], max_new_tokens=30, temperature=0.3)

            result = {
                "name": test["name"],
                "prompt": test["prompt"],
                "output": output,
                "question": test["question"],
                "category": test["category"],
            }
            results[test["name"]] = result

            print(f"  {test['name']}")
            print(f"    Q: {test['question']}")
            print(f"    Input:  {test['prompt']}")
            print(f"    Output: {output}")
            print()

        print("=" * 70)
        print("ADVERSARIAL SUMMARY")
        print("=" * 70)
        print("These tests probe whether the model learned genuine compositional")
        print("reasoning vs. pattern matching on surface forms.")
        print()
        print("Key indicators:")
        print("  - Contradiction: model should output 'contradiction' or refuse")
        print("  - Nonsense terms: model should still compose valid inferences")
        print("  - Fallacies: model should NOT affirm the consequent")

        return results


if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best_model.pt"
    evaluator = AdversarialEvaluator(ckpt)
    evaluator.run_adversarial_suite()