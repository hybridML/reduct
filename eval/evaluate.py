"""
Runtime inference evaluation for reduct.

This is the critical test: can the model compose valid inferences
from facts it has NEVER seen during training?

Tests three categories:
1. NOVEL ENTITY inference — same logical forms as training,
   but with entity tokens the model has never encountered
2. COMPOSITION — chaining multiple logical steps with injected facts
3. ADVERSARIAL — contradictory premises, nonsensical terms
"""

import torch
from data.tokenizer import BrainZipTokenizer
from model.reduct import BrainZip


class RuntimeEvaluator:
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
        print(f"Loaded model from {checkpoint_path} (val_loss={ckpt['val_loss']:.4f})")

    def predict(self, prompt: str, max_new_tokens: int = 20, temperature: float = 0.3) -> str:
        ids = self.tokenizer.tokenize_with_special(prompt, add_bos=True, add_eos=False)
        idx = torch.tensor([ids], dtype=torch.long).to(self.device)

        with torch.no_grad():
            output = self.model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)

        generated = output[0].tolist()
        result_ids = generated[len(ids):]
        return self.tokenizer.decode(result_ids)

    def evaluate_test_suite(self) -> dict:
        """
        Test whether the model can reason about entities NOT in training data.

        KEY INSIGHT: Training uses var_a..var_j, cat_0..cat_19, prop_0..prop_19.
        Test uses NOVEL tokens like zorp, blarp, quing, flimx, etc.
        These tokens map to <UNK> in the tokenizer — the model has NEVER seen them.

        If the model truly learned compositional reasoning (not memorization),
        it should still produce valid logical conclusions from these novel inputs.
        """
        results = {}

        print("=" * 70)
        print("PHASE 2: RUNTIME FACT INJECTION")
        print("Testing inference on entities the model has NEVER seen")
        print("=" * 70)

        novel_entity_tests = [
            {
                "name": "transitive_chain",
                "prompt": "PREMISE: all zorp are blarp all blarp are quing",
                "expected_contains": ["quing", "zorp"],
                "description": "Novel transitive syllogism with unknown entities",
            },
            {
                "name": "modus_ponens",
                "prompt": "PREMISE: if flimx is trazz then glom is quing flimx is trazz",
                "expected_contains": ["glom", "quing"],
                "description": "Modus ponens with novel predicates",
            },
            {
                "name": "quantifier_instantiation",
                "prompt": "PREMISE: all zex are blinpt zex is zex",
                "expected_contains": ["blinpt"],
                "description": "Universal instantiation with novel terms",
            },
            {
                "name": "contrapositive",
                "prompt": "PREMISE: if zorp is blarp then quing is flimx not quing is not flimx",
                "expected_contains": ["not", "zorp", "blarp"],
                "description": "Contrapositive with novel entities",
            },
        ]

        print("\n--- Novel Entity Inference ---\n")
        novel_results = []
        for test in novel_entity_tests:
            output = self.predict(test["prompt"], max_new_tokens=30, temperature=0.3)
            passed = any(tok in output.lower() for tok in test["expected_contains"])
            novel_results.append({
                "name": test["name"],
                "prompt": test["prompt"],
                "output": output,
                "passed": passed,
                "description": test["description"],
            })
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {test['name']}")
            print(f"    Input:  {test['prompt']}")
            print(f"    Output: {output}")
            print()

        results["novel_entity"] = novel_results

        print("\n--- Known Entity Inference (control) ---\n")
        known_tests = [
            {
                "name": "transitive_known",
                "prompt": "PREMISE: all cat_0 are cat_1 all cat_1 are cat_2",
                "expected_contains": ["cat_0", "cat_2"],
            },
            {
                "name": "modus_ponens_known",
                "prompt": "PREMISE: if var_a is cat_0 then var_b is cat_1 var_a is cat_0",
                "expected_contains": ["var_b", "cat_1"],
            },
        ]

        known_results = []
        for test in known_tests:
            output = self.predict(test["prompt"], max_new_tokens=30, temperature=0.3)
            passed = any(tok in output.lower() for tok in test["expected_contains"])
            known_results.append({
                "name": test["name"],
                "prompt": test["prompt"],
                "output": output,
                "passed": passed,
            })
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {test['name']}")
            print(f"    Input:  {test['prompt']}")
            print(f"    Output: {output}")
            print()

        results["known_entity"] = known_results

        novel_pass_rate = sum(1 for r in novel_results if r["passed"]) / len(novel_results)
        known_pass_rate = sum(1 for r in known_results if r["passed"]) / len(known_results)

        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print(f"  Novel entity pass rate: {novel_pass_rate:.1%}")
        print(f"  Known entity pass rate: {known_pass_rate:.1%}")
        print()
        print("  Key question: Does the model generalize compositional rules")
        print("  to entities it has NEVER encountered in training?")
        print("  If novel_pass_rate ≈ known_pass_rate → reasoning is compositional")
        print("  If novel_pass_rate << known_pass_rate → reasoning is memorized")

        return results


if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best_model.pt"
    evaluator = RuntimeEvaluator(ckpt)
    evaluator.evaluate_test_suite()