"""
Main experiment runner for reduct.

Orchestrates: train → evaluate → adversarial test
"""

import sys
import train as train_module
from eval.evaluate import RuntimeEvaluator
from eval.adversarial import AdversarialEvaluator


def run_experiment():
    print("=" * 70)
    print("  reduct — Compositional Reasoning Without World Knowledge")
    print("=" * 70)
    print()
    print("HYPOTHESIS:")
    print("  A Transformer trained only on structural logical forms")
    print("  (no facts, no entities from the real world) can learn")
    print("  compositional reasoning that generalizes to novel entities.")
    print()
    print("If successful, this demonstrates that reasoning and knowledge")
    print("can be decoupled — enabling privacy-preserving AI that processes")
    print("encrypted/unknown data through learned reasoning patterns alone.")
    print()

    print("=" * 70)
    print("PHASE 1: Training on synthetic logic (no facts)")
    print("=" * 70)
    print()

    model = train_module.train(
        n_examples=50000,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_len=128,
        dropout=0.1,
        batch_size=32,
        epochs=20,
        lr=3e-4,
    )

    print()
    print("=" * 70)
    print("PHASE 2: Runtime inference with novel entities")
    print("=" * 70)

    evaluator = RuntimeEvaluator("checkpoints/best_model.pt")
    phase2_results = evaluator.evaluate_test_suite()

    print()
    print("=" * 70)
    print("PHASE 3: Adversarial evaluation")
    print("=" * 70)

    adv_evaluator = AdversarialEvaluator("checkpoints/best_model.pt")
    phase3_results = adv_evaluator.run_adversarial_suite()

    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print()
    print("Next steps based on results:")
    print("  1. If novel entity pass rate ≈ known → reasoning IS compositional")
    print("     → Scale up, add more logical fragments, test encryption layer")
    print("  2. If novel << known → model memorized surface forms")
    print("     → Increase training diversity, add more structural templates")
    print("  3. If both low → model too small or training insufficient")
    print("     → Increase d_model/layers, train longer")

    return phase2_results, phase3_results


if __name__ == "__main__":
    run_experiment()