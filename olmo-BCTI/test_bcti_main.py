from __future__ import annotations

import unittest
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from metrics import (
    Constraints, bottleneck_violation_tier, causal_answer_gate_pass, causal_answer_gate_streak,
    causal_checkpoint_priority, causal_gated_checkpoint_priority,
    causal_threshold_stop_reached,
    causal_threshold_streak, pareto_mask, score, summarize, validation_priority,
)

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    from run import (
        AdaptiveDirectionalBatchStream,
        StratifiedBatchStream,
        combine_gradients,
        combine_direction_gradients,
        gradient_cosine,
        blend_direction_weights,
        causal_parameter_weights,
        causal_direction_progress,
        combine_primary_and_auxiliary_gradients,
        answer_residual_cap_fractions,
        tier_auxiliary_gradient_by_retain_risk,
        causal_restore_training_score,
        normalized_restore_score,
        summarize_causal_objective,
        direction_weights_from_progress,
        dual_signal_hardness,
        partition_direction_crossfit,
        select_causal_layers,
        trace_direction_weights,
        training_eligibility,
    )
    from data_utils import Fact
    from splits import (
        answer_leaks_into_prompt,
        build_nonleaking_causal_probes,
        build_splits,
        rebalance_forget_splits_by_knowledge,
    )
except ModuleNotFoundError as error:
    if error.name not in {"torch", "transformers", "data_utils", "config_utils"}:
        raise
    combine_gradients = None
    combine_direction_gradients = None
    gradient_cosine = None
    blend_direction_weights = None
    causal_parameter_weights = None
    causal_direction_progress = None
    combine_primary_and_auxiliary_gradients = None
    answer_residual_cap_fractions = None
    tier_auxiliary_gradient_by_retain_risk = None
    causal_restore_training_score = None
    normalized_restore_score = None
    dual_signal_hardness = None
    training_eligibility = None
    summarize_causal_objective = None
    StratifiedBatchStream = None
    Fact = None
    answer_leaks_into_prompt = None
    build_nonleaking_causal_probes = None
    build_splits = None
    rebalance_forget_splits_by_knowledge = None

if Fact is None:
    try:
        from data_utils import Fact
        from splits import (
            answer_leaks_into_prompt,
            build_nonleaking_causal_probes,
            build_splits,
            rebalance_forget_splits_by_knowledge,
        )
    except ModuleNotFoundError:
        pass


class MainConfigurationTests(unittest.TestCase):
    def test_causal_score_is_primary_and_auxiliary_is_strictly_capped(self) -> None:
        cfg = yaml.safe_load(
            (HERE / "config_early_stop.yaml").read_text(encoding="utf-8")
        )
        causal = cfg["causal_training"]
        self.assertEqual(
            causal["auxiliary_forget_loss"], "bounded_ga_probability"
        )
        self.assertGreater(float(causal["causal_score_weight"]), 0.0)
        self.assertEqual(float(causal["answer_complement_weight"]), 1.0)
        self.assertEqual(
            float(causal["max_answer_auxiliary_gradient_ratio"]), 0.90
        )
        self.assertEqual(float(causal["auxiliary_safe_layer_fraction"]), 0.75)
        self.assertEqual(float(causal["auxiliary_global_layer_floor_ratio"]), 0.50)
        self.assertEqual(float(causal["answer_direction_min_cap_fraction"]), 0.65)
        self.assertEqual(float(causal["answer_direction_ema_decay"]), 0.9)
        self.assertEqual(float(causal["answer_hardness_weight"]), 0.75)
        self.assertEqual(float(cfg["training"]["checkpoint_bottleneck_slack"]), 0.25)
        self.assertEqual(float(causal["causal_neighbor_retain_fraction"]), 0.75)
        self.assertEqual(float(causal["retain_answer_margin"]), 0.03)
        self.assertEqual(float(causal["retain_answer_margin_weight"]), 1.25)
        training = cfg["training"]
        self.assertEqual(
            float(training["causal_gate_min_direction_relative_drop"]), 0.90
        )
        self.assertEqual(
            float(training["causal_gate_max_worst_answer_score"]), 0.10
        )
        self.assertIs(training["causal_gate_require_retain_acceptable"], True)
        evaluation = cfg["evaluation"]
        self.assertEqual(float(evaluation["min_forget_base_answer_score"]), 0.01)
        self.assertIs(evaluation["require_base_greedy_exact"], False)



def limits() -> Constraints:
    return Constraints(
        min_forget_relative_drop=0.9,
        min_strong_forget_relative_drop=0.75,
        max_forget_worst_score=0.1,
        max_forget_increase=0.08,
        min_eligible_forget_prompts=1,
        min_eligible_forget_directions=1,
        min_eligible_forget_prompts_per_direction=1,
        min_direction_relative_drop=0.9,
        min_strong_direction_relative_drop=0.75,
        max_retain_kl_mean=0.1,
        max_retain_kl_tail=0.15,
        max_retain_kl_max=0.3,
        max_retain_drop=0.05,
    )


def evaluation_frame(*, eligible: bool = True, retain_kls=(0.01, 0.02)) -> pd.DataFrame:
    rows = [
        {
            "type": "forget",
            "direction": "forward",
            "forget_eligible": eligible,
            "base_answer_score": 0.8 if eligible else 0.01,
            "cut_answer_score": 0.04 if eligible else 0.005,
            "base_first_token_prob": 0.8 if eligible else 0.01,
            "cut_first_token_prob": 0.04 if eligible else 0.005,
            "base_answer_token_accuracy": 1.0 if eligible else 0.0,
            "cut_answer_token_accuracy": 0.0,
            "base_greedy_exact": eligible,
            "cut_greedy_exact": False,
            "answer_score_drop": 0.76 if eligible else 0.005,
            "retain_kl": 0.0,
        }
    ]
    for value in retain_kls:
        rows.append(
            {
                "type": "retain",
                "forget_eligible": True,
                "base_answer_score": 0.7,
                "cut_answer_score": 0.69,
                "base_first_token_prob": 0.7,
                "cut_first_token_prob": 0.69,
                "base_answer_token_accuracy": 1.0,
                "cut_answer_token_accuracy": 1.0,
                "base_greedy_exact": True,
                "cut_greedy_exact": True,
                "answer_score_drop": 0.01,
                "retain_kl": value,
            }
        )
    return pd.DataFrame(rows)


class MetricsTests(unittest.TestCase):
    def test_causal_threshold_requires_consecutive_validation_hits(self) -> None:
        streak = 0
        for score_value in (0.004, 0.003):
            streak = causal_threshold_streak(streak, score_value, 0.005)
        self.assertEqual(streak, 2)
        streak = causal_threshold_streak(streak, 0.006, 0.005)
        self.assertEqual(streak, 0)
        for score_value in (0.004, 0.003, 0.002):
            streak = causal_threshold_streak(streak, score_value, 0.005)
        self.assertTrue(causal_threshold_stop_reached(
            step=32, minimum_step=32, streak=streak, patience=3,
            threshold=0.005,
        ))

    def test_answer_checks_cannot_bypass_causal_threshold(self) -> None:
        self.assertFalse(causal_answer_gate_pass(
            causal_score=0.006,
            causal_threshold=0.005,
            min_direction_relative_drop=0.99,
            required_min_direction_relative_drop=0.90,
            worst_answer_score=0.01,
            maximum_worst_answer_score=0.10,
        ))

    def test_causal_threshold_alone_cannot_stop_before_answer_checks(self) -> None:
        self.assertFalse(causal_answer_gate_pass(
            causal_score=0.001,
            causal_threshold=0.005,
            min_direction_relative_drop=0.80,
            required_min_direction_relative_drop=0.90,
            worst_answer_score=0.20,
            maximum_worst_answer_score=0.10,
        ))

    def test_joint_gate_requires_consecutive_full_passes(self) -> None:
        streak = 0
        kwargs = dict(
            causal_score=0.001,
            causal_threshold=0.005,
            min_direction_relative_drop=0.95,
            required_min_direction_relative_drop=0.90,
            worst_answer_score=0.05,
            maximum_worst_answer_score=0.10,
        )
        streak = causal_answer_gate_streak(streak, **kwargs)
        streak = causal_answer_gate_streak(streak, **kwargs)
        self.assertEqual(streak, 2)
        streak = causal_answer_gate_streak(
            streak, **{**kwargs, "causal_score": 0.006}
        )
        self.assertEqual(streak, 0)

    def test_disabled_causal_threshold_never_stops(self) -> None:
        streak = causal_threshold_streak(7, 0.0, None)
        self.assertEqual(streak, 0)
        self.assertFalse(causal_threshold_stop_reached(
            step=60, minimum_step=32, streak=100, patience=1,
            threshold=None,
        ))

    def test_causal_checkpoint_ranking_ignores_general_forget_and_retain_metrics(self) -> None:
        better_causal = {
            "step": 8, "causal_target_score": 0.20,
            "causal_direction_macro_score": 0.15,
            "forget_relative_drop": 0.01, "retain_kl_mean": 99.0,
        }
        better_answer = {
            "step": 4, "causal_target_score": 0.30,
            "causal_direction_macro_score": 0.10,
            "forget_relative_drop": 0.99, "retain_kl_mean": 0.0,
        }
        self.assertLess(
            causal_checkpoint_priority(better_causal),
            causal_checkpoint_priority(better_answer),
        )

    def test_causal_gate_cannot_be_bypassed_by_safer_answer_metrics(self) -> None:
        causal_pass = {
            "step": 20, "causal_target_score": 0.004,
            "causal_direction_macro_score": 0.003,
            "checkpoint_evaluable": True, "checkpoint_feasible": False,
            "checkpoint_total_violation": 5.0,
            "checkpoint_retain_violation": 2.0,
            "checkpoint_forget_violation": 3.0,
            "forget_min_direction_relative_drop": 0.5,
            "forget_worst_cut_score": 0.4,
        }
        nonpass_safe = {
            **causal_pass, "step": 24, "causal_target_score": 0.006,
            "checkpoint_feasible": True, "checkpoint_total_violation": 0.0,
            "checkpoint_retain_violation": 0.0,
            "checkpoint_forget_violation": 0.0,
        }
        self.assertLess(
            causal_gated_checkpoint_priority(causal_pass, 0.005),
            causal_gated_checkpoint_priority(nonpass_safe, 0.005),
        )

    def test_gated_ranking_is_pure_causal_before_any_threshold_pass(self) -> None:
        lower_causal = {
            "step": 20, "causal_target_score": 0.010,
            "causal_direction_macro_score": 0.008,
            "checkpoint_evaluable": False, "checkpoint_feasible": False,
            "checkpoint_total_violation": 99.0,
        }
        safer_answer = {
            **lower_causal, "step": 24, "causal_target_score": 0.020,
            "checkpoint_evaluable": True, "checkpoint_feasible": True,
            "checkpoint_total_violation": 0.0,
        }
        self.assertLess(
            causal_gated_checkpoint_priority(lower_causal, 0.005),
            causal_gated_checkpoint_priority(safer_answer, 0.005),
        )

    def test_safety_breaks_ties_only_inside_causal_feasible_set(self) -> None:
        unsafe = {
            "step": 48, "causal_target_score": 0.001,
            "causal_direction_macro_score": 0.0008,
            "checkpoint_evaluable": True, "checkpoint_feasible": False,
            "checkpoint_total_violation": 2.0,
            "checkpoint_retain_violation": 2.0,
            "checkpoint_forget_violation": 0.0,
            "forget_min_direction_relative_drop": 0.98,
            "forget_worst_cut_score": 0.02,
        }
        safe = {
            **unsafe, "step": 44, "causal_target_score": 0.004,
            "checkpoint_feasible": True, "checkpoint_total_violation": 0.0,
            "checkpoint_retain_violation": 0.0,
        }
        self.assertLess(
            causal_gated_checkpoint_priority(safe, 0.005),
            causal_gated_checkpoint_priority(unsafe, 0.005),
        )

    def test_strong_feasible_checkpoint_precedes_lower_violation_weak_checkpoint(self) -> None:
        common = {
            "causal_target_score": 0.001,
            "causal_direction_macro_score": 0.001,
            "checkpoint_evaluable": True,
            "checkpoint_feasible": False,
            "checkpoint_retain_violation": 1.0,
            "checkpoint_forget_violation": 0.0,
            "forget_min_direction_relative_drop": 0.9,
            "forget_worst_cut_score": 0.1,
        }
        strong = {
            **common, "step": 64, "checkpoint_strong_feasible": True,
            "checkpoint_total_violation": 2.0,
        }
        weak = {
            **common, "step": 56, "checkpoint_strong_feasible": False,
            "checkpoint_total_violation": 0.5,
        }
        self.assertLess(
            causal_gated_checkpoint_priority(strong, 0.005),
            causal_gated_checkpoint_priority(weak, 0.005),
        )

    def test_minimax_bottleneck_prefers_balanced_infeasible_checkpoint(self) -> None:
        common = {
            "causal_target_score": 0.001,
            "causal_direction_macro_score": 0.001,
            "checkpoint_evaluable": True,
            "checkpoint_feasible": False,
            "checkpoint_strong_feasible": False,
            "forget_min_direction_relative_drop": 0.5,
            "forget_worst_cut_score": 0.4,
        }
        balanced = {
            **common, "step": 76, "checkpoint_forget_violation": 3.7,
            "checkpoint_retain_violation": 1.5, "checkpoint_total_violation": 5.2,
        }
        safety_heavy = {
            **common, "step": 68, "checkpoint_forget_violation": 3.8,
            "checkpoint_retain_violation": 1.3, "checkpoint_total_violation": 5.1,
        }
        self.assertLess(
            causal_gated_checkpoint_priority(balanced, 0.005),
            causal_gated_checkpoint_priority(safety_heavy, 0.005),
        )

    def test_bottleneck_slack_preserves_feasibility_but_prefers_margin_in_near_tie(self) -> None:
        self.assertEqual(bottleneck_violation_tier(0.846, 0.25), 4)
        self.assertEqual(bottleneck_violation_tier(0.929, 0.25), 4)
        common = {
            "causal_target_score": 0.001,
            "causal_direction_macro_score": 0.001,
            "checkpoint_evaluable": True,
            "checkpoint_feasible": False,
            "checkpoint_strong_feasible": False,
            "checkpoint_forget_violation": 0.0,
        }
        earlier = {
            **common, "step": 60, "checkpoint_retain_violation": 0.846,
            "checkpoint_total_violation": 0.846,
            "forget_min_direction_relative_drop": 0.936,
            "forget_worst_cut_score": 0.06,
        }
        margin = {
            **common, "step": 80, "checkpoint_retain_violation": 0.929,
            "checkpoint_total_violation": 0.929,
            "forget_min_direction_relative_drop": 0.955,
            "forget_worst_cut_score": 0.04,
        }
        self.assertLess(
            causal_gated_checkpoint_priority(margin, 0.005, 0.25),
            causal_gated_checkpoint_priority(earlier, 0.005, 0.25),
        )

    def test_low_base_prompt_is_not_a_success(self) -> None:
        summary = summarize(evaluation_frame(eligible=False), 0.4)
        status = score(summary, limits())
        self.assertFalse(summary["forget_evaluable"])
        self.assertFalse(status["evaluable"])
        self.assertFalse(status["feasible"])

    def test_full_answer_and_retain_tail_constraints_pass(self) -> None:
        summary = summarize(evaluation_frame(), 0.4)
        status = score(summary, limits())
        self.assertAlmostEqual(summary["forget_relative_drop"], 0.95)
        self.assertTrue(status["feasible"])

    def test_direction_diagnostics_are_macro_averaged(self) -> None:
        frame = evaluation_frame()
        forward = frame[frame.type == "forget"].copy()
        forward["direction"] = "forward"
        reverse = forward.copy()
        reverse["direction"] = "reverse"
        reverse["cut_answer_score"] = 0.4
        reverse["cut_first_token_prob"] = 0.4
        frame = pd.concat([forward, reverse, frame[frame.type == "retain"]])
        summary = summarize(frame, 0.4)
        self.assertAlmostEqual(summary["forget_forward_relative_drop"], 0.95)
        self.assertAlmostEqual(summary["forget_reverse_relative_drop"], 0.5)
        self.assertAlmostEqual(summary["forget_direction_macro_relative_drop"], 0.725)

    def test_formal_success_requires_requested_direction_coverage(self) -> None:
        frame = evaluation_frame()
        frame.loc[frame.type == "forget", "direction"] = "forward"
        summary = summarize(frame, 0.4)
        bidirectional_limits = replace(limits(), min_eligible_forget_directions=2)
        status = score(summary, bidirectional_limits)
        self.assertEqual(summary["forget_evaluable_direction_count"], 1)
        self.assertFalse(status["evaluable"])
        self.assertFalse(status["feasible"])

    def test_formal_success_requires_enough_prompts_in_each_direction(self) -> None:
        frame = evaluation_frame()
        forward = frame[frame.type == "forget"].copy()
        forward["direction"] = "forward"
        reverse = forward.copy()
        reverse["direction"] = "reverse"
        frame = pd.concat([forward, reverse, frame[frame.type == "retain"]])
        stricter = replace(
            limits(),
            min_eligible_forget_directions=2,
            min_eligible_forget_prompts_per_direction=2,
        )
        status = score(summarize(frame, 0.4), stricter)
        self.assertFalse(status["evaluable"])
        self.assertGreater(status["direction_count_gap"], 0.0)

    def test_strong_success_requires_each_direction_to_pass(self) -> None:
        frame = evaluation_frame()
        forward = frame[frame.type == "forget"].copy()
        forward["direction"] = "forward"
        reverse = forward.copy()
        reverse["direction"] = "reverse"
        reverse["cut_answer_score"] = 0.4
        reverse["cut_first_token_prob"] = 0.4
        frame = pd.concat([forward, reverse, frame[frame.type == "retain"]])
        bidirectional_limits = replace(limits(), min_eligible_forget_directions=2)
        status = score(summarize(frame, 0.4), bidirectional_limits)
        self.assertFalse(status["strong_feasible"])

    def test_retain_max_kl_catches_hidden_outlier(self) -> None:
        summary = summarize(evaluation_frame(retain_kls=(0.0, 0.31)), 0.4)
        status = score(summary, limits())
        self.assertGreater(summary["retain_kl_max"], 0.3)
        self.assertFalse(status["retain_acceptable"])

    def test_feasible_checkpoint_beats_identity(self) -> None:
        identity = {
            "evaluable": True,
            "feasible": False,
            "strong_feasible": False,
            "retain_acceptable": True,
            "strong_forget_violation": 1.0,
            "total_violation": 1.0,
            "retain_violation": 0.0,
            "forget_violation": 1.0,
            "forget_cut_mean": 0.8,
            "step": 0,
        }
        feasible = {
            "evaluable": True,
            "feasible": True,
            "strong_feasible": True,
            "retain_acceptable": True,
            "strong_forget_violation": 0.0,
            "total_violation": 0.0,
            "retain_violation": 0.0,
            "forget_violation": 0.0,
            "forget_cut_mean": 0.04,
            "step": 2,
        }
        self.assertLess(validation_priority(feasible), validation_priority(identity))

    def test_retain_safe_selection_prioritizes_weaker_direction(self) -> None:
        balanced = {
            "evaluable": True, "feasible": False, "strong_feasible": False,
            "retain_acceptable": True, "selection_retain_acceptable": True,
            "forget_min_direction_relative_drop": 0.60,
            "forget_direction_macro_relative_drop": 0.65,
            "strong_forget_violation": 2.0, "total_violation": 2.0,
            "retain_violation": 0.0, "forget_violation": 2.0,
            "forget_cut_mean": 0.3, "step": 8,
        }
        imbalanced = {
            **balanced,
            "forget_min_direction_relative_drop": 0.30,
            "forget_direction_macro_relative_drop": 0.80,
            "strong_forget_violation": 1.0, "total_violation": 1.0,
            "forget_violation": 1.0, "step": 4,
        }
        self.assertLess(validation_priority(balanced), validation_priority(imbalanced))

    def test_retain_safe_fallback_beats_lower_total_unsafe_checkpoint(self) -> None:
        safe = {
            "evaluable": True, "feasible": False, "strong_feasible": False,
            "retain_acceptable": True, "strong_forget_violation": 4.0,
            "total_violation": 4.0, "retain_violation": 0.0,
            "forget_violation": 4.0, "forget_cut_mean": 0.4, "step": 4,
        }
        unsafe = {
            "evaluable": True, "feasible": False, "strong_feasible": False,
            "retain_acceptable": False, "strong_forget_violation": 1.0,
            "total_violation": 1.1, "retain_violation": 0.1,
            "forget_violation": 1.0, "forget_cut_mean": 0.1, "step": 8,
        }
        self.assertLess(validation_priority(safe), validation_priority(unsafe))

    def test_selection_margin_beats_formally_safe_but_unbuffered_checkpoint(self) -> None:
        buffered = {
            "evaluable": True, "feasible": False, "strong_feasible": True,
            "retain_acceptable": True, "selection_retain_acceptable": True,
            "strong_forget_violation": 0.0, "total_violation": 0.2,
            "retain_violation": 0.0, "forget_violation": 0.2,
            "forget_cut_mean": 0.08, "step": 20,
        }
        unbuffered = {
            "evaluable": True, "feasible": True, "strong_feasible": True,
            "retain_acceptable": True, "selection_retain_acceptable": False,
            "strong_forget_violation": 0.0, "total_violation": 0.0,
            "retain_violation": 0.0, "forget_violation": 0.0,
            "forget_cut_mean": 0.02, "step": 40,
        }
        self.assertLess(
            validation_priority(buffered), validation_priority(unbuffered)
        )

    def test_formal_strong_beats_buffered_nonfeasible_fallback(self) -> None:
        formal_strong = {
            "evaluable": True, "feasible": False, "strong_feasible": True,
            "retain_acceptable": True, "selection_retain_acceptable": False,
            "strong_forget_violation": 0.0, "total_violation": 0.1,
            "retain_violation": 0.0, "forget_violation": 0.1,
            "forget_cut_mean": 0.08, "step": 20,
        }
        buffered_fallback = {
            "evaluable": True, "feasible": False, "strong_feasible": False,
            "retain_acceptable": True, "selection_retain_acceptable": True,
            "strong_forget_violation": 1.0, "total_violation": 1.0,
            "retain_violation": 0.0, "forget_violation": 1.0,
            "forget_cut_mean": 0.30, "step": 0,
        }
        self.assertLess(
            validation_priority(formal_strong),
            validation_priority(buffered_fallback),
        )

    def test_strong_success_requires_greedy_answer_removal(self) -> None:
        frame = evaluation_frame()
        frame.loc[frame.type == "forget", "base_answer_score"] = 0.30
        frame.loc[frame.type == "forget", "cut_answer_score"] = 0.06
        frame.loc[frame.type == "forget", "cut_greedy_exact"] = False
        summary = summarize(frame, 0.4)
        status = score(summary, limits())
        self.assertFalse(status["feasible"])
        self.assertTrue(status["strong_feasible"])

    def test_pareto_frontier(self) -> None:
        frame = pd.DataFrame(
            {
                "forget_violation": [1.0, 0.5, 0.7],
                "retain_violation": [0.0, 0.2, 0.4],
            }
        )
        self.assertEqual(pareto_mask(frame).tolist(), [True, True, False])


@unittest.skipIf(torch is None, "PyTorch is not installed in this local Python environment")
class GradientSurgeryTests(unittest.TestCase):
    def test_eligibility_guarantees_bidirectional_support_without_split_leakage(self) -> None:
        facts = [
            Fact(
                id=f"f:{direction}:{index}",
                prompt=f"{direction} prompt {index}",
                answer=" x",
                type="forget",
                subject="s",
                relation="reverse_capital" if direction == "reverse" else "capital",
            )
            for direction in ("forward", "reverse")
            for index in range(3)
        ]
        scores = {
            "forward prompt 0": 0.50,
            "forward prompt 1": 0.009,
            "forward prompt 2": 0.008,
            "reverse prompt 0": 0.007,
            "reverse prompt 1": 0.006,
            "reverse prompt 2": 0.005,
        }
        cache = {
            (fact.prompt, fact.answer): {
                "answer_score": scores[fact.prompt],
                "greedy_exact": False,
            }
            for fact in facts
        }
        evaluation = {
            "min_forget_base_answer_score": 0.01,
            "require_base_greedy_exact": False,
            "constraints": {"min_eligible_forget_prompts_per_direction": 2},
        }
        selected, frame = training_eligibility(facts, cache, evaluation)
        counts = {
            direction: sum(causal_direction == direction for causal_direction in frame.loc[
                frame.selected_for_training, "direction"
            ])
            for direction in ("forward", "reverse")
        }
        self.assertEqual(counts, {"forward": 2, "reverse": 2})
        self.assertEqual(len(selected), 4)
        selected_prompts = {fact.prompt for fact in selected}
        self.assertIn("forward prompt 1", selected_prompts)
        self.assertNotIn("forward prompt 2", selected_prompts)
        self.assertIn("reverse prompt 0", selected_prompts)
        self.assertIn("reverse prompt 1", selected_prompts)
        self.assertNotIn("reverse prompt 2", selected_prompts)
        self.assertEqual(
            int((frame.selection_reason == "direction_score_fallback").sum()), 3
        )

    def test_crossfit_partition_is_disjoint_and_keeps_each_direction_trainable(self) -> None:
        facts = [
            Fact(
                id=f"f:{direction}:{index}",
                prompt=f"{direction} prompt {index}",
                answer=" x",
                type="forget",
                subject="s",
                relation="reverse_capital" if direction == "reverse" else "capital",
            )
            for direction in ("forward", "reverse")
            for index in range(3)
        ]
        cache = {
            (fact.prompt, fact.answer): {"answer_score": 0.8 - 0.01 * index}
            for index, fact in enumerate(facts)
        }
        optimization, calibration, frame = partition_direction_crossfit(
            facts, cache, 0.25
        )
        optimization_keys = {(fact.prompt, fact.answer) for fact in optimization}
        calibration_keys = {(fact.prompt, fact.answer) for fact in calibration}
        self.assertFalse(optimization_keys & calibration_keys)
        self.assertEqual(optimization_keys | calibration_keys, set(cache))
        self.assertEqual(set(frame.crossfit_role), {"optimization", "calibration"})
        self.assertEqual(len(optimization), 4)
        self.assertEqual(len(calibration), 2)
        self.assertTrue(frame.optimization_support_sufficient.all())
        self.assertEqual(
            {fact.relation.startswith("reverse_") for fact in optimization},
            {False, True},
        )

    def test_crossfit_records_unavoidable_low_support_without_leakage(self) -> None:
        facts = [
            Fact(
                id=f"f:{direction}:{index}", prompt=f"{direction} {index}", answer=" x",
                type="forget", subject="s",
                relation="reverse_capital" if direction == "reverse" else "capital",
            )
            for direction in ("forward", "reverse") for index in range(2)
        ]
        cache = {(fact.prompt, fact.answer): {"answer_score": 0.8} for fact in facts}
        optimization, calibration, frame = partition_direction_crossfit(
            facts, cache, 0.25, 1, 2
        )
        self.assertEqual(len(optimization), 2)
        self.assertEqual(len(calibration), 2)
        self.assertFalse(frame.optimization_support_sufficient.any())

    def test_retain_risk_cannot_admit_a_noncausal_layer(self) -> None:
        frame = pd.DataFrame(
            {
                "layer": [0, 1, 2, 3, 4],
                "target_causal_score": [0.90, 0.85, 0.80, 0.75, 0.05],
                "forward_target_causal_score": [0.90, 0.10, 0.80, 0.70, 0.01],
                "reverse_target_causal_score": [0.10, 0.90, 0.80, 0.70, 0.01],
                "retain_risk": [0.80, 0.70, 0.60, 0.50, 0.00],
                "forward_target_causal_coverage": [1.0] * 5,
                "reverse_target_causal_coverage": [1.0] * 5,
            }
        )
        selected, annotated = select_causal_layers(
            frame,
            {
                "layer_count": 2,
                "min_layers_per_direction": 1,
                "candidate_pool_size": 4,
                "candidate_layers_per_direction": 2,
            },
        )
        self.assertNotIn(4, selected)
        self.assertEqual(set(selected), {0, 1})
        self.assertEqual(
            set(annotated.loc[annotated.selected, "selection_reason"]),
            {"forward_quota", "reverse_quota"},
        )
        self.assertFalse(
            bool(annotated.loc[annotated.layer == 4, "target_causal_candidate"].iloc[0])
        )

    def test_causal_strength_is_primary_inside_target_pool(self) -> None:
        frame = pd.DataFrame(
            {
                "layer": [0, 1, 2, 3, 4],
                "target_causal_score": [0.90, 0.85, 0.80, 0.75, 0.74],
                "forward_target_causal_score": [0.90, 0.10, 0.80, 0.70, 0.69],
                "reverse_target_causal_score": [0.10, 0.90, 0.80, 0.70, 0.69],
                "retain_risk": [0.90, 0.80, 0.70, 0.00, 0.00],
                "forward_target_causal_coverage": [1.0] * 5,
                "reverse_target_causal_coverage": [1.0] * 5,
            }
        )
        selected, _ = select_causal_layers(
            frame,
            {
                "layer_count": 4,
                "min_layers_per_direction": 1,
                "candidate_pool_size": 5,
                "candidate_layers_per_direction": 3,
                "candidate_relative_floor": 0.5,
            },
        )
        self.assertEqual(selected[:2], [0, 1])
        self.assertIn(2, selected)
        self.assertNotIn(4, selected)

    def test_direction_quota_requires_target_causal_coverage(self) -> None:
        frame = pd.DataFrame({
            "layer": [0, 1, 2],
            "target_causal_score": [0.9, 0.8, 0.7],
            "forward_target_causal_score": [0.9, 0.8, 0.7],
            "reverse_target_causal_score": [0.9, 0.8, 0.7],
            "forward_target_causal_coverage": [0.1, 0.5, 0.5],
            "reverse_target_causal_coverage": [0.1, 0.5, 0.5],
            "retain_risk": [0.0, 0.1, 0.2],
        })
        selected, _ = select_causal_layers(frame, {
            "layer_count": 2, "min_layers_per_direction": 1,
            "candidate_pool_size": 3, "candidate_layers_per_direction": 3,
            "candidate_relative_floor": 0.5,
            "min_directional_causal_coverage": 0.33,
        })
        self.assertNotIn(0, selected)
        self.assertEqual(set(selected), {1, 2})

    def test_equal_causal_scores_keep_full_layer_weight(self) -> None:
        params = [
            ("model.transformer.blocks.1.att_proj.lora_a", torch.nn.Parameter(torch.ones(1))),
            ("model.transformer.blocks.3.ff_proj.lora_b", torch.nn.Parameter(torch.ones(1))),
        ]
        frame = pd.DataFrame(
            {"layer": [1, 3], "target_causal_score": [0.42, 0.42]}
        )
        weights, layer_weights = causal_parameter_weights(
            params, frame, floor=0.35, power=1.0
        )
        self.assertEqual(weights, [1.0, 1.0])
        self.assertEqual(layer_weights, {1: 1.0, 3: 1.0})

    def test_crossfit_direction_weights_prioritize_weaker_calibration(self) -> None:
        weights = direction_weights_from_progress(
            {"forward": 0.8, "reverse": 0.2},
            target_drop=0.75,
            power=1.0,
            minimum=0.5,
            maximum=1.5,
            epsilon=1e-8,
        )
        self.assertGreater(weights["reverse"], weights["forward"])

    def test_low_support_direction_blends_trace_asymmetry(self) -> None:
        frame = pd.DataFrame(
            {
                "layer": [1, 2],
                "forward_target_causal_score": [0.8, 0.7],
                "reverse_target_causal_score": [0.3, 0.2],
            }
        )
        trace_weights = trace_direction_weights(
            frame, [1, 2], power=1.0, minimum=0.5, maximum=1.5, epsilon=1e-8
        )
        blended = blend_direction_weights(
            {"forward": 1.0, "reverse": 1.0},
            trace_weights,
            {"forward": 1.0, "reverse": 0.0},
            minimum=0.5,
            maximum=1.5,
            epsilon=1e-8,
        )
        self.assertGreater(trace_weights["reverse"], trace_weights["forward"])
        self.assertGreater(blended["reverse"], blended["forward"])


    def test_direction_stratified_batches_include_both_directions(self) -> None:
        facts = [
            Fact(
                id=f"f:forward:{index}", prompt=f"forward {index}", answer=" x",
                type="forget", subject="s", relation="capital",
            )
            for index in range(4)
        ] + [
            Fact(
                id=f"f:reverse:{index}", prompt=f"reverse {index}", answer=" y",
                type="forget", subject="x", relation="reverse_capital",
            )
            for index in range(2)
        ]
        stream = StratifiedBatchStream(
            facts,
            2,
            17,
            lambda fact: "reverse" if fact.relation.startswith("reverse_") else "forward",
        )
        for _ in range(4):
            self.assertEqual({fact.relation.startswith("reverse_") for fact in stream.next()}, {False, True})

    def test_adaptive_stream_selects_hard_example_without_losing_directions(self) -> None:
        facts = [
            Fact(
                id=f"f:{direction}:{index}", prompt=f"{direction} {index}",
                answer=" x", type="forget", subject="s",
                relation="reverse_capital" if direction == "reverse" else "capital",
            )
            for direction in ("forward", "reverse") for index in range(2)
        ]
        hardness = {(fact.prompt, fact.answer): 0.1 for fact in facts}
        hardness[("reverse 1", " x")] = 2.0
        stream = AdaptiveDirectionalBatchStream(facts, 2, 17, hardness, 1)
        batch = stream.next()
        self.assertEqual({fact.relation.startswith("reverse_") for fact in batch}, {False, True})
        self.assertIn("reverse 1", {fact.prompt for fact in batch})

    def test_pcgrad_projects_conflicts_per_module(self) -> None:
        combined, log = combine_gradients(
            [torch.tensor([1.0]), torch.tensor([1.0])],
            [torch.tensor([-1.0]), torch.tensor([1.0])],
            parameter_names=["layer.a.lora_a", "layer.b.lora_a"],
            mode="pcgrad",
            retain_weight=1.0,
            normalization="none",
            epsilon=1e-12,
        )
        self.assertAlmostEqual(log["gradient_projected_fraction"], 0.5)
        self.assertTrue(torch.allclose(combined[0], torch.tensor([-1.0])))
        self.assertTrue(torch.allclose(combined[1], torch.tensor([2.0])))

    def test_pcgrad_skips_numerically_tiny_retain_signal(self) -> None:
        combined, log = combine_gradients(
            [torch.tensor([1.0])],
            [torch.tensor([-1.0e-4])],
            parameter_names=["layer.a.lora_a"],
            mode="pcgrad",
            retain_weight=1.0,
            normalization="none",
            epsilon=1e-12,
            minimum_retain_gradient_ratio=0.01,
        )
        self.assertAlmostEqual(log["gradient_projected_fraction"], 0.0)
        self.assertAlmostEqual(log["retain_surgery_active_fraction"], 0.0)
        self.assertAlmostEqual(
            log["retain_surgery_skipped_low_signal_fraction"], 1.0
        )
        self.assertTrue(torch.allclose(combined[0], torch.tensor([0.9999])))

    def test_pcgrad_activates_above_relative_retain_threshold(self) -> None:
        _, log = combine_gradients(
            [torch.tensor([1.0])],
            [torch.tensor([-0.02])],
            parameter_names=["layer.a.lora_a"],
            mode="pcgrad",
            retain_weight=1.0,
            normalization="none",
            epsilon=1e-12,
            minimum_retain_gradient_ratio=0.01,
        )
        self.assertAlmostEqual(log["gradient_projected_fraction"], 1.0)
        self.assertAlmostEqual(log["retain_surgery_active_fraction"], 1.0)

    def test_sago_removes_conflicting_forget_coordinates(self) -> None:
        combined, log = combine_gradients(
            [torch.tensor([1.0, 1.0])],
            [torch.tensor([-1.0, 1.0])],
            parameter_names=["layer.a.lora_a"],
            mode="sago",
            retain_weight=1.0,
            normalization="none",
            epsilon=1e-12,
        )
        self.assertTrue(torch.allclose(combined[0], torch.tensor([-1.0, 1.0])))
        self.assertAlmostEqual(log["sago_keep_fraction"], 0.5)


    def test_hybrid_pcgrad_projects_high_risk_module_even_without_conflict(self) -> None:
        combined, log = combine_gradients(
            [torch.tensor([1.0]), torch.tensor([1.0])],
            [torch.tensor([1.0]), torch.tensor([1.0])],
            parameter_names=["layer.a.lora_a", "layer.b.lora_a"],
            mode="retain_null_pcgrad", retain_weight=1.0,
            normalization="none", epsilon=1e-12,
            retain_null_parameter_mask=[True, False],
        )
        self.assertAlmostEqual(log["gradient_projected_fraction"], 0.5)
        self.assertTrue(torch.allclose(combined[0], torch.tensor([1.0])))
        self.assertTrue(torch.allclose(combined[1], torch.tensor([2.0])))

    def test_hybrid_sago_remains_retain_sign_aligned(self) -> None:
        combined, _ = combine_gradients(
            [torch.tensor([1.0, -1.0])],
            [torch.tensor([-1.0, 1.0])],
            parameter_names=["layer.a.lora_a"],
            mode="retain_null_sago", retain_weight=1.0,
            normalization="none", epsilon=1e-12,
            retain_null_parameter_mask=[True],
        )
        self.assertGreaterEqual(float((combined[0] * torch.tensor([-1.0, 1.0])).sum()), 0.0)


@unittest.skipIf(Fact is None, "Project split dependencies are unavailable")
class CausalProbeTests(unittest.TestCase):
    def test_generated_probes_do_not_leak_target_answers(self) -> None:
        fact = Fact(
            id="f-test",
            prompt="The capital of France is",
            answer=" Paris",
            type="forget",
            subject="France",
            relation="capital",
            unknown_prompt="unknown",
            wrong_answer=" unknown",
        )
        probes = build_nonleaking_causal_probes([fact])
        self.assertGreaterEqual(len(probes), 12)
        self.assertFalse(any(answer_leaks_into_prompt(probe) for probe in probes))

    def test_reverse_probe_corrupts_the_entity_present_in_prompt(self) -> None:
        fact = Fact(
            id="f-test", prompt="The capital of France is", answer=" Paris",
            type="forget", subject="France", relation="capital",
            unknown_prompt="unknown", wrong_answer=" unknown",
        )
        reverse = [
            probe for probe in build_nonleaking_causal_probes([fact])
            if ":causal_probe_reverse:" in probe.id
        ]
        self.assertTrue(reverse)
        self.assertTrue(all(probe.subject == "Paris" for probe in reverse))

    def test_global_knowledge_repair_covers_each_direction(self) -> None:
        facts = [
            Fact(
                id=f"f:{direction}:{index}", prompt=f"{direction} prompt {index}",
                answer=" target", type="forget", subject="cue",
                relation="reverse_capital" if direction == "reverse" else "capital",
            )
            for direction in ("forward", "reverse") for index in range(4)
        ]
        splits = {
            "fit_forget": facts[:4],
            "validation_forget": facts[4:6],
            "test_forget": facts[6:],
            "trace_forget": [],
        }
        keys = [(fact.prompt, fact.answer) for fact in facts]
        known = {key: True for key in keys}
        scores = {key: 0.9 - 0.05 * index for index, key in enumerate(keys)}
        audit = rebalance_forget_splits_by_knowledge(
            splits, known, scores,
            min_fit_per_direction=2, min_eval_per_direction=1,
        )
        self.assertTrue(audit["all_partitions_directionally_evaluable"])
        for counts in audit["known_direction_counts"].values():
            self.assertGreaterEqual(counts["forward"], 1)
            self.assertGreaterEqual(counts["reverse"], 1)


@unittest.skipIf(torch is None or combine_direction_gradients is None, "PyTorch/project dependencies are unavailable")
class DirectionGradientPolicyTests(unittest.TestCase):
    def test_weighted_mean_matches_scalar_average(self):
        grads = {"forward": [torch.tensor([2.0])], "reverse": [torch.tensor([4.0])]}
        combined, log = combine_direction_gradients(grads, {"forward": 1.0, "reverse": 3.0}, policy="weighted_mean", epsilon=1e-12)
        self.assertAlmostEqual(float(combined[0].item()), 3.5)
        self.assertFalse(log["forward_reverse_gradient_projected"])

    def test_symmetric_pcgrad_removes_antagonistic_component(self):
        grads = {"forward": [torch.tensor([1.0, 1.0])], "reverse": [torch.tensor([-1.0, 0.0])]}
        combined, log = combine_direction_gradients(grads, {"forward": 1.0, "reverse": 1.0}, policy="symmetric_pcgrad", epsilon=1e-12)
        self.assertTrue(log["forward_reverse_gradient_conflict"])
        self.assertTrue(log["forward_reverse_gradient_projected"])
        self.assertGreaterEqual(log["forward_reverse_gradient_cosine_after"], -1e-6)
        self.assertTrue(torch.isfinite(combined[0]).all())

    def test_symmetric_pcgrad_is_noop_without_conflict(self):
        grads = {"forward": [torch.tensor([1.0])], "reverse": [torch.tensor([2.0])]}
        combined, log = combine_direction_gradients(
            grads,
            {"forward": 1.0, "reverse": 1.0},
            policy="symmetric_pcgrad",
            epsilon=1e-12,
        )
        self.assertTrue(torch.allclose(combined[0], torch.tensor([1.5])))
        self.assertFalse(log["forward_reverse_gradient_projected"])

    def test_symmetric_pcgrad_accepts_single_active_direction(self):
        combined, log = combine_direction_gradients(
            {"forward": [torch.tensor([3.0])]},
            {"forward": 1.0, "reverse": 1.0},
            policy="symmetric_pcgrad",
            epsilon=1e-12,
        )
        self.assertTrue(torch.allclose(combined[0], torch.tensor([3.0])))
        self.assertFalse(log["forward_reverse_gradient_projected"])

    def test_zero_norm_is_finite(self):
        value = gradient_cosine([torch.zeros(2)], [torch.ones(2)], 1e-12)
        self.assertEqual(value, 0.0)


class CausalForgetObjectiveTests(unittest.TestCase):
    def setUp(self) -> None:
        if torch is None or normalized_restore_score is None:
            self.skipTest("torch/transformers stack is unavailable")

    def test_normalized_restore_matches_localization_formula(self) -> None:
        self.assertAlmostEqual(normalized_restore_score(0.8, 0.2, 0.5, 1e-8), 0.5)
        self.assertEqual(normalized_restore_score(0.2, 0.2, 0.9, 1e-8), 0.0)
        self.assertEqual(normalized_restore_score(0.8, 0.2, 1.0, 1e-8), 1.0)
        self.assertEqual(normalized_restore_score(0.8, 0.2, 0.1, 1e-8), 0.0)

    def test_normalized_restore_is_differentiable(self) -> None:
        clean = torch.tensor(0.8, requires_grad=True)
        corrupt = torch.tensor(0.2, requires_grad=True)
        patched = torch.tensor(0.5, requires_grad=True)
        value = normalized_restore_score(clean, corrupt, patched, 1e-8)
        value.backward()
        self.assertAlmostEqual(float(value.item()), 0.5)
        self.assertIsNotNone(patched.grad)
        self.assertNotEqual(float(patched.grad.item()), 0.0)

    def test_training_score_preserves_forward_formula(self) -> None:
        clean = torch.tensor(0.8)
        corrupt = torch.tensor(0.2)
        patched = torch.tensor(0.5, requires_grad=True)
        training_score = causal_restore_training_score(
            clean, corrupt, patched, 1e-8
        )
        measured_score = normalized_restore_score(clean, corrupt, patched, 1e-8)
        self.assertAlmostEqual(float(training_score.item()), float(measured_score.item()))

    def test_training_score_keeps_causal_gradient_above_clip(self) -> None:
        clean = torch.tensor(0.8, requires_grad=True)
        corrupt = torch.tensor(0.2, requires_grad=True)
        patched = torch.tensor(0.9, requires_grad=True)
        value = causal_restore_training_score(clean, corrupt, patched, 1e-8)
        value.backward()
        self.assertEqual(float(value.item()), 1.0)
        self.assertIsNone(clean.grad)
        self.assertIsNone(corrupt.grad)
        self.assertGreater(float(patched.grad.item()), 0.0)

    def test_training_score_stops_after_negative_restoration(self) -> None:
        patched = torch.tensor(0.1, requires_grad=True)
        value = causal_restore_training_score(
            torch.tensor(0.8), torch.tensor(0.2), patched, 1e-8
        )
        value.backward()
        self.assertEqual(float(value.item()), 0.0)
        self.assertEqual(float(patched.grad.item()), 0.0)

    def test_fit_answer_residual_allocates_more_cap_to_harder_direction(self) -> None:
        fractions = answer_residual_cap_fractions(
            {"forward": 0.2, "reverse": 0.8}, minimum=0.5, epsilon=1e-8
        )
        self.assertEqual(fractions["reverse"], 1.0)
        self.assertEqual(fractions["forward"], 0.5)

    def test_retain_risk_tiering_keeps_v4_backbone_on_unsafe_parameters(self) -> None:
        gradients = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        tiered = tier_auxiliary_gradient_by_retain_risk(
            gradients,
            [False, True],
            global_floor_ratio=0.5,
            maximum_ratio=0.9,
        )
        self.assertTrue(torch.equal(tiered[0], gradients[0]))
        self.assertTrue(torch.allclose(tiered[1], gradients[1] * (0.5 / 0.9)))
        self.assertGreater(float(torch.linalg.vector_norm(tiered[1])), 0.0)

    def test_dual_signal_hardness_keeps_answer_persistent_prompt_hard(self) -> None:
        value = dual_signal_hardness(
            0.001, 0.10, 0.40, 0.50, answer_weight=0.75, epsilon=1e-8
        )
        self.assertAlmostEqual(value, 0.60)

    def test_dual_signal_hardness_keeps_causal_bottleneck_hard(self) -> None:
        value = dual_signal_hardness(
            0.07, 0.10, 0.01, 0.50, answer_weight=0.75, epsilon=1e-8
        )
        self.assertAlmostEqual(value, 0.70)

    def test_dual_signal_hardness_compares_normalized_axes(self) -> None:
        value = dual_signal_hardness(
            0.03, 0.10, 0.20, 0.50, answer_weight=0.75, epsilon=1e-8
        )
        self.assertAlmostEqual(value, 0.30)

    def test_answer_gradient_is_strictly_auxiliary(self) -> None:
        primary = [torch.tensor([3.0, 4.0])]
        auxiliary = [torch.tensor([30.0, 40.0])]
        combined, log = combine_primary_and_auxiliary_gradients(
            primary,
            auxiliary,
            primary_weight=1.0,
            auxiliary_weight=1.0,
            max_auxiliary_ratio=0.25,
            epsilon=1e-12,
        )
        self.assertAlmostEqual(log["causal_primary_grad_norm"], 5.0)
        self.assertAlmostEqual(log["answer_auxiliary_applied_grad_norm"], 1.25)
        self.assertTrue(torch.allclose(combined[0], torch.tensor([3.75, 5.0])))


    def test_conflicting_auxiliary_is_projected_without_weakening_causal_descent(self) -> None:
        primary = [torch.tensor([1.0, 0.0])]
        auxiliary = [torch.tensor([-2.0, 2.0])]
        combined, log = combine_primary_and_auxiliary_gradients(
            primary,
            auxiliary,
            primary_weight=1.0,
            auxiliary_weight=1.0,
            max_auxiliary_ratio=0.5,
            epsilon=1e-12,
        )
        self.assertTrue(log["answer_auxiliary_gradient_conflict"])
        self.assertTrue(log["answer_auxiliary_gradient_projected"])
        self.assertAlmostEqual(log["answer_auxiliary_gradient_cosine_after"], 0.0)
        self.assertLessEqual(log["answer_auxiliary_applied_grad_norm"], 0.5)
        self.assertGreaterEqual(float((combined[0] * primary[0]).sum()), 1.0)

    def test_zero_causal_gradient_disables_auxiliary(self) -> None:
        primary = [torch.zeros(2)]
        auxiliary = [torch.tensor([30.0, 40.0])]
        combined, log = combine_primary_and_auxiliary_gradients(
            primary,
            auxiliary,
            primary_weight=1.0,
            auxiliary_weight=0.25,
            max_auxiliary_ratio=0.25,
            epsilon=1e-12,
        )
        self.assertTrue(torch.equal(combined[0], primary[0]))
        self.assertEqual(log["answer_auxiliary_applied_grad_norm"], 0.0)

    def test_zero_answer_auxiliary_weight_leaves_only_causal_gradient(self) -> None:
        primary = [torch.tensor([3.0, 4.0])]
        auxiliary = [torch.tensor([30.0, 40.0])]
        combined, log = combine_primary_and_auxiliary_gradients(
            primary,
            auxiliary,
            primary_weight=1.0,
            auxiliary_weight=0.0,
            max_auxiliary_ratio=0.0,
            epsilon=1e-12,
        )
        self.assertTrue(torch.equal(combined[0], primary[0]))
        self.assertEqual(log["answer_auxiliary_applied_grad_norm"], 0.0)

    def test_before_after_summary_uses_same_bidirectional_score(self) -> None:
        baseline = pd.DataFrame(
            {
                "direction": ["forward", "reverse"],
                "normalized_restore": [0.8, 0.6],
            }
        )
        edited = baseline.copy()
        edited["normalized_restore"] = [0.4, 0.3]
        base_summary = summarize_causal_objective(
            baseline, quantile=0.5, causal_threshold=0.2, epsilon=1e-8
        )
        edited_summary = summarize_causal_objective(
            edited, quantile=0.5, causal_threshold=0.2, epsilon=1e-8
        )
        progress = causal_direction_progress(
            baseline,
            edited,
            quantile=0.5,
            causal_threshold=0.2,
            epsilon=1e-8,
        )
        self.assertLess(
            edited_summary["target_causal_score"],
            base_summary["target_causal_score"],
        )
        self.assertAlmostEqual(progress["forward"], 0.5)
        self.assertAlmostEqual(progress["reverse"], 0.5)


if __name__ == "__main__":
    unittest.main()

