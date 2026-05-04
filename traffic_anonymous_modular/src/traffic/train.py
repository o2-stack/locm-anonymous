import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.traffic.config import (
    CFG, GLOBAL_KNOWLEDGE_PATH, GLOBAL_MEMORY_DIR, GLOBAL_PHASE_ARCHIVE_PATH,
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, MAX_STEPS, N_ACTIONS, NUM_ROUNDS,
    PHASE_EPISODES, PLOT_INTERVAL, ROOT_DIR, ROUND_EPISODES, SEED, SMOOTH,
    device, ensure_output_dirs,
)
from src.traffic.discrete_sac import SACAgent
from src.traffic.env import MultiAgentTrafficEnvLite, make_preset
from src.traffic.llm import BasinLLMAnalyzer, SimpleLLMClient
from src.traffic.outputs import (
    append_phase_archive, export_round_outputs, load_knowledge_snapshot,
    plot_cognition_maturity, plot_round_comparison, save_all_agents,
)
from src.traffic.retriever import ExperienceRetriever
from src.traffic.reward import (
    BasinRewardEngine, PhaseActionBias, PhaseAdapter, PhaseSafetyDual,
    compute_phase_cost, make_phase_control_from_teacher, portrait_to_soft_target,
    vectorize_phase_summary,
)
from src.traffic.trajectory import (
    BasinKnowledgeBase, EpisodeTrajectory, PhaseSummarizer, PhaseTrajectoryRecorder,
)
from src.traffic.utils import SafeCSV, basin_is_acceptable, plot_curve, read_human_feedback, set_seed

def train_ma_sac(env, episodes, sA, sC, n_actions, outdir,
                 analyzer=None, knowledge=None, round_idx=0,
                 use_basin_reward=False, initial_phase_portrait=None,
                 baseline_ref=None):

    agents = [SACAgent(sA, sC, n_actions) for _ in range(env.n)]
    recorder = PhaseTrajectoryRecorder()
    summarizer = PhaseSummarizer()
    phase_idx = 0
    phase_summaries = []
    retriever = ExperienceRetriever(GLOBAL_PHASE_ARCHIVE_PATH)

    basin_reward_engine = BasinRewardEngine(n_agents=env.n)
    phase_action_bias = PhaseActionBias()
    phase_adapter = PhaseAdapter(in_dim=15).to(device)
    adapter_optim = optim.Adam(phase_adapter.parameters(), lr=1e-3)
    safety_dual = PhaseSafetyDual(init_lambda=0.0, lr=0.05)

    current_phase_portrait = initial_phase_portrait if initial_phase_portrait is not None else None
    current_phase_control = None
    adapter_loss_history = []
    dual_history = []

    if use_basin_reward and current_phase_portrait is not None:
        init_teacher = portrait_to_soft_target(current_phase_portrait)
        current_phase_control = make_phase_control_from_teacher(
            init_teacher, safety_penalty_weight=safety_dual.lmbda
        )
        basin_reward_engine.start_phase(current_phase_control)
        phase_action_bias.start_phase(current_phase_control)

    log = SafeCSV(
        os.path.join(outdir, "sac_log.csv"),
        ["ep", "train_reward_team", "env_reward_team", "avg_delay", "gsvar_team",
         "queue_overflow", "max_queue", "throughput", "alpha_avg", "q_mean_avg",
         "conflict", "dual_lambda", "adapter_loss"]
    )

    R_team, R_env, DELAY, GS_VAR, OVERFLOW, MAX_Q, THROUGHPUT = [], [], [], [], [], [], []
    R_i = [[] for _ in range(env.n)]
    alpha_history = []
    total_steps = 0
    best_reward = -float("inf")

    mode_str = "BASELINE-PHYSICAL" if not use_basin_reward else "ADAPTER-BASIN-REWARD"
    print(f"\n{'=' * 60}")
    print(f"Round {round_idx + 1} | Mode: {mode_str} | Episodes: {episodes}")
    print(f"Phase size: {PHASE_EPISODES} | Warmup: {CFG.warmup_steps}")
    print(f"Action space: Discrete ({n_actions} actions)")
    print(f"{'=' * 60}\n")

    for ep in range(episodes):
        ep_traj = EpisodeTrajectory(ep + round_idx * episodes)
        if use_basin_reward:
            basin_reward_engine.start_episode()

        env.set_progress(ep / max(1, episodes - 1))
        obsA, obsC = env.reset()
        ep_r_train = 0.0
        ep_r_env = 0.0
        ci_acc = 0.0
        r_agents = [0.0] * env.n
        metrics_acc = {"alpha": 0, "q_mean": 0, "count": 0}
        last_adapter_loss = adapter_loss_history[-1] if adapter_loss_history else 0.0

        for t in range(MAX_STEPS):
            actions = []
            for i in range(env.n):
                if total_steps < CFG.warmup_steps:
                    a = np.random.randint(0, n_actions)
                else:
                    a = agents[i].act(obsA[i], deterministic=False)
                actions.append(int(a))

            actions_arr = np.array(actions, dtype=np.int32)
            if use_basin_reward and current_phase_control is not None:
                actions_arr = phase_action_bias.apply(actions_arr)

            (nextA, nextC), r_env_team, done, info = env.step(actions_arr)
            ci_acc += info["conflict"]
            total_steps += 1

            if use_basin_reward:
                r_basin_step = basin_reward_engine.step_reward(info["exec_deltas"])
                r_basin_total = r_basin_step
                if done:
                    r_basin_total += basin_reward_engine.end_episode_reward()

                absolute_wall_penalty = 0.0
                if info["max_queue_ratio"] > 0.90:
                    absolute_wall_penalty += (info["max_queue_ratio"] - 0.90) * 10.0
                if info["avg_delay"] > 0.70:
                    absolute_wall_penalty += (info["avg_delay"] - 0.70) * 5.0
                if info["penalty_green"] > 0:
                    absolute_wall_penalty += float(info["penalty_green"]) * 0.5

                relative_safety_penalty = 0.0
                if baseline_ref is not None and current_phase_control is not None:
                    base_delay = float(baseline_ref.get("avg_delay", max(float(info["avg_delay"]), 1e-6)))
                    base_overflow = float(baseline_ref.get("avg_queue_overflow", max(float(info["queue_overflow"]), 1e-3)))
                    delay_excess = max(0.0, float(info["avg_delay"]) / (base_delay + 1e-6) - 1.02)
                    overflow_excess = max(0.0, float(info["queue_overflow"]) / (base_overflow + 1e-3) - 1.05)
                    relative_safety_penalty = current_phase_control.safety_penalty_weight * (
                        delay_excess + 0.5 * overflow_excess
                    )

                r_train = float(r_basin_total - absolute_wall_penalty - relative_safety_penalty)
            else:
                r_train = float(r_env_team)

            ep_traj.record_step(
                t,
                actions_arr,
                info["exec_deltas"],
                env.green_split.copy(),
                info["avg_delay"],
                float(r_train),
                info["queue_overflow"],
                info["max_queue_ratio"],
                info["throughput"],
                info["conflict"],
                info["a_sum"]
            )

            for i in range(env.n):
                agents[i].store(
                    obsA[i], obsC[i],
                    int(actions_arr[i]),
                    float(r_train),
                    nextA[i], nextC[i],
                    float(done)
                )

            for i in range(env.n):
                train_metrics = agents[i].train()
                if train_metrics:
                    metrics_acc["alpha"] += train_metrics.get("alpha", 0)
                    metrics_acc["q_mean"] += train_metrics.get("q_mean", 0)
                    metrics_acc["count"] += 1

            obsA, obsC = nextA, nextC
            ep_r_train += float(r_train)
            ep_r_env += float(r_env_team)
            for k in range(env.n):
                r_agents[k] += float(r_train)

            if done:
                break

        avg_d, gs_v, overflow, max_q, throughput = env.episode_metrics()

        ep_traj.finalize(ep_r_train, avg_d, gs_v, overflow)
        recorder.add_episode(ep_traj)

        R_team.append(ep_r_train)
        R_env.append(ep_r_env)
        DELAY.append(avg_d)
        GS_VAR.append(gs_v)
        OVERFLOW.append(overflow)
        MAX_Q.append(max_q)
        THROUGHPUT.append(throughput)
        for k in range(env.n):
            R_i[k].append(r_agents[k])

        ci_avg = ci_acc / (t + 1)
        alpha_avg = metrics_acc["alpha"] / max(1, metrics_acc["count"])
        q_mean_avg = metrics_acc["q_mean"] / max(1, metrics_acc["count"])
        alpha_history.append(alpha_avg)

        if ep_r_train > best_reward:
            best_reward = ep_r_train
            save_all_agents(agents, os.path.join(outdir, "models"), tag="best")

        log.write([
            ep + 1,
            f"{ep_r_train:.6f}",
            f"{ep_r_env:.6f}",
            f"{avg_d:.6f}",
            f"{gs_v:.6f}",
            f"{overflow:.6f}",
            f"{max_q:.6f}",
            f"{throughput:.6f}",
            f"{alpha_avg:.4f}",
            f"{q_mean_avg:.4f}",
            f"{ci_avg:.4f}",
            f"{safety_dual.lmbda:.4f}",
            f"{last_adapter_loss:.6f}",
        ])

        if (ep + 1) % 20 == 0:
            recent_r = np.mean(R_team[-20:]) if len(R_team) >= 20 else np.mean(R_team)
            print(f"Ep {ep + 1:4d}/{episodes} | TrainR: {ep_r_train:8.4f} | "
                  f"TrainR_avg20: {recent_r:8.4f} | EnvR: {ep_r_env:8.4f} | "
                  f"Delay: {avg_d:.4f} | α: {alpha_avg:.3f} | λ_dual: {safety_dual.lmbda:.3f} | Steps: {total_steps:7d}")

        if (ep + 1) % PLOT_INTERVAL == 0:
            plot_curve(
                R_team, f"Train Reward (Ep {ep + 1})",
                os.path.join(outdir, f"reward_ep{ep + 1}.png"), SMOOTH
            )

        if recorder.phase_episode_count() >= PHASE_EPISODES:
            print(f"\n{'─' * 60}")
            print(f"PHASE {phase_idx} COMPLETE ({recorder.phase_episode_count()} episodes)")
            print(f"{'─' * 60}")

            search_mode = (
                knowledge.get_search_mode(phase_idx, episodes // PHASE_EPISODES)
                if knowledge else
                {"exploration": 0.7, "verification": 0.2, "exploitation": 0.1}
            )

            prev_summary = phase_summaries[-1] if phase_summaries else None

            temp_summary = summarizer.summarize(
                recorder.get_current_phase(),
                phase_idx=phase_idx,
                knowledge=knowledge,
                prev_summary=prev_summary,
                search_mode=search_mode,
                retrieved_cases=None,
                verbose=False,
            )

            retrieved_cases = retriever.retrieve_similar_cases(temp_summary, top_k=5)

            summary = summarizer.summarize(
                recorder.get_current_phase(),
                phase_idx=phase_idx,
                knowledge=knowledge,
                prev_summary=prev_summary,
                search_mode=search_mode,
                retrieved_cases=retrieved_cases,
            )
            phase_summaries.append(summary)

            portrait = None
            teacher = None
            if analyzer:
                human_feedback = read_human_feedback("human_feedback.txt.txt", clear_after_read=True)

                acceptable = basin_is_acceptable(summary, baseline_ref)

                auto_feedback = ""
                if use_basin_reward and (baseline_ref is not None) and (not acceptable):
                    print("\n[Basin Check] Current basin is NOT physically acceptable compared to baseline.")
                    auto_feedback = (
                        "当前盆地在交通效率上不可接受，请不要继续强化当前方向。"
                        "下一阶段请减少对当前盆地的固化，转向探索替代协调模式，"
                        "不要继续单纯沿当前协同结构做强化。"
                    )
                    search_mode = {
                        "exploration": 0.7,
                        "verification": 0.2,
                        "exploitation": 0.1
                    }
                else:
                    if use_basin_reward:
                        print("\n[Basin Check] Current basin is physically acceptable.")
                    else:
                        print("\n[Basin Check] Baseline round, no basin acceptance filtering applied.")

                merged_feedback = ""
                if auto_feedback and human_feedback:
                    merged_feedback = auto_feedback + "\n" + human_feedback
                elif auto_feedback:
                    merged_feedback = auto_feedback
                else:
                    merged_feedback = human_feedback

                if merged_feedback:
                    print("\n[Human/Auto Feedback Used]:")
                    print(merged_feedback)
                else:
                    print("\n[Human/Auto Feedback] None.")

                portrait = analyzer.analyze(
                    summary,
                    knowledge,
                    search_mode,
                    human_feedback=merged_feedback
                )
                teacher = portrait_to_soft_target(portrait)

                x = vectorize_phase_summary(summary, baseline_ref, current_phase_control)
                x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
                pred = phase_adapter(x_t)
                target_t = {
                    k: torch.tensor([[teacher[k]]], dtype=torch.float32, device=device)
                    for k in ["lambda_counter", "lambda_switch", "lambda_volatility",
                              "intensity_scale", "collective_pull"]
                }
                adapter_loss = (
                    F.mse_loss(pred["lambda_counter"], target_t["lambda_counter"]) +
                    F.mse_loss(pred["lambda_switch"], target_t["lambda_switch"]) +
                    F.mse_loss(pred["lambda_volatility"], target_t["lambda_volatility"]) +
                    F.mse_loss(pred["intensity_scale"], target_t["intensity_scale"]) +
                    F.mse_loss(pred["collective_pull"], target_t["collective_pull"])
                )
                adapter_optim.zero_grad()
                adapter_loss.backward()
                torch.nn.utils.clip_grad_norm_(phase_adapter.parameters(), 1.0)
                adapter_optim.step()
                last_adapter_loss = float(adapter_loss.item())
                adapter_loss_history.append(last_adapter_loss)

                if knowledge and portrait:
                    knowledge.update_from_portrait(portrait, summary)
                append_phase_archive(
                    GLOBAL_PHASE_ARCHIVE_PATH,
                    round_idx=round_idx,
                    summary=summary,
                    portrait=portrait
                )

            if use_basin_reward:
                phase_cost = compute_phase_cost(summary, baseline_ref)
                dual_lambda = safety_dual.update(phase_cost)
                dual_history.append(dual_lambda)

                if teacher is not None:
                    x = vectorize_phase_summary(summary, baseline_ref, current_phase_control)
                    x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        pred = phase_adapter(x_t)
                    current_phase_control = PhaseControl(
                        lambda_counter=float(pred["lambda_counter"].item()),
                        lambda_switch=float(pred["lambda_switch"].item()),
                        lambda_volatility=float(pred["lambda_volatility"].item()),
                        intensity_scale=float(pred["intensity_scale"].item()),
                        collective_pull=float(pred["collective_pull"].item()),
                        safety_penalty_weight=float(dual_lambda),
                        focus=str(teacher.get("focus", "none")),
                    )
                elif current_phase_portrait is not None:
                    teacher = portrait_to_soft_target(current_phase_portrait)
                    current_phase_control = make_phase_control_from_teacher(
                        teacher, safety_penalty_weight=dual_lambda
                    )

                if current_phase_control is not None:
                    basin_reward_engine.start_phase(current_phase_control)
                    phase_action_bias.start_phase(current_phase_control)

            if portrait is not None:
                current_phase_portrait = portrait

            recorder.start_new_phase()
            phase_idx += 1
            print(f"{'─' * 60}\n")

    if recorder.phase_episode_count() > 0:
        print(f"\n{'─' * 60}")
        print(f"FINAL PARTIAL PHASE {phase_idx} ({recorder.phase_episode_count()} episodes)")
        print(f"{'─' * 60}")

        search_mode = (
            knowledge.get_search_mode(phase_idx, episodes // PHASE_EPISODES)
            if knowledge else
            {"exploration": 0.5, "verification": 0.3, "exploitation": 0.2}
        )

        prev_summary = phase_summaries[-1] if phase_summaries else None

        summary = summarizer.summarize(
            recorder.get_current_phase(),
            phase_idx=phase_idx,
            knowledge=knowledge,
            prev_summary=prev_summary,
            search_mode=search_mode,
        )
        phase_summaries.append(summary)

        if analyzer:
            human_feedback = read_human_feedback("human_feedback.txt.txt", clear_after_read=True)

            acceptable = basin_is_acceptable(summary, baseline_ref)

            auto_feedback = ""
            if use_basin_reward and (baseline_ref is not None) and (not acceptable):
                print("\n[Basin Check] Current basin is NOT physically acceptable compared to baseline.")
                auto_feedback = (
                    "当前盆地在交通效率上不可接受，请不要继续强化当前方向。"
                    "下一阶段请减少对当前盆地的固化，转向探索替代协调模式，"
                    "不要继续单纯沿当前协同结构做强化。"
                )
                search_mode = {
                    "exploration": 0.7,
                    "verification": 0.2,
                    "exploitation": 0.1
                }
            else:
                if use_basin_reward:
                    print("\n[Basin Check] Current basin is physically acceptable.")
                else:
                    print("\n[Basin Check] Baseline round, no basin acceptance filtering applied.")

            merged_feedback = ""
            if auto_feedback and human_feedback:
                merged_feedback = auto_feedback + "\n" + human_feedback
            elif auto_feedback:
                merged_feedback = auto_feedback
            else:
                merged_feedback = human_feedback

            if merged_feedback:
                print("\n[Human/Auto Feedback Used]:")
                print(merged_feedback)
            else:
                print("\n[Human/Auto Feedback] None.")

            portrait = analyzer.analyze(
                summary,
                knowledge,
                search_mode,
                human_feedback=merged_feedback
            )

            if knowledge and portrait:
                knowledge.update_from_portrait(portrait, summary)
            append_phase_archive(
                GLOBAL_PHASE_ARCHIVE_PATH,
                round_idx=round_idx,
                summary=summary,
                portrait=portrait
            )

    log.close()

    plot_curve(R_team, "Train Reward", os.path.join(outdir, "train_reward.png"), SMOOTH)
    plot_curve(R_env, "Reference Env Reward", os.path.join(outdir, "env_reward.png"), SMOOTH)
    plot_curve(DELAY, "Avg Delay", os.path.join(outdir, "delay.png"), SMOOTH)
    plot_curve(GS_VAR, "Green-split Variance", os.path.join(outdir, "gs_var.png"), SMOOTH)
    plot_curve(OVERFLOW, "Queue Overflow", os.path.join(outdir, "overflow.png"), SMOOTH)
    plot_curve(alpha_history, "Alpha", os.path.join(outdir, "alpha.png"), SMOOTH)
    plot_team_and_agents(
        R_team, R_i, "Train Reward (per-agent + team)", "Reward",
        os.path.join(outdir, "reward_team_agents.png"), SMOOTH
    )

    print(f"\n{'=' * 60}")
    print(f"Round {round_idx + 1} Complete! Best TrainReward: {best_reward:.4f}, "
          f"Final50 TrainReward: {np.mean(R_team[-50:]):.4f}")
    print(f"Phases analyzed: {len(phase_summaries)}")
    if knowledge:
        print(f"Cognition maturity: {knowledge.maturity:.2f}")
    print(f"{'=' * 60}")

    export_round_outputs(outdir, phase_summaries, knowledge, round_idx=round_idx)
    save_all_agents(agents, os.path.join(outdir, "models"), tag="latest")

    return {
        "R": R_team,
        "R_ENV": R_env,
        "DELAY": DELAY,
        "GS_VAR": GS_VAR,
        "OVERFLOW": OVERFLOW,
        "MAX_Q": MAX_Q,
        "THROUGHPUT": THROUGHPUT,
        "ALPHA": alpha_history,
        "phase_summaries": phase_summaries,
        "adapter_loss_history": adapter_loss_history,
        "dual_history": dual_history,
    }



# ==============================================================================
# Main
# ==============================================================================
def main():
    set_seed(SEED)
    preset = "MEDIUM"

    llm = SimpleLLMClient(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, model=LLM_MODEL)

    knowledge = load_knowledge_snapshot(GLOBAL_KNOWLEDGE_PATH)
    if knowledge is None:
        knowledge = BasinKnowledgeBase()
        print("[Knowledge] No previous knowledge found, starting fresh.")
    else:
        print(f"[Knowledge] Loaded previous knowledge. Maturity={knowledge.maturity:.2f}")

    analyzer = BasinLLMAnalyzer(llm)

    s_actor_dim  = 5
    s_critic_dim = 9
    n_actions    = N_ACTIONS

    print(f"\n{'#' * 60}")
    print(f"#  Basin-Oriented Multi-Round Training (Discrete SAC)")
    print(f"#  Domain: Multi-Intersection Traffic Signal Control")
    print(f"#  Action space: Discrete ({N_ACTIONS} actions per agent)")
    print(f"#  Actions: {ACTION_DELTAS}")
    print(f"#  Rounds: {NUM_ROUNDS}, Episodes per round: {ROUND_EPISODES}")
    print(f"#  Phase size: {PHASE_EPISODES} episodes")
    print(f"#  LLM: {'Connected' if llm.available else 'Rule-based fallback'}")
    print(f"{'#' * 60}\n")

    all_round_results = []
    maturity_history = []

    initial_portrait = None
    baseline_ref = None

    for round_idx in range(NUM_ROUNDS):
        set_seed(SEED + round_idx)
        is_baseline = (round_idx == 0)
        episodes_this_round = ROUND_EPISODES[round_idx]

        print(f"\n{'#' * 60}")
        if is_baseline:
            print(f"#  ROUND {round_idx + 1}/{NUM_ROUNDS}: BASELINE / BASIN DISCOVERY")
            print(f"#  Train reward = physical reward (delay + green penalty)")
        else:
            print(f"#  ROUND {round_idx + 1}/{NUM_ROUNDS}: LLM-GUIDED BASIN REWARD")
            print(f"#  Train reward = basin reward")
            print(f"#  LLM cognition maturity: {knowledge.maturity:.2f}")
        print(f"{'#' * 60}\n")

        env = MultiAgentTrafficEnvLite(n_agents=N_AGENTS, hard=make_preset(preset))
        outdir = os.path.join(ROOT_DIR, f"round{round_idx + 1}_{'baseline' if is_baseline else 'guided'}")

        round_analyzer = analyzer
        use_basin_reward = (round_idx >= 1)

        if use_basin_reward and initial_portrait is None and len(knowledge.phase_portraits) > 0:
            initial_portrait = knowledge.phase_portraits[-1]

        results = train_ma_sac(
            env, episodes_this_round,
            s_actor_dim, s_critic_dim, n_actions,
            outdir,
            analyzer=round_analyzer,
            knowledge=knowledge,
            round_idx=round_idx,
            use_basin_reward=use_basin_reward,
            initial_phase_portrait=initial_portrait,
            baseline_ref=baseline_ref
        )

        if is_baseline and baseline_ref is None:
            baseline_ref = {
                "avg_reward": float(np.mean(results["R"][-50:])),
                "avg_delay": float(np.mean(results["DELAY"][-50:])),
                "avg_queue_overflow": float(np.mean(results["OVERFLOW"][-50:])),
            }
            print("\n[Baseline Reference]")
            print(json.dumps(baseline_ref, ensure_ascii=False, indent=2))

        if len(knowledge.phase_portraits) > 0:
            initial_portrait = knowledge.phase_portraits[-1]

        if round_analyzer is not None:
            round_review = analyzer.round_review(results["phase_summaries"], knowledge)
            safe_json_dump(round_review, os.path.join(outdir, "round_review.json"))

            round_reflection = analyzer.reflect_round(results["phase_summaries"], knowledge)
            safe_json_dump(round_reflection, os.path.join(outdir, "round_reflection.json"))

            if knowledge is not None:
                knowledge.update_round_reflection(round_reflection)

        np.save(os.path.join(outdir, "train_rewards.npy"), np.array(results["R"], np.float32))
        np.save(os.path.join(outdir, "env_rewards.npy"), np.array(results["R_ENV"], np.float32))
        np.save(os.path.join(outdir, "delay.npy"), np.array(results["DELAY"], np.float32))
        np.save(os.path.join(outdir, "gs_var.npy"), np.array(results["GS_VAR"], np.float32))
        np.save(os.path.join(outdir, "overflow.npy"), np.array(results["OVERFLOW"], np.float32))
        np.save(os.path.join(outdir, "alpha.npy"), np.array(results["ALPHA"], np.float32))

        all_round_results.append(results)
        maturity_history.append(knowledge.maturity)

        knowledge.save(GLOBAL_KNOWLEDGE_PATH)

        print(f"\nRound {round_idx + 1} final avg train reward: "
              f"{np.mean(results['R'][-50:]):.4f}")
        print(f"Round {round_idx + 1} final avg env reward: "
              f"{np.mean(results['R_ENV'][-50:]):.4f}\n")

    print(f"\n{'#' * 60}")
    print(f"#  ALL {NUM_ROUNDS} ROUNDS COMPLETE")
    print(f"{'#' * 60}")
    for i, res in enumerate(all_round_results):
        r50 = np.mean(res["R"][-50:])
        env50 = np.mean(res["R_ENV"][-50:])
        d50 = np.mean(res["DELAY"][-50:])
        print(f"  Round {i + 1}: train_R={r50:.4f}, env_R={env50:.4f}, Delay={d50:.4f}")
    print(f"  Cognition maturity: {knowledge.maturity:.2f}")
    print(f"  Bad patterns found: {len(knowledge.bad_patterns)}")
    print(f"  Good clues found: {len(knowledge.good_clues)}")
    if llm.available:
        print(f"  LLM calls: {llm.call_count}")
    print(f"{'#' * 60}\n")

    plot_round_comparison(
        all_round_results,
        os.path.join(ROOT_DIR, "round_comparison.png")
    )
    plot_cognition_maturity(
        maturity_history,
        os.path.join(ROOT_DIR, "cognition_maturity.png")
    )

    knowledge.save(GLOBAL_KNOWLEDGE_PATH)


if __name__ == "__main__":
    main()
