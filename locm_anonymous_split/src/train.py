import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.agents.sac import SACAgent
from src.config import (
    CFG,
    GLOBAL_KNOWLEDGE_PATH,
    GLOBAL_MEMORY_DIR,
    GLOBAL_PHASE_ARCHIVE_PATH,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    MAX_STEPS,
    N_AGENTS,
    NUM_ROUNDS,
    PHASE_EPISODES,
    PLOT_INTERVAL,
    ROOT_DIR,
    ROUND_EPISODES,
    SEED,
    SMOOTH,
    device,
    ensure_output_dirs,
)
from src.envs.power_grid_env import MultiAgentPowerGridEnvLite, make_preset
from src.locm.knowledge import BasinKnowledgeBase
from src.locm.llm import BasinLLMAnalyzer, SimpleLLMClient
from src.locm.retriever import ExperienceRetriever
from src.locm.reward import (
    BasinRewardEngine,
    PhaseActionBias,
    PhaseAdapter,
    PhaseControl,
    PhaseSafetyDual,
    compute_phase_cost,
    make_phase_control_from_teacher,
    portrait_to_soft_target,
    vectorize_phase_summary,
)
from src.locm.trajectory import EpisodeTrajectory, PhaseSummarizer, PhaseTrajectoryRecorder
from src.outputs.logging_and_plots import (
    append_phase_archive,
    export_round_outputs,
    load_knowledge_snapshot,
    plot_cognition_maturity,
    plot_curve,
    plot_round_comparison,
    plot_team_and_agents,
    save_all_agents,
)
from src.utils import SafeCSV, basin_is_acceptable, read_human_feedback, safe_json_dump, set_seed

def train_ma_sac(env, episodes, sA, sC, adim, max_a, outdir,
                 analyzer=None, knowledge=None, round_idx=0,
                 use_basin_reward=False, initial_phase_portrait=None,
                 baseline_ref=None):

    agents = [SACAgent(sA, sC, adim, max_a) for _ in range(env.n)]
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
        ["ep", "train_reward_team", "env_reward_team", "avg_vbias", "socvar_team", "p_loss",
         "v_max", "v_min", "alpha_avg", "q_mean_avg", "conflict", "dual_lambda", "adapter_loss"]
    )

    R_team, R_env, VB, SV, P_LOSS, V_MAX, V_MIN = [], [], [], [], [], [], []
    R_i = [[] for _ in range(env.n)]
    alpha_history = []
    total_steps = 0
    best_reward = -float("inf")

    mode_str = "BASELINE-PHYSICAL" if not use_basin_reward else "ADAPTER-BASIN-REWARD"
    print()
    print('=' * 60)
    print(f"Round {round_idx + 1} | Mode: {mode_str} | Episodes: {episodes}")
    print(f"Phase size: {PHASE_EPISODES} | Warmup: {CFG.warmup_steps}")
    print('=' * 60)
    print()

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
        metrics_acc = {"alpha": 0.0, "q_mean": 0.0, "count": 0}
        last_adapter_loss = adapter_loss_history[-1] if adapter_loss_history else 0.0

        for t in range(MAX_STEPS):
            actions = []
            for i in range(env.n):
                if total_steps < CFG.warmup_steps:
                    a = np.random.uniform(-max_a, max_a, size=(adim,))
                else:
                    a = agents[i].act(obsA[i], deterministic=False)
                actions.append(float(a[0]) if len(a) > 0 else float(a))

            actions_arr = np.array(actions, dtype=np.float32)
            if use_basin_reward and current_phase_control is not None:
                actions_arr = phase_action_bias.apply(actions_arr)

            (nextA, nextC), r_env_team, done, info = env.step(actions_arr)
            ci_acc += info["conflict"]
            total_steps += 1

            if use_basin_reward:
                r_basin_step = basin_reward_engine.step_reward(info["exec_actions"])
                r_basin_total = r_basin_step
                if done:
                    r_basin_total += basin_reward_engine.end_episode_reward()

                absolute_wall_penalty = 0.0
                v_max_now, v_min_now = float(info["v_max"]), float(info["v_min"])
                if v_max_now > 1.05:
                    absolute_wall_penalty += (v_max_now - 1.05) * 100.0
                if v_min_now < 0.95:
                    absolute_wall_penalty += (0.95 - v_min_now) * 100.0
                if info["penalty_soc"] > 0:
                    absolute_wall_penalty += float(info["penalty_soc"]) * 2.0

                relative_safety_penalty = 0.0
                if baseline_ref is not None and current_phase_control is not None:
                    base_vb = float(baseline_ref.get("avg_vbias", max(float(info["vbias"]), 1e-6)))
                    base_pl = float(baseline_ref.get("avg_p_loss", max(float(info["p_loss"]), 1e-6)))
                    vb_excess = max(0.0, float(info["vbias"]) / (base_vb + 1e-6) - 1.02)
                    pl_excess = max(0.0, float(info["p_loss"]) / (base_pl + 1e-6) - 1.05)
                    relative_safety_penalty = current_phase_control.safety_penalty_weight * (
                        vb_excess + 0.5 * pl_excess
                    )

                r_train = float(r_basin_total - absolute_wall_penalty - relative_safety_penalty)
            else:
                r_train = float(r_env_team)

            ep_traj.record_step(
                t,
                actions_arr,
                info["exec_actions"],
                env.soc.copy(),
                info["vbias"],
                float(r_train),
                info["p_loss"],
                info["v_max"],
                info["v_min"],
                info["conflict"],
                info["a_sum"]
            )

            for i in range(env.n):
                agents[i].store(
                    obsA[i], obsC[i],
                    np.array([actions_arr[i]], dtype=np.float32),
                    float(r_train),
                    nextA[i], nextC[i],
                    float(done)
                )

            for i in range(env.n):
                train_metrics = agents[i].train()
                if train_metrics:
                    metrics_acc["alpha"] += train_metrics.get("alpha", 0.0)
                    metrics_acc["q_mean"] += train_metrics.get("q_mean", 0.0)
                    metrics_acc["count"] += 1

            obsA, obsC = nextA, nextC
            ep_r_train += float(r_train)
            ep_r_env += float(r_env_team)
            for k in range(env.n):
                r_agents[k] += float(r_train)

            if done:
                break

        vb, sv, p_loss, v_max, v_min = env.episode_metrics()
        ep_traj.finalize(ep_r_train, vb, sv, p_loss)
        recorder.add_episode(ep_traj)

        R_team.append(ep_r_train)
        R_env.append(ep_r_env)
        VB.append(vb)
        SV.append(sv)
        P_LOSS.append(p_loss)
        V_MAX.append(v_max)
        V_MIN.append(v_min)
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
            f"{vb:.6f}",
            f"{sv:.6f}",
            f"{p_loss:.6f}",
            f"{v_max:.6f}",
            f"{v_min:.6f}",
            f"{alpha_avg:.4f}",
            f"{q_mean_avg:.4f}",
            f"{ci_avg:.4f}",
            f"{safety_dual.lmbda:.4f}",
            f"{last_adapter_loss:.6f}",
        ])

        if (ep + 1) % 20 == 0:
            recent_r = np.mean(R_team[-20:]) if len(R_team) >= 20 else np.mean(R_team)
            print(
                f"Ep {ep + 1:4d}/{episodes} | TrainR: {ep_r_train:8.4f} | "
                f"TrainR_avg20: {recent_r:8.4f} | EnvR: {ep_r_env:8.4f} | "
                f"VB: {vb:.4f} | α: {alpha_avg:.3f} | λ_dual: {safety_dual.lmbda:.3f} | Steps: {total_steps:7d}"
            )

        if (ep + 1) % PLOT_INTERVAL == 0:
            plot_curve(
                R_team, f"Train Reward (Ep {ep + 1})",
                os.path.join(outdir, f"reward_ep{ep + 1}.png"), SMOOTH
            )

        if recorder.phase_episode_count() >= PHASE_EPISODES:
            print()
            print('─' * 60)
            print(f"PHASE {phase_idx} COMPLETE ({recorder.phase_episode_count()} episodes)")
            print('─' * 60)

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
                    print()
                    print("[Basin Check] Current basin is NOT physically acceptable compared to baseline.")
                    auto_feedback = (
                        "当前盆地在物理结果上不可接受，请不要继续强化当前方向。"
                        "下一阶段请减少对当前盆地的固化，转向探索替代协调模式，"
                        "不要继续单纯沿当前协同结构做强化。"
                    )
                    search_mode = {"exploration": 0.7, "verification": 0.2, "exploitation": 0.1}
                else:
                    if use_basin_reward:
                        print()
                        print("[Basin Check] Current basin is physically acceptable.")
                    else:
                        print()
                        print("[Basin Check] Baseline round, no basin acceptance filtering applied.")

                if auto_feedback and human_feedback:
                    merged_feedback = auto_feedback + "\n" + human_feedback
                elif auto_feedback:
                    merged_feedback = auto_feedback
                else:
                    merged_feedback = human_feedback

                if merged_feedback:
                    print()
                    print("[Human/Auto Feedback Used]:")
                    print(merged_feedback)
                else:
                    print()
                    print("[Human/Auto Feedback] None.")

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
            print('─' * 60)
            print()

    if recorder.phase_episode_count() > 0:
        print()
        print('─' * 60)
        print(f"FINAL PARTIAL PHASE {phase_idx} ({recorder.phase_episode_count()} episodes)")
        print('─' * 60)

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
                print()
                print("[Basin Check] Current basin is NOT physically acceptable compared to baseline.")
                auto_feedback = (
                    "当前盆地在物理结果上不可接受，请不要继续强化当前方向。"
                    "下一阶段请减少对当前盆地的固化，转向探索替代协调模式，"
                    "不要继续单纯沿当前协同结构做强化。"
                )
                search_mode = {"exploration": 0.7, "verification": 0.2, "exploitation": 0.1}
            else:
                if use_basin_reward:
                    print()
                    print("[Basin Check] Current basin is physically acceptable.")
                else:
                    print()
                    print("[Basin Check] Baseline round, no basin acceptance filtering applied.")

            if auto_feedback and human_feedback:
                merged_feedback = auto_feedback + "\n" + human_feedback
            elif auto_feedback:
                merged_feedback = auto_feedback
            else:
                merged_feedback = human_feedback

            if merged_feedback:
                print()
                print("[Human/Auto Feedback Used]:")
                print(merged_feedback)
            else:
                print()
                print("[Human/Auto Feedback] None.")

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

        recorder.flush_remaining()
        print('─' * 60)
        print()

    log.close()

    plot_curve(R_team, "Train Reward (team)", os.path.join(outdir, "reward.png"), SMOOTH)
    plot_curve(R_env, "Reference Env Reward", os.path.join(outdir, "env_reward.png"), SMOOTH)
    plot_curve(VB, "Avg Voltage Bias", os.path.join(outdir, "vbias.png"), SMOOTH)
    plot_curve(SV, "SOC Variance", os.path.join(outdir, "socvar.png"), SMOOTH)
    plot_curve(P_LOSS, "Power Loss", os.path.join(outdir, "p_loss.png"), SMOOTH)
    plot_curve(alpha_history, "Alpha", os.path.join(outdir, "alpha.png"), SMOOTH)
    if adapter_loss_history:
        plot_curve(adapter_loss_history, "Phase Adapter Loss", os.path.join(outdir, "phase_adapter_loss.png"), 1)
    if dual_history:
        plot_curve(dual_history, "Phase Dual Lambda", os.path.join(outdir, "phase_dual_lambda.png"), 1)
    plot_team_and_agents(
        R_team, R_i, "Train Reward (per-agent + team)", "Reward",
        os.path.join(outdir, "reward_team_agents.png"), SMOOTH
    )

    print()
    print('=' * 60)
    print(
        f"Round {round_idx + 1} Complete! Best TrainReward: {best_reward:.4f}, "
        f"Final50 TrainReward: {np.mean(R_team[-50:]):.4f}"
    )
    print(f"Phases analyzed: {len(phase_summaries)}")
    if knowledge:
        print(f"Cognition maturity: {knowledge.maturity:.2f}")
    print('=' * 60)

    export_round_outputs(outdir, phase_summaries, knowledge, round_idx=round_idx)
    save_all_agents(agents, os.path.join(outdir, "models"), tag="latest")

    torch.save({
        "phase_adapter": phase_adapter.state_dict(),
        "dual_lambda": safety_dual.lmbda,
        "adapter_loss_history": adapter_loss_history,
        "dual_history": dual_history,
    }, os.path.join(outdir, "phase_controller.pt"))

    return {
        "R": R_team,
        "R_ENV": R_env,
        "VB": VB,
        "SV": SV,
        "P_LOSS": P_LOSS,
        "V_MAX": V_MAX,
        "V_MIN": V_MIN,
        "ALPHA": alpha_history,
        "phase_summaries": phase_summaries,
        "ADAPTER_LOSS": adapter_loss_history,
        "DUAL_LAMBDA": dual_history,
    }


# ==============================================================================
# Main
# ==============================================================================
def main():
    ensure_output_dirs()
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

    s_actor_dim, s_critic_dim, adim, max_a = 5, 9, 1, 1.0

    print(f"\n{'#' * 60}")
    print(f"#  Basin-Oriented Multi-Round Training")
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
            print(f"#  Train reward = physical reward")
        else:
            print(f"#  ROUND {round_idx + 1}/{NUM_ROUNDS}: LLM-GUIDED BASIN REWARD")
            print(f"#  Train reward = basin reward")
            print(f"#  LLM cognition maturity: {knowledge.maturity:.2f}")
        print(f"{'#' * 60}\n")

        env = MultiAgentPowerGridEnvLite(n_agents=N_AGENTS, hard=make_preset(preset))
        outdir = os.path.join(ROOT_DIR, f"round{round_idx + 1}_{'baseline' if is_baseline else 'guided'}")

        round_analyzer = analyzer
        use_basin_reward = (round_idx >= 1)

        # Round2开始的第一个phase，用上一轮最后一个portrait初始化
        if use_basin_reward and initial_portrait is None and len(knowledge.phase_portraits) > 0:
            initial_portrait = knowledge.phase_portraits[-1]

        results = train_ma_sac(
            env, episodes_this_round, s_actor_dim, s_critic_dim, adim, max_a,
            outdir,
            analyzer=round_analyzer,
            knowledge=knowledge,
            round_idx=round_idx,
            use_basin_reward=use_basin_reward,
            initial_phase_portrait=initial_portrait,
            baseline_ref=baseline_ref
        )
        # baseline轮结束后，建立baseline参考
        if is_baseline and baseline_ref is None:
            baseline_ref = {
                "avg_reward": float(np.mean(results["R"][-50:])),
                "avg_vbias": float(np.mean(results["VB"][-50:])),
                "avg_p_loss": float(np.mean(results["P_LOSS"][-50:])),
            }
            print("\n[Baseline Reference]")
            print(json.dumps(baseline_ref, ensure_ascii=False, indent=2))

        # baseline轮结束后，用它的最后一个portrait作为下一轮初始portrait
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
        np.save(os.path.join(outdir, "vbias.npy"), np.array(results["VB"], np.float32))
        np.save(os.path.join(outdir, "socvar.npy"), np.array(results["SV"], np.float32))
        np.save(os.path.join(outdir, "p_loss.npy"), np.array(results["P_LOSS"], np.float32))
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
        vb50 = np.mean(res["VB"][-50:])
        print(f"  Round {i + 1}: train_R={r50:.4f}, env_R={env50:.4f}, VB={vb50:.4f}")
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
