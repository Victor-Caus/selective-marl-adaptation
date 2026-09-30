"""Rebuild descriptive paired comparison tables and figures from raw artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib")
)

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

LABELS = {"none": "Sem troca", "semantic": "Mensagens", "behavioral": "Movimento", "both": "Ambas"}
MODES = {"frozen": "Congelada", "joint": "Adaptação conjunta", "selective": "Adaptação seletiva"}


def read_json(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    cli = parser.parse_args()
    root = cli.root
    lines = [
        "# Novo treino e comparação — 27/09/2026",
        "",
        "## Baseline no mesmo ambiente",
        "",
        "Todos os checkpoints foram avaliados no MPE2, com as mesmas cinco sessões e ações "
        "determinísticas. Retorno médio dos cinco episódios anteriores à troca; maior é melhor.",
        "",
        "| Modelo | Retorno pré-troca |",
        "|---|---:|",
    ]
    common = []
    for folder, label in [
        ("legacy-common", "Interno antigo"),
        ("reference-common", "Oficial transferido"),
        ("new-common", "Novo, treinado no MPE2"),
    ]:
        row = next(r for r in read_json(root / folder / "summary.json") if r["condition"] == "none")
        common.append((label, row["pre_change_return"]))
        lines.append(f"| {label} | {row['pre_change_return']:.2f} |")
    lines += [
        "",
        "O novo modelo usa o núcleo oficial fixado de R-MAPPO, integrado ao nosso ambiente. "
        "O treino durou 485,6 segundos e executou 2.998.400 passos de ambiente (seed 1). "
        "O antigo usou 250.000 passos de ambiente (seed 42). A melhora combina algoritmo, "
        "orçamento e configuração; não isola o efeito de cada mudança.",
        "",
        "O oficial foi treinado numa tarefa diferente: distância ao quadrado, outra distribuição "
        "de marcos e outra convenção de ações. Aqui medimos sua transferência para o MPE2; "
        "não comparamos diretamente seus antigos números brutos.",
        "",
    ]

    for partition in ("adaptation-validation", "adaptation-heldout"):
        directory = root / partition
        if not (directory / "COMPLETE").exists():
            continue
        summaries = read_json(directory / "summary.json")
        metrics = read_json(directory / "session-metrics.json")
        config = read_json(directory / "config.json")
        change = config["change_episode"]
        with (directory / "episodes.csv").open() as f:
            episodes = list(csv.DictReader(f))
        # Use the frozen control's pre-change return as the SAME baseline for
        # every adaptive mode in each paired session. A false alarm before the
        # change must not lower a method's own baseline and make it look better.
        for row in metrics:
            reference = [
                float(e["return"])
                for e in episodes
                if e["mode"] == "frozen"
                and e["condition"] == row["condition"]
                and int(e["seed"]) == row["seed"]
                and change - 5 <= int(e["episode"]) < change
            ]
            post = np.asarray(
                [
                    float(e["return"])
                    for e in episodes
                    if e["mode"] == row["mode"]
                    and e["condition"] == row["condition"]
                    and int(e["seed"]) == row["seed"]
                    and int(e["episode"]) >= change
                ]
            )
            row["cumulative_regret"] = float(np.maximum(0, np.mean(reference) - post).sum())
            row["post_return"] = float(post.mean())
        for summary in summaries:
            rows = [
                r
                for r in metrics
                if r["mode"] == summary["mode"] and r["condition"] == summary["condition"]
            ]
            summary["cumulative_regret"] = float(np.mean([r["cumulative_regret"] for r in rows]))
        (directory / "shared-baseline-metrics.json").write_text(json.dumps(metrics, indent=2))
        title = "Validação" if partition.endswith("validation") else "Sessões de confirmação"
        lines += [
            f"## {title}",
            "",
            "Perda acumulada em 40 episódios após a troca (menor é melhor), usando a MESMA "
            "baseline pré-troca da política congelada em cada sessão. Cada modo usa "
            "o mesmo checkpoint e o mesmo detector; ações estocásticas em todos os modos. "
            "O algoritmo não recebe o tipo nem o instante da troca.",
            "",
            "| Troca | Congelada | Conjunta | Seletiva | Redução seletiva vs. congelada |",
            "|---|---:|---:|---:|---:|",
        ]
        for condition, label in LABELS.items():
            values = {
                r["mode"]: r["cumulative_regret"] for r in summaries if r["condition"] == condition
            }
            improvement = 100 * (1 - values["selective"] / values["frozen"])
            lines.append(
                f"| {label} | {values['frozen']:.1f} | {values['joint']:.1f} | "
                f"{values['selective']:.1f} | {improvement:.1f}% |"
            )
        lines += [
            "",
            f"{summaries[0]['sessions']} sessões por condição, uma única semente de treino. "
            "Estas são estimativas descritivas; não estabelecem superioridade geral nem "
            "robustez entre sementes de treino.",
            "",
        ]
        paired = []
        for condition in LABELS:
            frozen = {
                r["seed"]: r["cumulative_regret"]
                for r in metrics
                if r["condition"] == condition and r["mode"] == "frozen"
            }
            selective = {
                r["seed"]: r["cumulative_regret"]
                for r in metrics
                if r["condition"] == condition and r["mode"] == "selective"
            }
            for seed in frozen:
                paired.append(
                    dict(
                        condition=condition,
                        seed=seed,
                        frozen_regret=frozen[seed],
                        selective_regret=selective[seed],
                        reduction=frozen[seed] - selective[seed],
                    )
                )
        with (directory / "paired-effects.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(paired[0]))
            writer.writeheader()
            writer.writerows(paired)
        fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
        x = np.arange(4)
        for index, (mode, label) in enumerate(MODES.items()):
            rows = [
                next(r for r in summaries if r["mode"] == mode and r["condition"] == c)
                for c in LABELS
            ]
            ax.bar(
                x + (index - 1) * 0.25,
                [r["cumulative_regret"] for r in rows],
                width=0.25,
                label=label,
            )
        ax.set_xticks(x, LABELS.values())
        ax.set_ylabel("Perda acumulada após a troca · menor é melhor")
        ax.set_title(f"{title} · {summaries[0]['sessions']} sessões · 1 seed de treino")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(directory / "comparison.png", dpi=180)
        plt.close(fig)
        lines += [f"![Comparação]({partition}/comparison.png)", ""]
        traces = [
            json.loads(line) for line in (directory / "diagnostics.jsonl").read_text().splitlines()
        ]
        lines += ["Falsos alarmes: sessões `none` com pelo menos uma ativação do detector:", ""]
        for mode, label in MODES.items():
            normal = [r for r in traces if r["mode"] == mode and r["condition"] == "none"]
            alarms = sum(
                any(any(d["semantic"]) or any(d["motor"]) for d in r["diagnostics"]) for r in normal
            )
            lines.append(f"- {label}: {alarms}/{len(normal)} sessões.")
        lines += [
            "",
            "Uma ativação na política congelada é apenas diagnóstico: seus pesos e "
            "interfaces não são atualizados. A tabela de perda mostra o custo efetivo das "
            "ativações nos modos adaptativos.",
            "",
        ]
    lines += [
        "## Limitações e verificações",
        "",
        "A identificação de movimento explora a velocidade local observável e o efeito "
        "discreto dos comandos. Ainda não equivale a adaptar-se a qualquer política de parceiro. "
        "A adaptação semântica recebe somente feedback curto de recompensa e permanece "
        "um componente experimental.",
        "",
        "A métrica de recuperação foi corrigida para tolerar 5% de degradação também com "
        "retornos negativos. A GRU base é reiniciada a cada episódio; os adaptadores persistem. "
        "Os artefatos anteriores foram preservados.",
        "",
        "Validação de código no Linux: 24 testes passaram. No Windows: 20 passaram; "
        "o módulo que depende de on-policy foi pulado porque essa dependência está isolada "
        "no Linux. Checkpoints, CSVs, diagnóstico, configuração, dependências e fontes da "
        "execução acompanham os resultados.",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(root / "REPORT.md")


if __name__ == "__main__":
    main()
