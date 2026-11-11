"""원인 분석 결과를 리포트(md), JSON, 그림으로 저장한다. 자동 생성 파일이다.

JSON의 analysis 부분은 정답 없이 만든 분석 결과이고, evaluation 부분만 정답지로 채점한 값이다.
스펙 05의 리포트 에이전트는 analysis 부분만 입력으로 쓴다.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fabtrace.analysis.pipeline import RCAResult  # noqa: E402
from fabtrace.analysis.spc import plot_p_chart  # noqa: E402

TOP = 5


def _f(x, nd=3):
    return "< 0.0001" if isinstance(x, float) and 0 < x < 1e-4 else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def analysis_dict(result: RCAResult) -> dict:
    findings = []
    for f in result.findings:
        b = f.ranking_b.head(TOP)
        findings.append({
            "pattern": f.pattern,
            "window_fab_out_days": list(f.window),
            "wafers_in_window": f.population,
            "predicted_defects": f.defects,
            "candidates": [
                {"rank": int(r["rank"]), "chamber": r["chamber"], "passed": int(r["passed"]),
                 "defects": int(r["defect_passed"]), "defect_rate": round(float(r["defect_rate"]), 4),
                 "lift": round(float(r["lift"]), 3), "p": float(r["p"]), "q": float(r["q"]),
                 "significant": bool(r["significant"])}
                for _, r in b.iterrows()
            ],
            "naive_top": {"chamber": f.ranking_a.iloc[0]["chamber"], "defects": int(f.ranking_a.iloc[0]["defect_passed"])},
            "causes": f.causes,
            "conclusion": "cause_identified" if f.causes else "root_cause_not_confirmed",
            "containment": [
                {"chamber": c.chamber, "exposure_start": c.start.isoformat(), "exposure_end": c.end.isoformat(),
                 "buffer_hours": c.buffer_hours, "held_lots": c.lots, "recheck_wafers": int(len(c.wafers))}
                for c in f.containments
            ],
        })
    return {
        "monitoring": {p: [list(w) for w in ws] for p, ws in result.windows.items()},
        "findings": findings,
    }


def write_json(result: RCAResult, evaluation: dict, meta: dict, path: Path) -> None:
    payload = {"meta": meta, "analysis": analysis_dict(result),
               "evaluation": {"note": "정답지로 채점한 값. 분석에는 쓰이지 않았다.", **evaluation}}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def plot_commonality(finding, cause: str | None, path: Path) -> None:
    a, b = finding.ranking_a.head(10), finding.ranking_b.head(10)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors_a = ["tab:red" if c == cause else "0.7" for c in a["chamber"]]
    axes[0].barh(a["chamber"], a["defect_passed"], color=colors_a)
    axes[0].invert_yaxis()
    axes[0].set_title("Method A: defect wafers passed (top 10)")
    axes[0].set_xlabel("predicted defect wafers")
    colors_b = ["tab:red" if c == cause else ("tab:blue" if s else "0.7") for c, s in zip(b["chamber"], b["significant"])]
    axes[1].barh(b["chamber"], b["lift"], color=colors_b)
    axes[1].axvline(1, color="0.3", lw=0.8, ls="--")
    axes[1].invert_yaxis()
    axes[1].set_title("Method B: lift, sorted by q-value (top 10)")
    axes[1].set_xlabel("lift (red/blue = q < 0.05)")
    fig.suptitle(f"{finding.pattern}, fab-out days {finding.window[0]}-{finding.window[1]}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(result: RCAResult, evaluation: dict, meta: dict, name: str,
                 reports_dir: Path, figures_dir: Path) -> dict[str, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {"md": reports_dir / f"{name}.md", "json": reports_dir / f"{name}.json"}
    write_json(result, evaluation, meta, paths["json"])

    L = [f"# 원인 분석 리포트: {meta['scenario']} (seed {meta['seed']})", "",
         "> `scripts/run_rca.py`가 자동 생성한다. 직접 고치지 말 것.", "",
         f"- 예측 모델: `{meta['model']}`",
         f"- 격리 여유 시간: {result.buffer_hours:g}시간", "",
         "## 1. 감시: 패턴별 이상 구간 (팹 아웃 날짜 기준)", "",
         "| 패턴 | 이상 구간 |", "| --- | --- |"]
    for p, ws in result.windows.items():
        L.append(f"| {p} | {', '.join(f'{s}~{e}일차' for s, e in ws) if ws else '없음'} |")

    for i, f in enumerate(result.findings, 1):
        tag = f"{f.pattern.lower().replace('-', '')}_{f.window[0]}_{f.window[1]}"
        chart_png, comm_png = figures_dir / f"{name}_p_chart_{tag}.png", figures_dir / f"{name}_commonality_{tag}.png"
        plot_p_chart(result.charts[f.pattern], result.windows[f.pattern], f.pattern, chart_png)
        plot_commonality(f, f.causes[0] if f.causes else None, comm_png)
        paths[f"p_chart_{i}"], paths[f"commonality_{i}"] = chart_png, comm_png

        L += ["", f"## 2.{i} {f.pattern}, {f.window[0]}~{f.window[1]}일차", "",
              f"구간에 팹 아웃된 웨이퍼 {f.population:,}장 중 예측 {f.pattern} {f.defects}장.", "",
              f"![p 관리도](../figures/{chart_png.name})", "",
              "### 공통성 분석", "",
              f"방법 A(단순 집계) 1위: `{f.ranking_a.iloc[0]['chamber']}` (불량 웨이퍼 {int(f.ranking_a.iloc[0]['defect_passed'])}장 경유)", "",
              "방법 B 상위 후보:", "",
              "| 순위 | 챔버 | 경유 | 불량 | 불량률 | lift | p | q | 유의 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for _, r in f.ranking_b.head(TOP).iterrows():
            L.append(f"| {r['rank']} | {r['chamber']} | {r['passed']} | {r['defect_passed']} | {_f(r['defect_rate'])} "
                     f"| {_f(r['lift'], 2)} | {_f(r['p'], 4)} | {_f(r['q'], 4)} | {'O' if r['significant'] else ''} |")
        L += ["", f"![공통성 비교](../figures/{comm_png.name})", "", "### 결론과 격리 범위", ""]
        if not f.causes:
            L.append("유의한 원인 후보가 없다(q < 0.05). 원인 미확정, 추가 데이터가 필요하다.")
        for c in f.containments:
            L += [f"- 원인 후보 `{c.chamber}`: 노출 기간 {c.start:%m-%d %H:%M} ~ {c.end:%m-%d %H:%M} (여유 {c.buffer_hours:g}시간)",
                  f"  - 출하 보류 로트 {len(c.lots)}개: {', '.join(c.lots)}",
                  f"  - 재검사 웨이퍼 {len(c.wafers)}장"]

    L += ["", "## 3. 채점 (여기만 정답지를 사용한다)", "",
          f"정답: `{meta['truth_chamber']}`, {meta['truth_pattern']}, ETCH 통과 {meta['truth_window'][0]}~{meta['truth_window'][1]}일차", "",
          "| 항목 | 값 |", "| --- | --- |"]
    for k, v in evaluation.items():
        L.append(f"| {k} | {_f(v) if isinstance(v, float) else (', '.join(v) if isinstance(v, list) and v and isinstance(v[0], str) else v)} |")
    paths["md"].write_text("\n".join(L) + "\n", encoding="utf-8")
    return paths