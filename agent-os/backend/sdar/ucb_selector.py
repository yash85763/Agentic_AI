"""
UCB Skill Selector for SDAR.

Replaces keyword-based skill selection in CognitiveFSLoader with an
Upper Confidence Bound (UCB1) exploration-exploitation strategy.

After ~30 runs per task type, UCB learns which skills actually improve
outcomes. Before that, unvisited skills are prioritised for exploration.

UCB1 formula:
    score(skill, task_type) = mean_reward(skill, task_type)
                            + C × sqrt( ln(N + 1) / (n + 1) )

Where:
    mean_reward  = average reward when this skill was used for this task type
    N            = total completed runs for this task type
    n            = completed runs where this skill was used for this task type
    C            = exploration constant (√2 ≈ 1.41 is the UCB1 optimal value)

When n = 0 (unvisited skill), score = ∞ — the skill is always tried before
exploitation begins. This guarantees every skill gets evaluated before the
selector starts optimising.

Phases:
    Exploration phase (n < 10 per skill): UCB heavily favours unvisited skills.
    Transition phase (10–30 runs): mean_reward and exploration balance.
    Exploitation phase (30+ runs): mean_reward dominates for well-sampled skills.

Storage: (skill_name, task_type, reward) triples in the `skill_rewards`
PostgreSQL table. One row per skill per job (not one row per run).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# UCB exploration constant — √2 is the theoretically optimal value for UCB1
UCB_EXPLORATION_CONSTANT = math.sqrt(2)

# ---------------------------------------------------------------------------
# Task type classification
# ---------------------------------------------------------------------------

_TASK_TYPE_PATTERNS: dict[str, list[str]] = {
    "expense_consolidation": [
        "expense", "consolidat", "budget", "finance", "cost",
        "spend", "reimburs", "receipt", "invoice", "purchase",
    ],
    "visualization": [
        "chart", "graph", "plot", "visual", "dashboard", "trend",
        "echarts", "histogram", "bar chart", "pie chart", "scatter",
    ],
    "reporting": [
        "report", "summary", "executive", "monthly", "quarterly",
        "annual", "fiscal", "narrative", "memo",
    ],
    "sql_analysis": [
        "sql", "query", "duckdb", "database", "table", "join",
        "aggregate", "select", "group by", "filter",
    ],
    "data_cleaning": [
        "clean", "normaliz", "standar", "deduplic", "format",
        "transform", "map column", "rename", "schema",
    ],
}


def classify_task_type(task_description: str) -> str:
    """Classify task description into one of the known task types.

    Uses keyword scoring: the type with the most keyword matches wins.
    Returns "general" if no type scores above zero.
    """
    task_lower = task_description.lower()
    scores: dict[str, int] = {t: 0 for t in _TASK_TYPE_PATTERNS}

    for task_type, keywords in _TASK_TYPE_PATTERNS.items():
        scores[task_type] = sum(1 for kw in keywords if kw in task_lower)

    best_type = max(scores, key=lambda t: scores[t])
    return best_type if scores[best_type] > 0 else "general"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class SkillStats:
    """Aggregated statistics for a (skill_name, task_type) pair."""
    skill_name: str
    task_type: str
    pull_count: int       # times this skill was selected for this task type
    mean_reward: float    # average reward across those runs
    last_reward: float    # most recent reward (for trend inspection)
    ucb_score: float      # current UCB score (set by selector, for display)


# ---------------------------------------------------------------------------
# UCB selector
# ---------------------------------------------------------------------------


class UCBSkillSelector:
    """Select skills using UCB1 exploration-exploitation balancing.

    Designed for synchronous use inside Celery tasks. Reads from and writes
    to the ``skill_rewards`` PostgreSQL table via a raw SQLAlchemy session.

    Usage::

        selector = UCBSkillSelector()

        # At task start — select best skills
        selected = selector.select_skills(
            task_description="Consolidate Q1 expense files for 8 teams",
            available_skills=["excel-ingestion.md", "data-cleaning.md", ...],
            session=db_session,
        )

        # After pipeline completion — update reward table
        selector.record_reward(
            job_id=job_id,
            skills_used=selected,
            task_type="expense_consolidation",
            reward=0.85,
            reward_breakdown={...},
            session=db_session,
        )
    """

    def __init__(self, exploration_constant: float = UCB_EXPLORATION_CONSTANT):
        self.C = exploration_constant

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def select_skills(
        self,
        task_description: str,
        available_skills: list[str],
        session: Any,
        top_k: int = 3,
        task_type: Optional[str] = None,
    ) -> list[str]:
        """Return top-k skills ordered by UCB score.

        Unvisited skills receive infinite UCB score (exploration guarantee).
        After enough pulls, mean_reward drives selection (exploitation).

        Args:
            task_description: Natural language task for type classification.
            available_skills: All skill filenames to consider.
            session: Sync SQLAlchemy session.
            top_k: Number of skills to return.
            task_type: Override task type inference (optional).

        Returns:
            List of up to top_k skill filenames, highest UCB first.
        """
        if not available_skills:
            return []

        effective_task_type = task_type or classify_task_type(task_description)
        total_pulls = self._get_total_pulls(effective_task_type, session)

        scored: list[tuple[str, float]] = []
        for skill in available_skills:
            stats = self._get_skill_stats(skill, effective_task_type, session)

            if stats.pull_count == 0:
                ucb = float("inf")
            else:
                exploitation = stats.mean_reward
                # Laplace smoothing (+1) prevents log(0)
                exploration = self.C * math.sqrt(
                    math.log(total_pulls + 1) / (stats.pull_count + 1)
                )
                ucb = exploitation + exploration

            scored.append((skill, ucb))

        # Sort descending; break ties by name for determinism
        scored.sort(key=lambda x: (-x[1] if x[1] != float("inf") else -999999.0, x[0]))

        # Bring inf-scored skills to front
        inf_skills = [s for s, sc in scored if sc == float("inf")]
        finite_skills = [s for s, sc in scored if sc != float("inf")]
        ordered = (inf_skills + finite_skills)[:top_k]

        logger.info(
            "UCB selected %d/%d skills  task_type=%s  total_pulls=%d  selected=%s",
            len(ordered), len(available_skills), effective_task_type, total_pulls, ordered,
        )
        return ordered

    def record_reward(
        self,
        job_id: str,
        skills_used: list[str],
        task_type: str,
        reward: float,
        reward_breakdown: dict[str, Any],
        session: Any,
    ) -> None:
        """Insert (skill, task_type, reward) rows into skill_rewards table.

        One row per skill used. Called after every completed pipeline run.
        Silently logs and skips on DB errors to avoid breaking the pipeline.
        """
        import json as _json
        from sqlalchemy import text

        for skill_name in skills_used:
            try:
                session.execute(
                    text("""
                        INSERT INTO skill_rewards
                            (id, job_id, skill_name, task_type, reward, reward_breakdown, created_at)
                        VALUES
                            (gen_random_uuid(), :job_id::uuid, :skill, :task_type,
                             :reward, :breakdown::jsonb, NOW())
                    """),
                    {
                        "job_id": job_id,
                        "skill": skill_name,
                        "task_type": task_type,
                        "reward": float(reward),
                        "breakdown": _json.dumps(reward_breakdown),
                    },
                )
            except Exception as exc:
                logger.error(
                    "Failed to record reward for skill=%s job=%s: %s",
                    skill_name, job_id, exc,
                )

        try:
            session.commit()
        except Exception as exc:
            logger.error("skill_rewards commit failed: %s", exc)
            session.rollback()

    # ------------------------------------------------------------------
    # Analytics / Studio display
    # ------------------------------------------------------------------

    def get_all_stats(
        self, task_type: str, session: Any
    ) -> list[SkillStats]:
        """Return UCB stats for all skills for a given task type.

        Used by the Studio UI to display learning progress.
        """
        from sqlalchemy import text

        total_pulls = self._get_total_pulls(task_type, session)

        rows = session.execute(
            text("""
                SELECT skill_name,
                       COUNT(*)        AS pull_count,
                       AVG(reward)     AS mean_reward,
                       MAX(reward)     AS last_reward
                FROM skill_rewards
                WHERE task_type = :task_type
                GROUP BY skill_name
                ORDER BY AVG(reward) DESC
            """),
            {"task_type": task_type},
        ).fetchall()

        result: list[SkillStats] = []
        for row in rows:
            n = int(row.pull_count)
            mean = float(row.mean_reward or 0)
            if n == 0:
                ucb = float("inf")
            else:
                ucb = mean + self.C * math.sqrt(math.log(total_pulls + 1) / (n + 1))

            result.append(SkillStats(
                skill_name=row.skill_name,
                task_type=task_type,
                pull_count=n,
                mean_reward=mean,
                last_reward=float(row.last_reward or 0),
                ucb_score=round(ucb, 4) if ucb != float("inf") else 9999.0,
            ))

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_total_pulls(self, task_type: str, session: Any) -> int:
        """Count distinct job runs for this task type."""
        from sqlalchemy import text
        row = session.execute(
            text("SELECT COUNT(DISTINCT job_id) FROM skill_rewards WHERE task_type = :t"),
            {"t": task_type},
        ).fetchone()
        return int(row[0]) if row else 0

    def _get_skill_stats(
        self, skill_name: str, task_type: str, session: Any
    ) -> SkillStats:
        """Fetch aggregate stats for one (skill, task_type) pair."""
        from sqlalchemy import text
        row = session.execute(
            text("""
                SELECT COUNT(*)        AS pull_count,
                       COALESCE(AVG(reward), 0.0) AS mean_reward,
                       COALESCE(MAX(reward), 0.0) AS last_reward
                FROM skill_rewards
                WHERE skill_name = :skill AND task_type = :task_type
            """),
            {"skill": skill_name, "task_type": task_type},
        ).fetchone()

        if not row or int(row.pull_count) == 0:
            return SkillStats(
                skill_name=skill_name,
                task_type=task_type,
                pull_count=0,
                mean_reward=0.0,
                last_reward=0.0,
                ucb_score=float("inf"),
            )

        return SkillStats(
            skill_name=skill_name,
            task_type=task_type,
            pull_count=int(row.pull_count),
            mean_reward=float(row.mean_reward),
            last_reward=float(row.last_reward),
            ucb_score=0.0,  # caller computes this
        )
