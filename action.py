import asyncio
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore[import]
from apscheduler.triggers.cron import CronTrigger  # type: ignore[import]
from fastapi import FastAPI, HTTPException  # type: ignore[import]
from pydantic import BaseModel, Field, validator

from main import (
    NewsAnalyzer,
    calculate_news_weight,
    parse_file_titles,
    read_all_today_titles,
)

logger = logging.getLogger("trendradar.action")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="TrendRadar Service", version="1.0.0")

tz_shanghai = pytz.timezone("Asia/Shanghai")
scheduler = AsyncIOScheduler(timezone=tz_shanghai)
crawl_lock = asyncio.Lock()


class QueryRequest(BaseModel):
    query: str = Field(..., description="查询文本，支持空格分词")
    limit: int = Field(20, ge=1, le=100, description="返回结果数量上限")

    @validator("query")
    def validate_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("查询文本不能为空")
        return cleaned


@lru_cache(maxsize=1)
def _output_root() -> Path:
    return Path("output")


def _ensure_scheduler_started() -> None:
    if scheduler.running:
        return

    scheduler.add_job(
        run_crawl_job,
        CronTrigger(hour=7, minute=0, timezone=tz_shanghai),
        id="crawl_morning",
        coalesce=True,
        max_instances=1,
        replace_existing=True,
    )
    scheduler.add_job(
        run_crawl_job,
        CronTrigger(hour=13, minute=0, timezone=tz_shanghai),
        id="crawl_noon",
        coalesce=True,
        max_instances=1,
        replace_existing=True,
    )
    scheduler.add_job(
        run_crawl_job,
        CronTrigger(hour=17, minute=30, timezone=tz_shanghai),
        id="crawl_evening",
        coalesce=True,
        max_instances=1,
        replace_existing=True,
    )

    scheduler.start()
    logger.info("定时任务已启动：每天 07:00、13:00、17:30 执行一次爬取")


def _run_analyzer_sync() -> None:
    analyzer = NewsAnalyzer()
    analyzer.run()


async def run_crawl_job() -> None:
    if crawl_lock.locked():
        logger.warning("爬虫任务仍在运行，本次定时触发将跳过")
        return

    async with crawl_lock:
        logger.info("开始执行定时爬虫任务")
        try:
            await asyncio.to_thread(_run_analyzer_sync)
            logger.info("定时爬虫任务执行完毕")
        except Exception as exc:  # pragma: no cover - 日志记录
            logger.exception("定时爬虫任务执行失败：%s", exc)


def _get_latest_txt_file() -> Optional[Path]:
    output_root = _output_root()
    if not output_root.exists():
        return None

    date_dirs = sorted(
        [entry for entry in output_root.iterdir() if entry.is_dir()], reverse=True
    )
    for date_dir in date_dirs:
        txt_dir = date_dir / "txt"
        if not txt_dir.exists():
            continue
        txt_files = sorted([f for f in txt_dir.iterdir() if f.suffix == ".txt"])
        if txt_files:
            return txt_files[-1]
    return None


def _build_latest_topics_payload(
    titles_by_id: Dict[str, Dict[str, Dict]],
    id_to_name: Dict[str, str],
    batch_path: Path,
) -> Dict:
    sources: List[Dict] = []
    for source_id, title_map in titles_by_id.items():
        source_topics = []
        for title, title_data in title_map.items():
            ranks = title_data.get("ranks", [])
            primary_rank = ranks[0] if ranks else None
            source_topics.append(
                {
                    "title": title,
                    "rank": primary_rank,
                    "ranks": ranks,
                    "url": title_data.get("url") or None,
                    "mobile_url": title_data.get("mobileUrl") or None,
                }
            )

        sources.append(
            {
                "source_id": source_id,
                "source_name": id_to_name.get(source_id, source_id),
                "total": len(source_topics),
                "topics": source_topics,
            }
        )

    return {
        "date_folder": batch_path.parent.parent.name,
        "batch_time": batch_path.stem,
        "sources": sources,
    }


def _search_titles_by_query(query: str, limit: int) -> Dict[str, Any]:
    all_results, id_to_name, title_info = read_all_today_titles()

    if not all_results:
        raise HTTPException(status_code=404, detail="今日暂无热门话题数据")

    keywords = [token.lower() for token in query.split()]

    matched_items = []
    for source_id, titles_data in all_results.items():
        source_info = title_info.get(source_id, {})

        for title, base_data in titles_data.items():
            title_lower = title.lower()
            if not all(keyword in title_lower for keyword in keywords):
                continue

            enriched = source_info.get(title, {})
            ranks = enriched.get("ranks") or base_data.get("ranks", [])
            weight_payload = {
                "ranks": ranks,
                "count": enriched.get("count", len(ranks)),
            }
            weight = calculate_news_weight(weight_payload)

            matched_items.append(
                {
                    "title": title,
                    "source_id": source_id,
                    "source_name": id_to_name.get(source_id, source_id),
                    "score": weight,
                    "count": enriched.get("count", len(ranks)),
                    "first_time": enriched.get("first_time"),
                    "last_time": enriched.get("last_time"),
                    "ranks": ranks,
                    "url": enriched.get("url") or base_data.get("url") or None,
                    "mobile_url": enriched.get("mobileUrl")
                    or base_data.get("mobileUrl")
                    or None,
                }
            )

    matched_items.sort(
        key=lambda item: (
            item["score"],
            -min(item["ranks"]) if item["ranks"] else float("-inf"),
        ),
        reverse=True,
    )

    limited_items = matched_items[:limit]

    return {
        "query": query,
        "matched": len(matched_items),
        "returned": len(limited_items),
        "results": limited_items,
    }


@app.on_event("startup")
async def on_startup() -> None:
    _ensure_scheduler_started()


@app.on_event("shutdown")
async def on_shutdown() -> None:
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("定时任务已停止")


@app.get("/api/latest-topics")
async def api_latest_topics() -> Dict:
    latest_file = await asyncio.to_thread(_get_latest_txt_file)
    if not latest_file:
        raise HTTPException(status_code=404, detail="暂无热门话题数据")

    parse_result = await asyncio.to_thread(parse_file_titles, latest_file)
    titles_by_id, id_to_name = parse_result

    payload = await asyncio.to_thread(
        _build_latest_topics_payload, titles_by_id, id_to_name, latest_file
    )
    return payload


@app.post("/api/query")
async def api_query_topics(request: QueryRequest) -> Dict:
    payload = await asyncio.to_thread(
        _search_titles_by_query, request.query, request.limit
    )
    return payload


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5856)

# 部署在 43.136.88.66 /home/csm/project/TrendRadar
