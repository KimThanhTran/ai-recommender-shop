"""FastAPI service exposing the recommenders.

    uvicorn app.main:app --reload
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request

from recsys import load_movielens

from .service import ItemNotFound, RecommenderService

MAX_K = 50


class PersonalModel(str, Enum):
    hybrid = "hybrid"
    item_knn = "item_knn"
    content = "content"


def create_app(service: RecommenderService | None = None) -> FastAPI:
    """Build the app. Pass a ready ``service`` (tests do); otherwise the dataset in $DATA_DIR is loaded on startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is not None:
            app.state.service = service
        else:
            data_dir = Path(os.environ.get("DATA_DIR", "data"))
            if not (data_dir / "ratings.csv").exists():
                raise RuntimeError(
                    f"No dataset in {data_dir.resolve()}. Run: python scripts/download_data.py"
                )
            dataset = load_movielens(data_dir)
            app.state.service = RecommenderService(dataset.ratings, dataset.items)
        yield

    app = FastAPI(
        title="AI Recommender Shop",
        description="Product recommendations from collaborative filtering and content features.",
        version="1.0.0",
        lifespan=lifespan,
    )

    def svc(request: Request) -> RecommenderService:
        return request.app.state.service

    @app.get("/health", tags=["meta"])
    def health(request: Request):
        return {"status": "ok", **svc(request).stats}

    @app.get("/items/{item_id}", tags=["catalogue"])
    def get_item(item_id: int, request: Request):
        try:
            return svc(request).get_item(item_id)
        except ItemNotFound:
            raise HTTPException(status_code=404, detail=f"Item {item_id} not found")

    @app.get("/items/{item_id}/similar", tags=["recommendations"])
    def similar_items(
        item_id: int,
        request: Request,
        k: int = Query(10, ge=1, le=MAX_K),
        model: PersonalModel = PersonalModel.hybrid,
    ):
        """Items most similar to the given one ("customers also viewed")."""
        try:
            items = svc(request).similar(item_id, k, model.value)
        except ItemNotFound:
            raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
        return {"item_id": item_id, "model": model.value, "items": items}

    @app.get("/recommend/{user_id}", tags=["recommendations"])
    def recommend(
        user_id: int,
        request: Request,
        k: int = Query(10, ge=1, le=MAX_K),
        model: PersonalModel = PersonalModel.hybrid,
    ):
        """Top-k items for a user. Unknown users get popular items (`personalised: false`)."""
        result = svc(request).recommend(user_id, k, model.value)
        return {
            "user_id": user_id,
            "model": result.model,
            "personalised": result.personalised,
            "items": result.items,
        }

    @app.get("/popular", tags=["recommendations"])
    def popular(request: Request, k: int = Query(10, ge=1, le=MAX_K)):
        return {"items": svc(request).popular(k)}

    @app.get("/users/{user_id}/history", tags=["catalogue"])
    def history(user_id: int, request: Request, limit: int = Query(10, ge=1, le=MAX_K)):
        """The user's highest-rated items, useful to explain a recommendation."""
        items = svc(request).history(user_id, limit)
        if items is None:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        return {"user_id": user_id, "items": items}

    return app


app = create_app()
