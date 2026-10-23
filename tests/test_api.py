import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.service import RecommenderService


@pytest.fixture
def client(ratings, items):
    service = RecommenderService(ratings, items)
    with TestClient(create_app(service)) as test_client:
        yield test_client


def test_health_reports_catalogue_size(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "users": 7, "items": 9}


def test_recommend_known_user_is_personalised(client):
    body = client.get("/recommend/1", params={"k": 1}).json()
    assert body["personalised"] is True
    assert body["model"] == "hybrid"
    assert [item["item_id"] for item in body["items"]] == [4]
    assert body["items"][0]["title"] == "Nebula Wars (2005)"
    assert body["items"][0]["genres"] == ["Sci-Fi"]


@pytest.mark.parametrize("model", ["hybrid", "item_knn", "content"])
def test_every_model_is_selectable(client, model):
    response = client.get("/recommend/1", params={"model": model, "k": 3})
    assert response.status_code == 200
    assert response.json()["model"] == model


def test_unknown_user_falls_back_to_popular_items(client):
    body = client.get("/recommend/12345", params={"k": 3}).json()
    assert body["personalised"] is False
    assert body["model"] == "popularity"
    assert len(body["items"]) == 3


def test_short_lists_are_topped_up_with_unseen_popular_items(client):
    # user 7 rated one item only: a model can return little, the API still fills k slots
    body = client.get("/recommend/7", params={"k": 5}).json()
    ids = [item["item_id"] for item in body["items"]]
    assert len(ids) == 5
    assert len(set(ids)) == 5
    assert 1 not in ids  # already rated


def test_k_is_validated(client):
    assert client.get("/recommend/1", params={"k": 0}).status_code == 422
    assert client.get("/recommend/1", params={"k": 500}).status_code == 422
    assert client.get("/recommend/1", params={"model": "magic"}).status_code == 422


def test_similar_items(client):
    body = client.get("/items/1/similar", params={"k": 3, "model": "content"}).json()
    assert {item["item_id"] for item in body["items"]} == {2, 3, 4}
    assert client.get("/items/999/similar").status_code == 404


def test_item_lookup(client):
    assert client.get("/items/2").json()["title"] == "Galaxy Run (2001)"
    assert client.get("/items/999").status_code == 404


def test_popular_lists_most_rated_items(client):
    items = client.get("/popular", params={"k": 2}).json()["items"]
    assert len(items) == 2
    assert items[0]["item_id"] in {1, 2}  # the most-rated titles


def test_history_returns_top_rated_first(client):
    body = client.get("/users/1/history", params={"limit": 3}).json()
    assert [item["rating"] for item in body["items"]] == [5.0, 5.0, 5.0]
    assert client.get("/users/999/history").status_code == 404
