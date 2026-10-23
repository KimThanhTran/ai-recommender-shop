import { useState } from 'react';
import { api } from './api';
import { splitTitle } from './format';
import { ItemGrid } from './components/ItemGrid';
import { useAsync } from './useAsync';

const MODELS = [
  { id: 'hybrid', label: 'Hybrid', hint: 'Blend of collaborative filtering (70%) and content features (30%).' },
  { id: 'item_knn', label: 'Item-based CF', hint: 'Items that people with similar ratings also liked.' },
  { id: 'content', label: 'Content-based', hint: 'Items with genres and tags close to what the user rated highly.' },
];
const RECOMMENDATIONS = 12;

export default function App() {
  const [userId, setUserId] = useState(1);
  const [draft, setDraft] = useState('1');
  const [formError, setFormError] = useState('');
  const [model, setModel] = useState('hybrid');
  const [selected, setSelected] = useState(null);

  const health = useAsync((signal) => api.health(signal), []);
  const recommendations = useAsync(
    (signal) => api.recommend(userId, { k: RECOMMENDATIONS, model }, signal),
    [userId, model]
  );
  const history = useAsync(
    (signal) =>
      api.history(userId, { limit: 6 }, signal).catch((error) => {
        if (error.status === 404) return { items: [] }; // unknown user: simply no history
        throw error;
      }),
    [userId]
  );
  const popular = useAsync((signal) => api.popular({ k: RECOMMENDATIONS }, signal), []);
  const similar = useAsync(
    (signal) => (selected ? api.similar(selected.item_id, { k: 8, model }, signal) : Promise.resolve(null)),
    [selected, model]
  );

  const showUser = (event) => {
    event.preventDefault();
    const parsed = Number(draft);
    if (!Number.isInteger(parsed) || parsed < 1) {
      setFormError('Enter a whole number of 1 or more.');
      return;
    }
    setFormError('');
    setUserId(parsed);
    setSelected(null);
  };

  const pickRandomUser = () => {
    const total = health.data?.users ?? 610;
    const next = 1 + Math.floor(Math.random() * total);
    setDraft(String(next));
    setFormError('');
    setUserId(next);
    setSelected(null);
  };

  const toggleSelected = (item) => setSelected((current) => (current?.item_id === item.item_id ? null : item));
  const coldStart = recommendations.data && recommendations.data.personalised === false;

  return (
    <div className="page">
      <header className="header">
        <div>
          <h1>AI Recommender Shop</h1>
          <p className="tagline">Personalised picks from collaborative filtering and content features.</p>
        </div>
        <p className={`pill ${health.error ? 'pill-bad' : 'pill-ok'}`} role="status">
          {health.error
            ? 'API offline'
            : health.data
              ? `${health.data.users} users · ${health.data.items.toLocaleString('en-US')} items`
              : 'Connecting…'}
        </p>
      </header>

      {health.error && (
        <p className="message message-error" role="alert">
          Cannot reach the API. Start it with <code>uvicorn app.main:app</code> (the dev server proxies{' '}
          <code>/api</code> to port 8000).
        </p>
      )}

      <section className="panel controls" aria-label="Recommendation settings">
        <form className="user-form" onSubmit={showUser} noValidate>
          <label htmlFor="user-id">User ID</label>
          <div className="row">
            <input
              id="user-id"
              type="number"
              inputMode="numeric"
              min="1"
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              aria-invalid={Boolean(formError)}
              aria-describedby={formError ? 'user-error' : undefined}
            />
            <button type="submit" className="button">
              Show recommendations
            </button>
            <button type="button" className="button button-ghost" onClick={pickRandomUser}>
              Random user
            </button>
          </div>
          {formError && (
            <p id="user-error" className="field-error" role="alert">
              {formError}
            </p>
          )}
          <p className="hint">The demo data has users 1 to {health.data?.users ?? 610}. Any other ID is treated as a new user.</p>
        </form>

        <fieldset className="models">
          <legend>Model</legend>
          <div className="segmented">
            {MODELS.map((option) => (
              <label key={option.id} className={model === option.id ? 'segment segment-on' : 'segment'}>
                <input
                  type="radio"
                  name="model"
                  value={option.id}
                  checked={model === option.id}
                  onChange={() => setModel(option.id)}
                />
                {option.label}
              </label>
            ))}
          </div>
          <p className="hint">{MODELS.find((option) => option.id === model).hint}</p>
        </fieldset>
      </section>

      <div className="layout">
        <main className="main">
          <section aria-labelledby="recs-title">
            <h2 id="recs-title">
              {coldStart ? 'Popular right now' : `Recommended for user ${userId}`}
            </h2>
            {coldStart && (
              <p className="message">
                User {userId} has no rating history, so there is nothing to personalise yet and popular items are
                shown instead (cold start).
              </p>
            )}
            <ItemGrid
              items={recommendations.data?.items}
              loading={recommendations.loading}
              error={recommendations.error}
              onSelect={toggleSelected}
              selectedId={selected?.item_id}
              showScore={!coldStart}
              skeletons={RECOMMENDATIONS}
              emptyText="No recommendations for this user."
            />
          </section>

          {history.data?.items.length > 0 && (
            <section aria-labelledby="history-title">
              <h2 id="history-title">Because user {userId} rated these highly</h2>
              <ItemGrid
                items={history.data.items}
                loading={history.loading}
                error={history.error}
                onSelect={toggleSelected}
                selectedId={selected?.item_id}
                skeletons={6}
              />
            </section>
          )}

          <section aria-labelledby="popular-title">
            <h2 id="popular-title">Most rated overall</h2>
            <ItemGrid
              items={popular.data?.items}
              loading={popular.loading}
              error={popular.error}
              onSelect={toggleSelected}
              selectedId={selected?.item_id}
              skeletons={6}
            />
          </section>
        </main>

        <aside className="aside" aria-label="Similar items">
          {selected ? (
            <div className="panel sticky">
              <div className="aside-head">
                <h2>Similar to {splitTitle(selected.title).name}</h2>
                <button type="button" className="button button-ghost" onClick={() => setSelected(null)}>
                  Close
                </button>
              </div>
              <ItemGrid
                items={similar.data?.items}
                loading={similar.loading}
                error={similar.error}
                onSelect={toggleSelected}
                selectedId={selected.item_id}
                showScore
                skeletons={4}
                emptyText="No similar items found."
              />
            </div>
          ) : (
            <p className="message">Select any item to see similar ones ("customers also viewed").</p>
          )}
        </aside>
      </div>

      <footer className="footer">
        Data: MovieLens latest-small (GroupLens). Movies stand in for products; posters are placeholders.
      </footer>
    </div>
  );
}
