import { initials, posterHue, splitTitle } from '../format';

/** One catalogue item. Clicking it asks the parent to show similar items. */
export function ItemCard({ item, selected = false, onSelect, showScore = false }) {
  const { name, year } = splitTitle(item.title);
  const hue = posterHue(item.genres);

  return (
    <button
      type="button"
      className={`card${selected ? ' card-selected' : ''}`}
      aria-pressed={selected}
      onClick={() => onSelect(item)}
    >
      <span
        className="poster"
        aria-hidden="true"
        style={{ '--hue': hue }}
      >
        {initials(name)}
      </span>
      <span className="card-body">
        <span className="card-title">{name}</span>
        <span className="card-meta">
          {year && <span>{year}</span>}
          {item.rating != null && <span className="rating">★ {item.rating.toFixed(1)}</span>}
          {showScore && item.score != null && <span title="Model score (higher is better)">score {item.score.toFixed(2)}</span>}
        </span>
        {item.genres.length > 0 && (
          <span className="chips">
            {item.genres.slice(0, 3).map((genre) => (
              <span className="chip" key={genre}>
                {genre}
              </span>
            ))}
          </span>
        )}
      </span>
    </button>
  );
}
